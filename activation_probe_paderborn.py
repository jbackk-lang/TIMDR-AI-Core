"""v0.7 single-layer activation test on real Paderborn bearing measurements.

The holdout entry point refuses access until the separate preregistration
is present in HEAD. This module never produces a TIMDRProtocol verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np

from activation_diagnostics_v03 import nearest_rank, transition_signature
from neural_network import MLPClassifier
from paderborn_extractor import _dependencies


PLAN = "prereg/ACTIVATION_PROBE_v0.7_PADERBORN_PLAN.md"
SPLIT_MANIFEST = "prereg/PADERBORN_MS_REPLICATION_v0.1.json"
SAMPLE_RATE = 64000
WINDOW = 32000
CLASS = {"K001.rar": 0, "KI01.rar": 1, "KA01.rar": 2}
BANDS = ((0, 500), (500, 2000), (2000, 8000), (8000, 16000), (16000, 32001))


class V05Error(RuntimeError):
    pass


def _raw_vibration(root: Path, archive: str, member: str) -> np.ndarray:
    _, loadmat = _dependencies()
    source = root / "data" / "paderborn_candidate" / "raw" / archive
    try:
        raw = subprocess.check_output(["tar", "-xOf", str(source), member], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise V05Error(f"Cannot read {archive}:{member}") from exc
    payload = loadmat(BytesIO(raw), squeeze_me=True, struct_as_record=False)
    names = [name for name in payload if not name.startswith("__")]
    if len(names) != 1:
        raise V05Error(f"Unexpected MATLAB schema in {member}.")
    channels = payload[names[0]].Y
    vibration = next((channel.Data for channel in channels if str(channel.Name) == "vibration_1"), None)
    if vibration is None:
        raise V05Error(f"Missing vibration_1 in {member}.")
    return np.asarray(vibration, dtype=float).reshape(-1)


def _vibration_v07(root: Path, archive: str, member: str) -> tuple[np.ndarray, int]:
    values = _raw_vibration(root, archive, member)
    if not 255998 <= len(values) <= 320000:
        raise V05Error(f"Unexpected vibration length {len(values)} in {member}.")
    adjustment = len(values) - 256000
    if adjustment < 0:
        return np.pad(values, (0, -adjustment)), adjustment
    return values[:256000], adjustment


def audit_train_calibration_lengths(root: Path) -> dict:
    """Metadata-only audit of already authorized splits; never names holdout."""
    manifest = json.loads((root / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    answer = {}
    for split in ("train", "calibration"):
        answer[split] = {}
        for item in manifest["split"]["members"][split]:
            values = _raw_vibration(root, item["archive"], item["member"])
            answer[split][item["member"]] = int(len(values))
    return answer


def committed_plan_matches(root: Path) -> bool:
    """Require the exact plan bytes in HEAD before any holdout payload read."""
    current = (root / PLAN).read_bytes()
    try:
        frozen = subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:{PLAN}"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return current == frozen


def window_features(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.shape != (WINDOW,) or not np.isfinite(values).all():
        raise V05Error("One finite, 0.5-second vibration window is required.")
    rms = float(np.sqrt(np.mean(values * values)))
    crest = float(np.max(np.abs(values)) / rms) if rms > 0 else 0.0
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered * centered))
    kurtosis = float(np.mean(centered ** 4) / (variance * variance)) if variance > 0 else 0.0
    spectrum = np.abs(np.fft.rfft(values)) ** 2 / (WINDOW * WINDOW)
    frequencies = np.fft.rfftfreq(WINDOW, d=1 / SAMPLE_RATE)
    band_power = [float(np.sum(spectrum[(frequencies >= low) & (frequencies < high)])) for low, high in BANDS]
    result = np.log1p([rms, crest, kurtosis, *band_power])
    if result.shape != (8,) or not np.isfinite(result).all():
        raise V05Error("Feature extraction produced invalid values.")
    return result


@dataclass(frozen=True)
class Measurement:
    member: str
    label: int
    features: np.ndarray  # chronological (8, 8)
    tail_adjustment_samples: int


def load_split(root: Path, split: str) -> list[Measurement]:
    if split not in ("train", "calibration", "holdout"):
        raise V05Error("Unknown split.")
    if split == "holdout":
        if not committed_plan_matches(root):
            raise V05Error("Holdout locked: exact v0.7 plan must be committed in HEAD first.")
        if not calibration_report(root)["calibration_power"]["passed"]:
            raise V05Error("Holdout locked: calibration power gate did not pass.")
    manifest = json.loads((root / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_WAVEFORM_EXTRACTION":
        raise V05Error("Dataset split is not frozen.")
    items = manifest["split"]["members"][split]
    seen = set()
    measurements = []
    for item in items:
        archive, member = item["archive"], item["member"]
        if archive not in CLASS or member in seen:
            raise V05Error("Unexpected archive or duplicate member.")
        seen.add(member)
        signal, adjustment = _vibration_v07(root, archive, member)
        if len(signal) != 8 * WINDOW:
            raise V05Error("Unexpected measurement length.")
        features = np.stack([window_features(signal[i * WINDOW:(i + 1) * WINDOW]) for i in range(8)])
        measurements.append(Measurement(member, CLASS[archive], features, adjustment))
    return measurements


@dataclass(frozen=True)
class FittedProbe:
    model: MLPClassifier
    mean: np.ndarray
    scale: np.ndarray
    train_accuracy: float

    def standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def fit_probe(train: list[Measurement]) -> FittedProbe:
    x = np.vstack([measurement.features for measurement in train])
    y = np.concatenate([np.full(8, measurement.label) for measurement in train])
    if x.shape != (1152, 8) or set(y) != {0, 1, 2}:
        raise V05Error("Frozen train count/classes do not match the plan.")
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    model = MLPClassifier(8, 12, 3, seed=510)
    model.fit(z, y, epochs=300, lr=0.05)
    accuracy = float(np.mean(model.predict(z) == y))
    return FittedProbe(model, mean, scale, accuracy)


def transition_rows(probe: FittedProbe, measurements: list[Measurement]) -> list[dict]:
    rows = []
    for measurement in measurements:
        z = probe.standardize(measurement.features)
        hidden = probe.model.hidden_activations(z)
        probabilities = probe.model.predict_proba(z)
        for step in range(1, 8):
            signature = transition_signature(hidden[step - 1:step + 1])
            probabilities_now = probabilities[step]
            rows.append({
                "member": measurement.member,
                "step": step,
                "error": int(np.argmax(probabilities_now) != measurement.label),
                "shape_js": signature.shape_js,
                "delta_rms": signature.delta_rms,
                "uncertainty": float(1 - np.max(probabilities_now)),
            })
    return rows


def power_counts(rows: list[dict]) -> dict[str, int | bool]:
    errors = [row for row in rows if row["error"]]
    correct = [row for row in rows if not row["error"]]
    error_members = len({row["member"] for row in errors})
    correct_members = len({row["member"] for row in correct})
    return {
        "errors": len(errors), "correct": len(correct),
        "error_members": error_members, "correct_members": correct_members,
        "passed": len(errors) >= 30 and len(correct) >= 30 and error_members >= 5 and correct_members >= 5,
    }


def calibration_report(root: Path) -> dict:
    """Read train and calibration only; no call path to holdout."""
    train = load_split(root, "train")
    probe = fit_probe(train)
    calibration = load_split(root, "calibration")
    rows = transition_rows(probe, calibration)
    power = power_counts(rows)
    return {
        "status": "READY_FOR_FROZEN_HOLDOUT" if power["passed"] else "INCONCLUSIVE_INSUFFICIENT_CALIBRATION_POWER",
        "train_members": len(train), "calibration_members": len(calibration),
        "train_accuracy": probe.train_accuracy,
        "calibration_power": power,
        "tail_adjustment_by_member": {row.member: row.tail_adjustment_samples for row in train + calibration},
        "holdout_accessed": False,
    }


def calibration_threshold(rows: list[dict]) -> float:
    correct = [row["shape_js"] for row in rows if not row["error"]]
    if not correct:
        raise V05Error("No correct calibration transitions.")
    return nearest_rank(correct, 0.95)
