"""Fixed waveform features for the separately frozen PROTECT-90 hypothesis.

Only an explicitly supplied train/calibration subset may be opened. The module
does not expose an evaluation function for the holdout set.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from environment_policy import assert_episode_size
from protect90_learning_adapter import Protect90Error, WAVEFORM_FEATURES, _read_json, _source_paths


class WaveformDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WaveformTrainingRun:
    classes: tuple[int, ...]
    train_count: int
    calibration_count: int
    holdout_accessed: bool
    feature_count: int
    calibration_accuracy: float


def _libraries():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise WaveformDependencyError(
            "Waveform baseline requires numpy and pandas. Install with: "
            ".venv\\Scripts\\python.exe -m pip install -e \".[waveform]\""
        ) from exc
    return np, pd


def _episode_path(source_root: str | Path, sample_id: int) -> Path:
    path = Path(source_root) / "data" / "protect90" / "frozen" / "preprocessed_data" / f"{int(sample_id)}_sample_hv_double_line_90kv.pkl"
    if not path.is_file():
        raise Protect90Error(f"Missing frozen waveform episode {sample_id}: {path}")
    assert_episode_size(path)
    return path


def extract_features(source_root: str | Path, sample_id: int) -> tuple[float, ...]:
    """Return the pre-registered 16 normalized RMS features for one waveform."""
    np, pd = _libraries()
    frame = pd.read_pickle(_episode_path(source_root, sample_id))
    locations = sorted(column[:-len("_vol_L1_V")] for column in frame.columns if column.endswith("_vol_L1_V"))
    if len(locations) != 8:
        raise Protect90Error(f"Expected eight voltage locations, found {len(locations)} in episode {sample_id}.")
    block, baseline_blocks = int(WAVEFORM_FEATURES["rms_block_samples"]), 20
    features: list[float] = []
    for location in locations:
        current = [frame[f"{location}_cur_L{phase}_A"].to_numpy(float) for phase in (1, 2, 3)]
        voltage = [frame[f"{location}_vol_L{phase}_V"].to_numpy(float) for phase in (1, 2, 3)]
        def block_rms(values):
            n = len(values) // block * block
            if n < baseline_blocks * block:
                raise Protect90Error(f"Episode {sample_id} is too short for fixed RMS extraction.")
            return np.sqrt(np.mean(values[:n].reshape(-1, block) ** 2, axis=1))
        current_rms = np.mean(np.vstack([block_rms(values) for values in current]), axis=0)
        voltage_rms = np.mean(np.vstack([block_rms(values) for values in voltage]), axis=0)
        current_baseline, voltage_baseline = float(np.mean(current_rms[:baseline_blocks])), float(np.mean(voltage_rms[:baseline_blocks]))
        current_feature = 0.0 if current_baseline <= 1e-12 else float(np.max(current_rms) / current_baseline)
        voltage_feature = 0.0 if voltage_baseline <= 1e-12 else float(np.min(voltage_rms) / voltage_baseline)
        features.extend((current_feature, voltage_feature))
    if len(features) != int(WAVEFORM_FEATURES["expected_feature_count"]) or not all(isfinite(value) for value in features):
        raise Protect90Error(f"Invalid waveform feature vector for episode {sample_id}.")
    return tuple(features)


def run_train_calibration(source_root: str | Path, prereg_path: str | Path) -> WaveformTrainingRun:
    """Run the pre-authorized stage without loading any holdout waveform file."""
    prereg = _read_json(Path(prereg_path))
    if prereg.get("version") != "PROTECT90_WAVEFORM_MULTICLASS_v0.3":
        raise Protect90Error("Not the frozen waveform preregistration.")
    _, _, labels_path = _source_paths(source_root)
    import csv
    with labels_path.open(encoding="utf-8", newline="") as handle:
        labels = {int(row["sample_id"]): int(row["sc_type"]) for row in csv.DictReader(handle)}
    splits = prereg["split"]["episode_ids"]
    if set(splits) != {"train", "calibration", "holdout"}:
        raise Protect90Error("Unexpected split names.")
    train = [(extract_features(source_root, sample_id), labels[int(sample_id)]) for sample_id in splits["train"]]
    calibration = [(extract_features(source_root, sample_id), labels[int(sample_id)]) for sample_id in splits["calibration"]]
    np, _ = _libraries()
    x_train, y_train = np.asarray([row[0] for row in train], dtype=float), np.asarray([row[1] for row in train], dtype=int)
    means, scales = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (x_train - means) / scales
    classes = tuple(sorted(set(int(label) for label in y_train)))
    if classes != (0, 1, 2, 3):
        raise Protect90Error(f"Unexpected training classes: {classes}")
    centroids = {label: np.mean(standardized[y_train == label], axis=0) for label in classes}
    correct = 0
    for features, label in calibration:
        point = (np.asarray(features) - means) / scales
        predicted = min(classes, key=lambda item: (float(np.sum((point - centroids[item]) ** 2)), item))
        correct += int(predicted == label)
    return WaveformTrainingRun(classes, len(train), len(calibration), False, x_train.shape[1], correct / len(calibration))
