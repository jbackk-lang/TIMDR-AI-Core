"""A small MLP (numpy-only) baseline for the SAME frozen PROTECT-90 waveform
hypothesis as protect90_waveform_learning.py's nearest-centroid baseline.

Same dataset, same frozen split, same holdout wall: this module never opens
or reports on the holdout episodes. Its output is a research candidate, not
a TIMDR result -- exactly the same firewall pattern as
learning_sandbox.py's CandidateProposal (requires_human_preregistration).
Nothing here can set a TIMDRProtocol verdict.

Hyperparameters are frozen below BEFORE running against calibration data and
are not tuned after seeing a result -- same discipline as a preregistered
parameter, even though this module itself is pre-verdict research code.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from neural_network import MLPClassifier
from protect90_learning_adapter import Protect90Error, _read_json, _source_paths
from protect90_waveform_learning import WaveformDependencyError, extract_features

# Frozen before running on calibration data; not tuned after seeing a result.
HIDDEN_UNITS = 12
EPOCHS = 800
LEARNING_RATE = 0.05
SEED = 0


@dataclass(frozen=True)
class WaveformNeuralRun:
    classes: tuple[int, ...]
    train_count: int
    calibration_count: int
    holdout_accessed: bool
    feature_count: int
    hidden_units: int
    epochs: int
    learning_rate: float
    seed: int
    calibration_accuracy: float
    final_train_loss: float


def _libraries():
    try:
        import numpy as np
    except ImportError as exc:
        raise WaveformDependencyError(
            "Neural baseline requires numpy. Install with: "
            '.venv\\Scripts\\python.exe -m pip install -e ".[waveform]"'
        ) from exc
    return np


def run_train_calibration(source_root: str | Path, prereg_path: str | Path) -> WaveformNeuralRun:
    """Run the pre-authorized stage without loading any holdout waveform file."""
    prereg = _read_json(Path(prereg_path))
    if prereg.get("version") != "PROTECT90_WAVEFORM_MULTICLASS_v0.3":
        raise Protect90Error("Not the frozen waveform preregistration.")
    _, _, labels_path = _source_paths(source_root)
    with labels_path.open(encoding="utf-8", newline="") as handle:
        labels = {int(row["sample_id"]): int(row["sc_type"]) for row in csv.DictReader(handle)}
    splits = prereg["split"]["episode_ids"]
    if set(splits) != {"train", "calibration", "holdout"}:
        raise Protect90Error("Unexpected split names.")

    train = [(extract_features(source_root, sid), labels[int(sid)]) for sid in splits["train"]]
    calibration = [(extract_features(source_root, sid), labels[int(sid)]) for sid in splits["calibration"]]

    np = _libraries()
    x_train = np.asarray([row[0] for row in train], dtype=float)
    y_train = np.asarray([row[1] for row in train], dtype=int)
    x_cal = np.asarray([row[0] for row in calibration], dtype=float)
    y_cal = np.asarray([row[1] for row in calibration], dtype=int)

    means, scales = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    x_train_std = (x_train - means) / scales
    x_cal_std = (x_cal - means) / scales

    classes = tuple(sorted(set(int(label) for label in y_train)))
    if classes != (0, 1, 2, 3):
        raise Protect90Error(f"Unexpected training classes: {classes}")

    net = MLPClassifier(n_features=x_train_std.shape[1], n_hidden=HIDDEN_UNITS, n_classes=len(classes), seed=SEED)
    history = net.fit(x_train_std, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
    calibration_accuracy = float(np.mean(net.predict(x_cal_std) == y_cal))

    return WaveformNeuralRun(
        classes=classes,
        train_count=len(train),
        calibration_count=len(calibration),
        holdout_accessed=False,
        feature_count=x_train.shape[1],
        hidden_units=HIDDEN_UNITS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        calibration_accuracy=calibration_accuracy,
        final_train_loss=float(history.losses[-1]),
    )
