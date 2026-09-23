"""Calibration-only v0.9 probe on UCI HAR; never opens official test members.

Plan: prereg/ACTIVATION_PROBE_v0.9_UCI_HAR_PLAN.md. The data are real
sensor windows; activations come from a separately fitted neural network.
This is not a confirmatory TIMDR result.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from activation_diagnostics_v03 import auc_greater, transition_signature
from neural_network import MLPClassifier

INNER_NAME = "UCI HAR Dataset.zip"
TRAIN_NAMES = (
    "UCI HAR Dataset/train/X_train.txt",
    "UCI HAR Dataset/train/y_train.txt",
    "UCI HAR Dataset/train/subject_train.txt",
)
SOURCE_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"


def load_train_only(archive_path: Path):
    """Explicit member whitelist: no test member is opened or extracted."""
    with zipfile.ZipFile(archive_path) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(INNER_NAME))) as inner:
            arrays = [np.loadtxt(inner.open(name)) for name in TRAIN_NAMES]
    X, y, subjects = arrays
    y = y.astype(int)
    subjects = subjects.astype(int)
    if X.ndim != 2 or X.shape[1] != 561 or len(X) != len(y) or len(X) != len(subjects):
        raise ValueError("Unexpected UCI HAR train schema; no analysis performed.")
    if not np.isfinite(X).all() or set(np.unique(y)) != set(range(1, 7)):
        raise ValueError("Non-finite features or unexpected train classes.")
    return X, y, subjects


def split_subjects(subjects):
    unique = sorted(set(int(value) for value in subjects))
    ranked = sorted(unique, key=lambda sid: hashlib.sha256(
        f"uci-har-v08-subject-{sid}".encode("ascii")
    ).digest())
    calibration = set(ranked[:5])
    fit = set(unique) - calibration
    return fit, calibration


def adjacent_pairs(subjects, labels, selected):
    """No transition crosses row discontinuity, person, or activity."""
    chosen = np.isin(subjects, list(selected))
    return np.flatnonzero(
        chosen[1:] & chosen[:-1]
        & (subjects[1:] == subjects[:-1])
        & (labels[1:] == labels[:-1])
    ) + 1


def _auc(rows, key):
    return auc_greater(
        [r[key] for r in rows if r["wrong"]],
        [r[key] for r in rows if not r["wrong"]],
    )


def _control():
    same = transition_signature([[1, 0], [1, 0]]).shape_js
    changed = transition_signature([[1, 0], [0, 1]]).shape_js
    return bool(same == 0.0 and changed > 0.0 and np.isfinite(changed))


def run(archive_path: Path):
    if not _control():
        raise RuntimeError("Technical activation control failed.")
    X, y, subjects = load_train_only(archive_path)
    fit_subjects, calibration_subjects = split_subjects(subjects)
    fit_mask = np.isin(subjects, list(fit_subjects))
    cal_mask = np.isin(subjects, list(calibration_subjects))
    center = np.mean(X[fit_mask], axis=0)
    scale = np.std(X[fit_mask], axis=0)
    scale[scale == 0] = 1.0
    X_fit = (X[fit_mask] - center) / scale
    X_cal = (X[cal_mask] - center) / scale
    model = MLPClassifier(n_features=561, n_hidden=64, n_classes=6, seed=510)
    history = model.fit(X_fit, y[fit_mask] - 1, epochs=500, lr=0.05)
    hidden = model.hidden_activations(X_cal)
    probabilities = model.predict_proba(X_cal)
    predictions = np.argmax(probabilities, axis=1) + 1
    cal_indices = np.flatnonzero(cal_mask)
    position = {int(original): current for current, original in enumerate(cal_indices)}
    pair_indices = adjacent_pairs(subjects, y, calibration_subjects)
    rows = []
    for current in pair_indices:
        before, after = position[int(current - 1)], position[int(current)]
        signature = transition_signature(hidden[[before, after]])
        rows.append({
            "subject": int(subjects[current]),
            "wrong": int(predictions[after] != y[current]),
            "shape_js": signature.shape_js,
            "delta_rms": signature.delta_rms,
            "uncertainty": float(1.0 - np.max(probabilities[after])),
        })
    errors = [r for r in rows if r["wrong"]]
    correct = [r for r in rows if not r["wrong"]]
    eligible = bool(
        len(errors) >= 30 and len(correct) >= 30
        and len({r["subject"] for r in errors}) >= 3
        and len({r["subject"] for r in correct}) >= 3
    )
    report = {
        "status": "CALIBRATION_ONLY" if eligible else "INCONCLUSIVE",
        "reason": None if eligible else "Insufficient errors/correct pairs or subject coverage on calibration.",
        "source_url": SOURCE_URL,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "official_test_members_opened": False,
        "fit_subjects": sorted(fit_subjects),
        "calibration_subjects": sorted(calibration_subjects),
        "fit_windows": int(np.sum(fit_mask)),
        "calibration_windows": int(np.sum(cal_mask)),
        "calibration_pairs": len(rows),
        "errors": len(errors), "correct": len(correct),
        "error_subjects": len({r["subject"] for r in errors}),
        "correct_subjects": len({r["subject"] for r in correct}),
        "calibration_window_accuracy": float(np.mean(predictions == y[cal_mask])),
        "fit_window_accuracy": float(np.mean(model.predict(X_fit) == y[fit_mask] - 1)),
        "model_iterations": 500,
        "fit_final_loss": history.losses[-1],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "technical_control_passed": True,
        "auc": None,
        "delta_quartiles": None,
    }
    if eligible:
        report["auc"] = {name: _auc(rows, name) for name in ("shape_js", "delta_rms", "uncertainty")}
        values = np.array([r["delta_rms"] for r in rows])
        edges = np.quantile(values, [0, .25, .5, .75, 1])
        bins = np.searchsorted(edges[1:-1], values, side="right")
        quartiles = []
        for q in range(4):
            subset = [row for row, index in zip(rows, bins) if index == q]
            n_error = sum(r["wrong"] for r in subset)
            n_correct = len(subset) - n_error
            quartiles.append({
                "bin": q + 1, "pairs": len(subset),
                "errors": n_error, "correct": n_correct,
                "shape_js_auc": _auc(subset, "shape_js") if n_error >= 10 and n_correct >= 10 else None,
            })
        report["delta_quartiles"] = {"edges": edges.tolist(), "bins": quartiles}
    return report


if __name__ == "__main__":
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\jback\Downloads\a\data\uci_har_240.zip")
    print(json.dumps(run(archive), ensure_ascii=False, indent=2))
