"""Fresh-seed, synthetic v0.2 comparison; not a real-data TIMDR verdict.

Fixed before first run: train/calibration/test RNG seeds 41/42/43;
200 calibration and 400 test trajectories; two input-noise scales .35/.70;
v0.1, v0.2 global and v0.2 difficulty-stratified all use q=.95.
The last perturbation (if any) is invisible to the difficulty context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activation_diagnostics import fit_healthy_reference  # noqa: E402
from activation_diagnostics_v02 import fit_context_reference  # noqa: E402
from examples.activation_probe_demo import CENTERS, STEPS, _auc  # noqa: E402
from neural_network import MLPClassifier  # noqa: E402


def _sequence(rng: np.random.Generator, *, disturbed: bool, difficult: bool) -> tuple[np.ndarray, int]:
    label = int(rng.integers(0, len(CENTERS)))
    noise = 0.70 if difficult else 0.35
    x = CENTERS[label] + rng.normal(0.0, noise, size=(STEPS, 2))
    if disturbed:
        other = (label + 1) % len(CENTERS)
        x[-1] += float(rng.uniform(0.3, 1.25)) * (CENTERS[other] - CENTERS[label])
    return x, label


def _rate(alarms: list[bool], indices: list[int]) -> float | None:
    return sum(alarms[i] for i in indices) / len(indices) if indices else None


def run_v02() -> dict[str, float | int | None]:
    train_rng = np.random.default_rng(41)
    x_train = np.vstack([center + train_rng.normal(0.0, 0.45, size=(120, 2)) for center in CENTERS])
    y_train = np.repeat(np.arange(3), 120)
    model = MLPClassifier(2, 12, 3, seed=41)
    model.fit(x_train, y_train, epochs=250, lr=0.1)

    cal_rng = np.random.default_rng(42)
    calibration = [_sequence(cal_rng, disturbed=False, difficult=i % 2 == 1) for i in range(200)]
    calibration_correct = [(x, y) for x, y in calibration if int(model.predict(x[-1:])[0]) == y]
    cal_inputs = [x for x, _ in calibration_correct]
    cal_activations = [model.hidden_activations(x) for x in cal_inputs]
    old_reference = fit_healthy_reference(cal_activations, false_alarm_quantile=0.95)
    new_reference = fit_context_reference(cal_activations, cal_inputs, false_alarm_quantile=0.95)

    test_rng = np.random.default_rng(43)
    cases = [_sequence(test_rng, disturbed=i >= 200, difficult=i % 2 == 1) for i in range(400)]
    errors: list[bool] = []
    scores: dict[str, list[float]] = {name: [] for name in ("v01", "v02_global", "v02_context", "confidence")}
    alarms: dict[str, list[bool]] = {name: [] for name in ("v01", "v02_global", "v02_context")}
    contexts: list[str] = []
    for x, y in cases:
        probs = model.predict_proba(x[-1:])[0]
        errors.append(int(np.argmax(probs)) != y)
        hidden = model.hidden_activations(x)
        context = new_reference.context(x)
        contexts.append(context)
        old_score = old_reference.score(hidden)
        global_score, context_score = new_reference.score(hidden, context=context)
        old_alarm = old_score > old_reference.threshold
        global_alarm, context_alarm = new_reference.alarms(hidden, context=context)
        for name, score, alarm in (
            ("v01", old_score, old_alarm),
            ("v02_global", global_score, global_alarm),
            ("v02_context", context_score, context_alarm),
        ):
            scores[name].append(score)
            alarms[name].append(alarm)
        scores["confidence"].append(float(1.0 - np.max(probs)))

    clean_correct = [i for i in range(200) if not errors[i]]
    hard_correct = [i for i in range(200, 400) if not errors[i]]
    disturbed_errors = [i for i in range(200, 400) if errors[i]]
    report: dict[str, float | int | None] = {
        "calibration_correct": len(calibration_correct),
        "test_count": len(cases),
        "test_errors": sum(errors),
        "clean_correct_count": len(clean_correct),
        "hard_correct_count": len(hard_correct),
        "disturbed_error_count": len(disturbed_errors),
        "context_easy_count": contexts.count("easy"),
        "context_hard_count": contexts.count("hard"),
        "confidence_auc": _auc(errors, scores["confidence"]),
    }
    for name in alarms:
        report[f"{name}_auc"] = _auc(errors, scores[name])
        report[f"{name}_clean_fpr"] = _rate(alarms[name], clean_correct)
        report[f"{name}_hard_correct_fpr"] = _rate(alarms[name], hard_correct)
        report[f"{name}_error_sensitivity"] = _rate(alarms[name], disturbed_errors)
    return report


if __name__ == "__main__":
    print("SYNTHETIC V0.2 - exploratory, no TIMDR or hallucination verdict")
    for key, value in run_v02().items():
        print(f"{key}: {value}")
