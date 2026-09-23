"""Small SYNTHETIC illustration of a single-layer activation diagnostic.

No real-world claim, hallucination claim, TIMDRProtocol verdict, or holdout
from the existing PROTECT-90/B4/Paderborn experiments is produced here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activation_diagnostics import fit_healthy_reference  # noqa: E402
from neural_network import MLPClassifier  # noqa: E402


CENTERS = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5]])
STEPS = 8


def _sequence(rng: np.random.Generator, *, disturbed: bool) -> tuple[np.ndarray, int]:
    label = int(rng.integers(0, len(CENTERS)))
    samples = CENTERS[label] + rng.normal(0.0, 0.35, size=(STEPS, 2))
    if disturbed:
        other = (label + 1) % len(CENTERS)
        strength = float(rng.uniform(0.3, 1.25))
        samples[-1] += strength * (CENTERS[other] - CENTERS[label])
    return samples, label


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def run_demo() -> dict[str, float | int | None]:
    train_rng = np.random.default_rng(10)
    x_train = np.vstack([CENTER + train_rng.normal(0.0, 0.45, size=(120, 2)) for CENTER in CENTERS])
    y_train = np.repeat(np.arange(3), 120)
    model = MLPClassifier(2, 12, 3, seed=7)
    model.fit(x_train, y_train, epochs=250, lr=0.1)

    cal_rng = np.random.default_rng(20)
    calibration = [_sequence(cal_rng, disturbed=False) for _ in range(100)]
    healthy = [model.hidden_activations(x) for x, y in calibration if int(model.predict(x[-1:])[0]) == y]
    reference = fit_healthy_reference(healthy, false_alarm_quantile=0.95)

    test_rng = np.random.default_rng(30)
    cases = [_sequence(test_rng, disturbed=i >= 100) for i in range(200)]
    errors: list[bool] = []
    diagnostic: list[float] = []
    uncertainty: list[float] = []
    hard_correct_alarms = 0
    hard_correct_count = 0
    clean_correct_alarms = 0
    clean_correct_count = 0
    disturbed_error_alarms = 0
    disturbed_error_count = 0
    for index, (x, y) in enumerate(cases):
        probs = model.predict_proba(x[-1:])[0]
        wrong = int(np.argmax(probs)) != y
        score = reference.score(model.hidden_activations(x))
        alarm = score > reference.threshold
        errors.append(wrong)
        diagnostic.append(score)
        uncertainty.append(float(1.0 - np.max(probs)))
        if index < 100 and not wrong:
            clean_correct_count += 1
            clean_correct_alarms += int(alarm)
        if index >= 100 and not wrong:
            hard_correct_count += 1
            hard_correct_alarms += int(alarm)
        if index >= 100 and wrong:
            disturbed_error_count += 1
            disturbed_error_alarms += int(alarm)
    return {
        "calibration_correct": len(healthy),
        "test_count": len(cases),
        "test_errors": sum(errors),
        "clean_correct_count": clean_correct_count,
        "clean_correct_alarm_rate": clean_correct_alarms / clean_correct_count if clean_correct_count else None,
        "hard_correct_count": hard_correct_count,
        "hard_correct_alarm_rate": hard_correct_alarms / hard_correct_count if hard_correct_count else None,
        "disturbed_error_count": disturbed_error_count,
        "disturbed_error_alarm_rate": disturbed_error_alarms / disturbed_error_count if disturbed_error_count else None,
        "diagnostic_auc": _auc(errors, diagnostic),
        "confidence_baseline_auc": _auc(errors, uncertainty),
    }


if __name__ == "__main__":
    print("SYNTHETIC DEMO ONLY - no empirical TIMDR or hallucination verdict")
    for key, value in run_demo().items():
        print(f"{key}: {value}")
