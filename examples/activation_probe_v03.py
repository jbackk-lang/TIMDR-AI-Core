"""One predefined synthetic v0.3 evaluation; run only after reading prereg plan."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activation_diagnostics_v03 import (  # noqa: E402
    auc_greater, first_alarm, ks_two_sided, lambda_entropy_series,
    nearest_rank, transition_signature,
)
from neural_network import MLPClassifier  # noqa: E402

CENTERS = np.array([[-2.5, -2.5], [2.5, 2.5], [2.5, -2.5]])


def train_model() -> MLPClassifier:
    rng = np.random.default_rng(51)
    x = np.vstack([center + rng.normal(0, 0.45, size=(120, 2)) for center in CENTERS])
    y = np.repeat(np.arange(3), 120)
    model = MLPClassifier(2, 12, 3, seed=51)
    model.fit(x, y, epochs=250, lr=0.1)
    return model


def _sequence(rng, steps: int, noise: float, strength: float = 0.0, *, ramp: bool = False):
    label = int(rng.integers(0, 3))
    x = CENTERS[label] + rng.normal(0, noise, size=(steps, 2))
    if strength:
        displacement = CENTERS[(label + 1) % 3] - CENTERS[label]
        if ramp:
            fractions = np.zeros(steps)
            fractions[2:] = np.arange(1, steps - 1) / (steps - 2)
            x += strength * fractions[:, None] * displacement
        else:
            x[-1] += strength * displacement
    return x, label


def _correct_all(model, x, label):
    return bool(np.all(model.predict(x) == label))


def _healthy_sequences(model, seed: int, count: int, steps: int):
    rng = np.random.default_rng(seed)
    sequences = []
    attempts = 0
    while len(sequences) < count and attempts < count * 100:
        noise = 0.35 if len(sequences) % 2 == 0 else 0.70
        x, label = _sequence(rng, steps, noise)
        attempts += 1
        if _correct_all(model, x, label):
            sequences.append((x, label))
    if len(sequences) != count:
        raise RuntimeError("Insufficient all-correct calibration sequences; stop without changing plan.")
    return sequences


def _matched_analysis(model):
    calibration = _healthy_sequences(model, 52, 240, 8)
    deltas = [transition_signature(model.hidden_activations(x)).delta_rms for x, _ in calibration]
    cuts = [nearest_rank(deltas, q) for q in (0.25, 0.50, 0.75, 0.90)]
    bins = [[] for _ in range(5)]
    rng = np.random.default_rng(53)
    clean_correct = clean_error = disturbed_correct = disturbed_error = 0
    for index in range(1000):
        noise = 0.35 if index % 2 == 0 else 0.70
        strength = 0.0 if index < 200 else float(rng.uniform(0.3, 1.25))
        x, label = _sequence(rng, 8, noise, strength)
        error = int(model.predict(x[-1:])[0]) != label
        if index < 200:
            clean_error += int(error)
            clean_correct += int(not error)
        else:
            disturbed_error += int(error)
            disturbed_correct += int(not error)
            signature = transition_signature(model.hidden_activations(x))
            bin_index = int(np.searchsorted(cuts, signature.delta_rms, side="right"))
            bins[bin_index].append((error, signature))

    reports = []
    matched_pairs = 0
    weighted_locality = weighted_shape = weighted_ks = 0.0
    for index, rows in enumerate(bins):
        errors = [s for error, s in rows if error]
        correct = [s for error, s in rows if not error]
        eligible = len(errors) >= 10 and len(correct) >= 10
        report = {"bin": index, "errors": len(errors), "correct": len(correct), "eligible": eligible}
        if eligible:
            matched_pairs += min(len(errors), len(correct))
            direction_error = [s.direction for s in errors]
            direction_correct = [s.direction for s in correct]
            report.update({
                "median_direction_error": float(np.median(direction_error)),
                "median_direction_correct": float(np.median(direction_correct)),
                "direction_ks": ks_two_sided(direction_error, direction_correct),
                "locality_auc": auc_greater([s.locality for s in errors], [s.locality for s in correct]),
                "shape_js_auc": auc_greater([s.shape_js for s in errors], [s.shape_js for s in correct]),
            })
            # Weight only comparisons from the same delta bin.
            n = min(len(errors), len(correct))
            weighted_locality += n * report["locality_auc"]
            weighted_shape += n * report["shape_js_auc"]
            weighted_ks += n * report["direction_ks"]
        reports.append(report)
    locality_auc = weighted_locality / matched_pairs if matched_pairs else None
    shape_auc = weighted_shape / matched_pairs if matched_pairs else None
    direction_ks = weighted_ks / matched_pairs if matched_pairs else None
    return {
        "calibration_count": len(calibration), "delta_bin_boundaries": cuts,
        "clean_correct": clean_correct, "clean_error": clean_error,
        "disturbed_correct": disturbed_correct, "disturbed_error": disturbed_error,
        "matched_pairs": matched_pairs, "bins": reports,
        "weighted_within_bin_locality_auc": locality_auc,
        "weighted_within_bin_shape_js_auc": shape_auc,
        "weighted_within_bin_direction_ks": direction_ks,
        "replication_candidate": matched_pairs >= 30 and (
            (locality_auc is not None and locality_auc >= 0.60) or
            (shape_auc is not None and shape_auc >= 0.60) or
            (direction_ks is not None and direction_ks >= 0.20)
        ),
    }


def _early_warning(model):
    calibration = _healthy_sequences(model, 62, 240, 10)
    lambda_maxima = []
    confidence_maxima = []
    for x, _ in calibration:
        lam = lambda_entropy_series(model.hidden_activations(x))
        lambda_maxima.append(float(np.max(np.abs(np.diff(lam)))))
        confidence_maxima.append(float(np.max(1 - np.max(model.predict_proba(x)[1:], axis=1))))
    lambda_threshold = nearest_rank(lambda_maxima, 0.95)
    confidence_threshold = nearest_rank(confidence_maxima, 0.95)

    rng = np.random.default_rng(63)
    eligible_errors = early_lambda = early_confidence = 0
    no_error = false_lambda = false_confidence = zero_error = 0
    leads = []
    by_initial_prediction = {str(i): {"no_error": 0, "false_lambda": 0} for i in range(3)}
    for index in range(600):
        noise = 0.35 if index % 2 == 0 else 0.70
        strength = float(rng.uniform(0.1, 0.45) if index < 300 else rng.uniform(0.7, 1.35))
        x, label = _sequence(rng, 10, noise, strength, ramp=True)
        probs = model.predict_proba(x)
        wrong = np.flatnonzero(np.argmax(probs, axis=1) != label)
        t_error = int(wrong[0]) if wrong.size else None
        lam = lambda_entropy_series(model.hidden_activations(x))
        t_lambda = first_alarm(np.abs(np.diff(lam)), lambda_threshold)
        t_confidence = first_alarm(1 - np.max(probs[1:], axis=1), confidence_threshold)
        initial_type = str(int(np.argmax(probs[0])))
        if t_error is None:
            no_error += 1
            false_lambda += int(t_lambda is not None)
            false_confidence += int(t_confidence is not None)
            by_initial_prediction[initial_type]["no_error"] += 1
            by_initial_prediction[initial_type]["false_lambda"] += int(t_lambda is not None)
        elif t_error == 0:
            zero_error += 1
        else:
            eligible_errors += 1
            if t_lambda is not None and t_lambda < t_error:
                early_lambda += 1
                leads.append(t_error - t_lambda)
            early_confidence += int(t_confidence is not None and t_confidence < t_error)

    lambda_recall = early_lambda / eligible_errors if eligible_errors else None
    confidence_recall = early_confidence / eligible_errors if eligible_errors else None
    fpr = false_lambda / no_error if no_error else None
    return {
        "calibration_count": len(calibration),
        "lambda_threshold": lambda_threshold, "confidence_threshold": confidence_threshold,
        "eligible_errors": eligible_errors, "errors_at_t0": zero_error,
        "no_error": no_error,
        "lambda_early_count": early_lambda, "confidence_early_count": early_confidence,
        "lambda_lead_recall": lambda_recall, "confidence_lead_recall": confidence_recall,
        "lambda_false_alarm_rate": fpr,
        "confidence_false_alarm_rate": false_confidence / no_error if no_error else None,
        "median_lead_steps": float(np.median(leads)) if leads else None,
        "control_by_initial_prediction": by_initial_prediction,
        "early_warning_candidate": eligible_errors >= 30 and no_error >= 30 and
            fpr is not None and fpr <= 0.15 and lambda_recall is not None and
            lambda_recall >= 0.50 and confidence_recall is not None and
            lambda_recall - confidence_recall >= 0.10,
    }


def run_v03():
    model = train_model()
    return {"status": "SYNTHETIC_EXPLORATORY_ONLY", "A_matched_delta": _matched_analysis(model), "B_early_warning": _early_warning(model)}


if __name__ == "__main__":
    print(json.dumps(run_v03(), ensure_ascii=False, indent=2))
