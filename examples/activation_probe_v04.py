"""One synthetic v0.4 test; read prereg/ACTIVATION_PROBE_v0.4_SYNTHETIC_PLAN.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activation_diagnostics_v03 import (  # noqa: E402
    auc_greater, first_alarm, ks_two_sided, lambda_entropy_series,
    transition_signature,
)
from activation_probe_v03 import CENTERS, _sequence, train_model  # noqa: E402

TARGET_LEVELS = (1.5, 2.0, 2.5, 3.0, 4.0)
LAMBDA_THRESHOLD = 0.1488384562221906
CONFIDENCE_THRESHOLD = 0.10808176126798885


def match_activation_delta(model, previous: np.ndarray, direction: np.ndarray, target: float):
    """Match only activation RMS, never prediction correctness or class."""
    baseline = model.hidden_activations(previous[None, :])[0]

    def delta(amplitude):
        changed = model.hidden_activations((previous + amplitude * direction)[None, :])[0]
        return float(np.sqrt(np.mean((changed - baseline) ** 2)))

    if delta(20.0) < target:
        return None
    low, high = 0.0, 20.0
    for _ in range(32):
        middle = (low + high) / 2
        if delta(middle) < target:
            low = middle
        else:
            high = middle
    candidate = previous + ((low + high) / 2) * direction
    return candidate if abs(delta((low + high) / 2) - target) <= 0.02 else None


def _group_report(rows):
    error = [row for row in rows if row["wrong"]]
    correct = [row for row in rows if not row["wrong"]]
    report = {"errors": len(error), "correct": len(correct), "eligible": len(error) >= 20 and len(correct) >= 20}
    if report["eligible"]:
        report.update({
            "shape_js_auc": auc_greater([r["shape_js"] for r in error], [r["shape_js"] for r in correct]),
            "delta_auc": auc_greater([r["delta"] for r in error], [r["delta"] for r in correct]),
            "confidence_auc": auc_greater([r["uncertainty"] for r in error], [r["uncertainty"] for r in correct]),
            "direction_error_median": float(np.median([r["direction"] for r in error])),
            "direction_correct_median": float(np.median([r["direction"] for r in correct])),
            "direction_ks": ks_two_sided([r["direction"] for r in error], [r["direction"] for r in correct]),
        })
    return report


def _matched_levels(model):
    rng = np.random.default_rng(73)
    levels = []
    type_rows = {str(i): [] for i in range(3)}
    for target in TARGET_LEVELS:
        rows = []
        excluded = 0
        for index in range(300):
            label = index % 3
            noise = 0.35 if index % 2 == 0 else 0.70
            base = CENTERS[label] + rng.normal(0, noise, size=(7, 2))
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            final = match_activation_delta(model, base[-1], direction, target)
            if final is None:
                excluded += 1
                continue
            trajectory = np.vstack((base, final))
            signature = transition_signature(model.hidden_activations(trajectory))
            probs = model.predict_proba(final[None, :])[0]
            row = {
                "wrong": int(np.argmax(probs)) != label,
                "shape_js": signature.shape_js,
                "delta": signature.delta_rms,
                "direction": signature.direction,
                "uncertainty": float(1 - np.max(probs)),
            }
            rows.append(row)
            first_type = str(int(model.predict(base[:1])[0]))
            type_rows[first_type].append(row)
        report = _group_report(rows)
        report.update({"target_delta": target, "attempted": 300, "technical_exclusions": excluded})
        levels.append(report)
    types = {type_name: _group_report(rows) for type_name, rows in type_rows.items()}
    eligible_levels = [row for row in levels if row["eligible"]]
    eligible_types = [row for row in types.values() if row["eligible"]]
    candidate = (
        len(eligible_levels) >= 3 and len(eligible_types) >= 2 and
        all(row["shape_js_auc"] >= 0.65 and row["shape_js_auc"] - row["delta_auc"] >= 0.05 for row in eligible_levels) and
        sum(row["shape_js_auc"] >= 0.60 for row in eligible_types) >= 2
    )
    return {"levels": levels, "by_initial_prediction": types, "replication_candidate": bool(candidate)}


def _ramp_replication(model):
    rng = np.random.default_rng(64)
    eligible_errors = zero_errors = no_error = 0
    early_lambda = early_confidence = false_lambda = false_confidence = 0
    leads = []
    for index in range(600):
        noise = 0.35 if index % 2 == 0 else 0.70
        strength = float(rng.uniform(0.1, 0.45) if index < 300 else rng.uniform(0.7, 1.35))
        x, label = _sequence(rng, 10, noise, strength, ramp=True)
        probs = model.predict_proba(x)
        mistakes = np.flatnonzero(np.argmax(probs, axis=1) != label)
        first_error = int(mistakes[0]) if mistakes.size else None
        lam = lambda_entropy_series(model.hidden_activations(x))
        lambda_alarm = first_alarm(np.abs(np.diff(lam)), LAMBDA_THRESHOLD)
        confidence_alarm = first_alarm(1 - np.max(probs[1:], axis=1), CONFIDENCE_THRESHOLD)
        if first_error is None:
            no_error += 1
            false_lambda += int(lambda_alarm is not None)
            false_confidence += int(confidence_alarm is not None)
        elif first_error == 0:
            zero_errors += 1
        else:
            eligible_errors += 1
            if lambda_alarm is not None and lambda_alarm < first_error:
                early_lambda += 1
                leads.append(first_error - lambda_alarm)
            early_confidence += int(confidence_alarm is not None and confidence_alarm < first_error)
    lambda_recall = early_lambda / eligible_errors if eligible_errors else None
    confidence_recall = early_confidence / eligible_errors if eligible_errors else None
    fpr = false_lambda / no_error if no_error else None
    candidate = (eligible_errors >= 30 and no_error >= 30 and fpr is not None and fpr <= 0.15 and
                 lambda_recall is not None and lambda_recall >= 0.50 and confidence_recall is not None and
                 lambda_recall - confidence_recall >= 0.10)
    return {
        "eligible_errors": eligible_errors, "errors_at_t0": zero_errors, "no_error": no_error,
        "lambda_early_count": early_lambda, "confidence_early_count": early_confidence,
        "lambda_lead_recall": lambda_recall, "confidence_lead_recall": confidence_recall,
        "lambda_false_alarm_count": false_lambda, "confidence_false_alarm_count": false_confidence,
        "lambda_false_alarm_rate": fpr,
        "confidence_false_alarm_rate": false_confidence / no_error if no_error else None,
        "median_lambda_lead_steps": float(np.median(leads)) if leads else None,
        "early_warning_candidate": bool(candidate),
    }


def run_v04():
    model = train_model()
    return {
        "status": "SYNTHETIC_EXPLORATORY_ONLY",
        "A_matched_levels": _matched_levels(model),
        "B_ramp_replication": _ramp_replication(model),
    }


if __name__ == "__main__":
    print(json.dumps(run_v04(), ensure_ascii=False, indent=2))
