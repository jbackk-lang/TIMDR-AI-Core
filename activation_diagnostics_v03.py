"""Predefined, single-layer v0.3 activation observables.

These are local measurements of a frozen model, not TIMDR axioms or a
validated error detector. A row is a later input to the SAME hidden layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activation_diagnostics import ActivationDiagnosticError, _trajectory


@dataclass(frozen=True)
class TransitionSignature:
    delta_rms: float
    direction: float
    locality: float
    shape_js: float


def _channel_distribution(row: np.ndarray) -> np.ndarray:
    weights = np.abs(row)
    total = float(np.sum(weights))
    if total == 0:
        return np.full(row.size, 1.0 / row.size)
    return weights / total


def _kl_to_middle(probabilities: np.ndarray, middle: np.ndarray) -> float:
    present = probabilities > 0
    return float(np.sum(probabilities[present] * np.log(probabilities[present] / middle[present])))


def lambda_entropy_series(values) -> np.ndarray:
    """Normalized entropy of absolute activations, one value per step."""
    a = _trajectory(values)
    p = np.stack([_channel_distribution(row) for row in a])
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=1)
    return entropy / np.log(a.shape[1])


def transition_signature(values) -> TransitionSignature:
    """Direction, spatial concentration and shape change of the final step."""
    a = _trajectory(values)
    difference = a[-1] - a[-2]
    magnitude = float(np.sqrt(np.mean(difference * difference)))
    direction = float(np.mean(difference) / magnitude) if magnitude else 0.0
    absolute = np.abs(difference)
    total = float(np.sum(absolute))
    locality = float(np.max(absolute) / total) if total else 0.0
    before = _channel_distribution(a[-2])
    after = _channel_distribution(a[-1])
    middle = (before + after) / 2
    js = 0.5 * _kl_to_middle(before, middle) + 0.5 * _kl_to_middle(after, middle)
    return TransitionSignature(magnitude, direction, locality, js)


def nearest_rank(values, quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ActivationDiagnosticError("Expected a nonempty finite vector.")
    if not 0 < quantile < 1:
        raise ActivationDiagnosticError("Quantile must be between zero and one.")
    return float(np.sort(array)[int(np.ceil(quantile * array.size)) - 1])


def first_alarm(scores, threshold: float, *, first_step: int = 1) -> int | None:
    """Online alarm: return earliest t, without looking at later scores."""
    array = np.asarray(scores, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all() or not np.isfinite(threshold):
        raise ActivationDiagnosticError("Alarm scores and threshold must be finite.")
    for index, score in enumerate(array):
        if score > threshold:
            return first_step + index
    return None


def auc_greater(positive, negative) -> float | None:
    """Probability that a predefined higher-is-error feature ranks an error above a correct case."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return None
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def ks_two_sided(first, second) -> float | None:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    grid = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / a.size - np.searchsorted(np.sort(b), grid, side="right") / b.size)))
