"""Exploratory, single-layer activation-trajectory diagnostics.

This is a domain-local TIMDR-inspired adapter, not a TIMDR axiom, a
cross-layer vector field, or a hallucination detector. A trajectory has
shape (successive steps, channels) from the SAME layer of a FROZEN model.
No learning, control gate, or empirical verdict is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class ActivationDiagnosticError(ValueError):
    pass


def _trajectory(values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ActivationDiagnosticError("Expected at least 2 steps and 2 channels from one layer.")
    if not np.isfinite(array).all():
        raise ActivationDiagnosticError("Activations must be finite.")
    return array


@dataclass(frozen=True)
class ActivationMetrics:
    delta_time: np.ndarray
    lambda_channel: np.ndarray
    tau_lambda_time: np.ndarray


def measure_trajectory(values: Sequence[Sequence[float]], *, step_interval: float = 1.0) -> ActivationMetrics:
    """Compute three explicitly local observables, never cross-layer diffs.

    delta_time[t] = RMS(a[t+1]-a[t]) across channels.
    lambda_channel[t] = std(a[t]) / RMS(a[t]) across channels (0 for zero vector).
    tau_lambda_time[t] = abs(lambda[t+1]-lambda[t]) / step_interval.
    The last is a rate of *dispersion* change, not a rate of cognition.
    """
    a = _trajectory(values)
    if not np.isfinite(step_interval) or step_interval <= 0:
        raise ActivationDiagnosticError("step_interval must be finite and positive.")
    delta = np.sqrt(np.mean(np.diff(a, axis=0) ** 2, axis=1))
    rms = np.sqrt(np.mean(a * a, axis=1))
    dispersion = np.divide(np.std(a, axis=1), rms, out=np.zeros_like(rms), where=rms > 0)
    tau = np.abs(np.diff(dispersion)) / step_interval
    return ActivationMetrics(delta, dispersion, tau)


def _features(values: Sequence[Sequence[float]], *, step_interval: float) -> np.ndarray:
    metrics = measure_trajectory(values, step_interval=step_interval)
    return np.column_stack((metrics.delta_time, metrics.lambda_channel[1:], metrics.tau_lambda_time))


@dataclass(frozen=True)
class HealthyReference:
    """Frozen reference estimated only from known-correct calibration trajectories."""

    center: np.ndarray
    scale: np.ndarray
    threshold: float
    channels: int
    steps: int
    step_interval: float

    def score(self, values: Sequence[Sequence[float]]) -> float:
        a = _trajectory(values)
        if a.shape != (self.steps, self.channels):
            raise ActivationDiagnosticError("Trajectory shape differs from frozen reference.")
        z = np.abs((_features(a, step_interval=self.step_interval) - self.center) / self.scale)
        return float(np.max(z))

    def alarm(self, values: Sequence[Sequence[float]]) -> bool:
        return self.score(values) > self.threshold


def fit_healthy_reference(
    trajectories: Sequence[Sequence[Sequence[float]]],
    *,
    step_interval: float = 1.0,
    false_alarm_quantile: float = 0.95,
) -> HealthyReference:
    """Fit on known-correct calibration trajectories; never inspect test labels.

    Threshold uses the empirical nearest-rank quantile of per-trajectory
    maximum scores. A separate holdout is needed to measure actual FPR.
    """
    if not 0 < false_alarm_quantile < 1:
        raise ActivationDiagnosticError("false_alarm_quantile must lie strictly between 0 and 1.")
    arrays = [_trajectory(row) for row in trajectories]
    if len(arrays) < 2:
        raise ActivationDiagnosticError("At least two healthy trajectories are required.")
    shape = arrays[0].shape
    if any(row.shape != shape for row in arrays):
        raise ActivationDiagnosticError("Calibration trajectories must share shape and layer.")
    features = np.concatenate([_features(row, step_interval=step_interval) for row in arrays])
    center = np.median(features, axis=0)
    mad = np.median(np.abs(features - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-8)
    scores = [float(np.max(np.abs((_features(row, step_interval=step_interval) - center) / scale))) for row in arrays]
    rank = int(np.ceil(false_alarm_quantile * len(scores))) - 1
    return HealthyReference(center, scale, float(np.sort(scores)[rank]), shape[1], shape[0], step_interval)
