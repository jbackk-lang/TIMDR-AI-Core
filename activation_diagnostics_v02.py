"""Exploratory v0.2 of the single-layer activation probe.

Separate from v0.1: no threshold or definition is changed after inspecting
its result. Context is determined from input movement BEFORE the final step,
never from the true class, final prediction, or final perturbation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from activation_diagnostics import ActivationDiagnosticError, _trajectory


def prefinal_difficulty(inputs: Sequence[Sequence[float]]) -> float:
    """RMS input movement on steps 0..T-2; the last step is excluded."""
    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or x.shape[0] < 4 or not np.isfinite(x).all():
        raise ActivationDiagnosticError("Expected at least four finite input steps.")
    return float(np.sqrt(np.mean(np.diff(x[:-1], axis=0) ** 2)))


def final_transition_features(activations: Sequence[Sequence[float]], *, step_interval: float = 1.0) -> np.ndarray:
    """Four fixed, same-layer features of the final transition.

    signed mean change; RMS of mean-centered channel changes; Shannon
    entropy of absolute channel activity at the final step; absolute entropy
    rate divided by the median pre-final entropy rate (+0.05 floor).
    Entropy is a *local activation-shape statistic*, not physical entropy.
    """
    a = _trajectory(activations)
    if a.shape[0] < 4:
        raise ActivationDiagnosticError("At least four steps are needed for pre-final normalization.")
    if not np.isfinite(step_interval) or step_interval <= 0:
        raise ActivationDiagnosticError("step_interval must be finite and positive.")
    difference = a[-1] - a[-2]
    mean_change = float(np.mean(difference))
    spread_change = float(np.sqrt(np.mean((difference - mean_change) ** 2)))
    weights = np.abs(a) + 1e-12
    weights /= np.sum(weights, axis=1, keepdims=True)
    entropy = -np.sum(weights * np.log(weights), axis=1) / np.log(a.shape[1])
    prefinal_rate = np.abs(np.diff(entropy[:-1])) / step_interval
    tau = float(np.abs(entropy[-1] - entropy[-2]) / step_interval / (0.05 + np.median(prefinal_rate)))
    return np.array([mean_change, spread_change, float(entropy[-1]), tau])


def _fit_one(rows: Sequence[np.ndarray], *, quantile: float) -> tuple[np.ndarray, np.ndarray, float]:
    features = np.stack([final_transition_features(row) for row in rows])
    center = np.median(features, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(features - center), axis=0), 1e-8)
    scores = np.max(np.abs((features - center) / scale), axis=1)
    rank = int(np.ceil(quantile * len(scores))) - 1
    return center, scale, float(np.sort(scores)[rank])


@dataclass(frozen=True)
class ContextReference:
    difficulty_cutoff: float
    global_stats: tuple[np.ndarray, np.ndarray, float]
    easy_stats: tuple[np.ndarray, np.ndarray, float]
    hard_stats: tuple[np.ndarray, np.ndarray, float]
    steps: int
    channels: int

    def context(self, inputs: Sequence[Sequence[float]]) -> str:
        return "hard" if prefinal_difficulty(inputs) > self.difficulty_cutoff else "easy"

    def score(self, activations: Sequence[Sequence[float]], *, context: str | None = None) -> tuple[float, float]:
        a = _trajectory(activations)
        if a.shape != (self.steps, self.channels):
            raise ActivationDiagnosticError("Trajectory shape differs from frozen reference.")
        if context not in {"easy", "hard"}:
            raise ActivationDiagnosticError("Context must be computed from pre-final inputs.")
        features = final_transition_features(a)
        def score_for(stats: tuple[np.ndarray, np.ndarray, float]) -> float:
            center, scale, _ = stats
            return float(np.max(np.abs((features - center) / scale)))
        return score_for(self.global_stats), score_for(self.hard_stats if context == "hard" else self.easy_stats)

    def alarms(self, activations: Sequence[Sequence[float]], *, context: str | None = None) -> tuple[bool, bool]:
        global_score, conditioned_score = self.score(activations, context=context)
        stats = self.hard_stats if context == "hard" else self.easy_stats
        return global_score > self.global_stats[2], conditioned_score > stats[2]


def fit_context_reference(
    activations: Sequence[Sequence[Sequence[float]]],
    inputs: Sequence[Sequence[Sequence[float]]],
    *,
    false_alarm_quantile: float = 0.95,
) -> ContextReference:
    """Use known-correct calibration trajectories only; split by pre-final difficulty."""
    if not 0 < false_alarm_quantile < 1:
        raise ActivationDiagnosticError("false_alarm_quantile must lie strictly between 0 and 1.")
    if len(activations) != len(inputs) or len(activations) < 4:
        raise ActivationDiagnosticError("Need matched calibration trajectories and inputs.")
    arrays = [_trajectory(row) for row in activations]
    shape = arrays[0].shape
    if any(row.shape != shape for row in arrays):
        raise ActivationDiagnosticError("All trajectories must come from the same layer and length.")
    difficulties = np.array([prefinal_difficulty(row) for row in inputs])
    cutoff = float(np.median(difficulties))
    easy = [row for row, value in zip(arrays, difficulties) if value <= cutoff]
    hard = [row for row, value in zip(arrays, difficulties) if value > cutoff]
    if min(len(easy), len(hard)) < 2:
        raise ActivationDiagnosticError("Both difficulty strata need at least two healthy trajectories.")
    return ContextReference(
        cutoff,
        _fit_one(arrays, quantile=false_alarm_quantile),
        _fit_one(easy, quantile=false_alarm_quantile),
        _fit_one(hard, quantile=false_alarm_quantile),
        shape[0],
        shape[1],
    )
