"""A small, dependency-free (numpy-only) multilayer perceptron.

This module is generic -- it knows nothing about TIMDR, preregistration,
datasets, or verdicts. It is a plain, from-scratch classifier: one hidden
ReLU layer, softmax output, cross-entropy loss, full-batch gradient descent,
manual (hand-derived) backpropagation, deterministic given a seed.

Anything that wires this into a TIMDR-adjacent dataset (e.g.
``protect90_waveform_nn_learning.py``) is responsible for keeping its output
firewalled from ``TIMDRProtocol`` exactly the way ``learning_sandbox.py``
does for the nearest-centroid baseline: a fitted model or its accuracy is a
research candidate, never a ``SUPPORTED``/``NOT_SUPPORTED``/``INCONCLUSIVE``
verdict, and never something that bypasses preregistered controls.

The manual backprop derivatives here are verified against numerical
(finite-difference) gradients in ``tests/test_neural_network.py`` -- the
same discipline as a positive control: it does not prove the model is
*useful* on any given dataset, only that the gradient math this fitting
procedure relies on is correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised via NeuralNetworkError path
    _np = None


class NeuralNetworkError(RuntimeError):
    pass


def _require_numpy():
    if _np is None:
        raise NeuralNetworkError(
            "neural_network needs numpy. Install it with: "
            'pip install -e ".[waveform]" (or add numpy>=1.24 yourself).'
        )
    return _np


def _one_hot(y, n_classes: int):
    np = _require_numpy()
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def _relu(x):
    np = _require_numpy()
    return np.maximum(0.0, x)


def _relu_grad(x):
    np = _require_numpy()
    return (x > 0.0).astype(float)


def _softmax(x):
    np = _require_numpy()
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class TrainingHistory:
    losses: list[float] = field(default_factory=list)
    validation_accuracies: list[float] = field(default_factory=list)


class MLPClassifier:
    """One-hidden-layer MLP: input -> Dense(hidden, ReLU) -> Dense(classes, softmax).

    Fully deterministic given ``seed`` (weight init only source of
    randomness; training itself is full-batch gradient descent, no
    mini-batch shuffling).
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int, seed: int = 0) -> None:
        np = _require_numpy()
        if n_features < 1:
            raise NeuralNetworkError("n_features must be >= 1.")
        if n_hidden < 1:
            raise NeuralNetworkError("n_hidden must be >= 1.")
        if n_classes < 2:
            raise NeuralNetworkError("n_classes must be >= 2.")
        rng = np.random.default_rng(seed)
        # He init (ReLU hidden layer); smaller-scale init for the linear output layer.
        self.w1 = rng.normal(0.0, (2.0 / n_features) ** 0.5, size=(n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0.0, (1.0 / n_hidden) ** 0.5, size=(n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.seed = seed

    def _forward(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.w2 + self.b2
        probs = _softmax(z2)
        return probs, (X, z1, a1, z2)

    def predict_proba(self, X: Sequence[Sequence[float]]):
        np = _require_numpy()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise NeuralNetworkError(f"Expected input shape (n, {self.n_features}); got {X.shape}.")
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: Sequence[Sequence[float]]):
        np = _require_numpy()
        return np.argmax(self.predict_proba(X), axis=1)

    def _loss_and_grads(self, X, y_onehot) -> tuple[float, dict[str, Any]]:
        np = _require_numpy()
        n = X.shape[0]
        probs, (X_in, z1, a1, z2) = self._forward(X)
        eps = 1e-12
        loss = float(-np.sum(y_onehot * np.log(np.clip(probs, eps, 1.0))) / n)

        d_z2 = (probs - y_onehot) / n
        d_w2 = a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)
        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * _relu_grad(z1)
        d_w1 = X_in.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)
        return loss, {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        epochs: int = 500,
        lr: float = 0.05,
        X_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[int] | None = None,
    ) -> TrainingHistory:
        np = _require_numpy()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise NeuralNetworkError(f"Expected X shape (n, {self.n_features}); got {X.shape}.")
        if len(y) != len(X):
            raise NeuralNetworkError("X and y must have the same number of rows.")
        if epochs < 1:
            raise NeuralNetworkError("epochs must be >= 1.")
        y_onehot = _one_hot(y, self.n_classes)
        history = TrainingHistory()
        for _ in range(epochs):
            loss, grads = self._loss_and_grads(X, y_onehot)
            self.w1 -= lr * grads["w1"]
            self.b1 -= lr * grads["b1"]
            self.w2 -= lr * grads["w2"]
            self.b2 -= lr * grads["b2"]
            history.losses.append(loss)
            if X_val is not None and y_val is not None:
                acc = float(np.mean(self.predict(X_val) == np.asarray(y_val, dtype=int)))
                history.validation_accuracies.append(acc)
        return history
