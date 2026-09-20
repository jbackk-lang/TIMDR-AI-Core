"""Tests for the generic MLP: gradient correctness, ability to actually
learn, and determinism. Not a validation of usefulness on any real dataset
-- that lives in whatever wires this in (e.g. protect90_waveform_nn_learning.py)."""
from __future__ import annotations

import pytest

from neural_network import MLPClassifier, NeuralNetworkError, _one_hot

numpy = pytest.importorskip("numpy")


def test_rejects_bad_shapes():
    with pytest.raises(NeuralNetworkError):
        MLPClassifier(n_features=0, n_hidden=4, n_classes=3)
    with pytest.raises(NeuralNetworkError):
        MLPClassifier(n_features=4, n_hidden=4, n_classes=1)


def test_backprop_gradients_match_numerical_finite_differences():
    """The core correctness check for hand-derived backprop: for every
    weight, the analytic gradient from _loss_and_grads must match a
    finite-difference numerical gradient of the same loss. This is the
    neural-network equivalent of a positive control -- it does not show the
    network is useful on any dataset, only that its training math is right."""
    rng = numpy.random.default_rng(0)
    n, n_features, n_hidden, n_classes = 6, 4, 5, 3
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, size=n)
    y_onehot = _one_hot(y, n_classes)

    net = MLPClassifier(n_features, n_hidden, n_classes, seed=1)
    _, analytic_grads = net._loss_and_grads(X, y_onehot)

    eps = 1e-5
    for name in ("w1", "b1", "w2", "b2"):
        param = getattr(net, name)
        analytic = analytic_grads[name]
        it = numpy.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            loss_plus, _ = net._loss_and_grads(X, y_onehot)
            param[idx] = original - eps
            loss_minus, _ = net._loss_and_grads(X, y_onehot)
            param[idx] = original
            numerical = (loss_plus - loss_minus) / (2 * eps)
            assert abs(numerical - analytic[idx]) < 1e-4, (
                f"Gradient mismatch in {name}{idx}: numerical={numerical}, analytic={analytic[idx]}"
            )


def test_network_can_actually_learn_a_separable_synthetic_problem():
    """Sanity check that training reduces loss and reaches high training
    accuracy on an easy, linearly-separable-ish 3-blob problem -- proves the
    fit loop does more than run without crashing."""
    rng = numpy.random.default_rng(0)
    centers = numpy.array([[-3.0, -3.0], [3.0, 3.0], [3.0, -3.0]])
    X, y = [], []
    for label, center in enumerate(centers):
        pts = center + rng.normal(scale=0.5, size=(40, 2))
        X.append(pts)
        y.extend([label] * 40)
    X = numpy.vstack(X)
    y = numpy.array(y)

    net = MLPClassifier(n_features=2, n_hidden=8, n_classes=3, seed=0)
    history = net.fit(X, y, epochs=300, lr=0.1)

    assert history.losses[-1] < history.losses[0]
    accuracy = float(numpy.mean(net.predict(X) == y))
    assert accuracy > 0.95


def test_same_seed_gives_identical_results():
    rng = numpy.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 2, size=20)

    net_a = MLPClassifier(n_features=3, n_hidden=4, n_classes=2, seed=7)
    net_a.fit(X, y, epochs=50, lr=0.1)
    net_b = MLPClassifier(n_features=3, n_hidden=4, n_classes=2, seed=7)
    net_b.fit(X, y, epochs=50, lr=0.1)

    assert numpy.array_equal(net_a.predict(X), net_b.predict(X))
    assert numpy.allclose(net_a.w1, net_b.w1)


def test_predict_rejects_wrong_feature_count():
    net = MLPClassifier(n_features=5, n_hidden=3, n_classes=2, seed=0)
    with pytest.raises(NeuralNetworkError):
        net.predict([[1.0, 2.0, 3.0]])
