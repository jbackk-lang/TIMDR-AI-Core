"""Dependency-light tests: python -m unittest discover -s tests -p test_activation_diagnostics.py"""
import unittest

import numpy as np

from activation_diagnostics import ActivationDiagnosticError, fit_healthy_reference, measure_trajectory
from examples.activation_probe_demo import run_demo
from neural_network import MLPClassifier


class ActivationDiagnosticTests(unittest.TestCase):
    def test_local_metrics_have_explicit_shapes_and_time_scale(self):
        a = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        metrics = measure_trajectory(a, step_interval=2.0)
        self.assertEqual(metrics.delta_time.shape, (2,))
        self.assertEqual(metrics.lambda_channel.shape, (3,))
        self.assertEqual(metrics.tau_lambda_time.shape, (2,))
        np.testing.assert_allclose(metrics.delta_time, [1.0, 1.0])
        np.testing.assert_allclose(metrics.tau_lambda_time, [0.0, 0.0])

    def test_positive_rescale_preserves_dispersion_not_delta(self):
        a = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        original = measure_trajectory(a)
        scaled = measure_trajectory(3 * a)
        np.testing.assert_allclose(original.lambda_channel, scaled.lambda_channel)
        np.testing.assert_allclose(3 * original.delta_time, scaled.delta_time)

    def test_invalid_shapes_and_nonfinite_values_rejected(self):
        with self.assertRaises(ActivationDiagnosticError):
            measure_trajectory([[1.0, 2.0]])
        with self.assertRaises(ActivationDiagnosticError):
            measure_trajectory([[1.0, 2.0], [float("nan"), 2.0]])
        with self.assertRaises(ActivationDiagnosticError):
            measure_trajectory([[1.0, 2.0], [2.0, 3.0]], step_interval=0)

    def test_reference_is_shape_frozen_and_detects_late_jump(self):
        rng = np.random.default_rng(4)
        healthy = [np.ones((5, 3)) + rng.normal(0, 0.01, (5, 3)) for _ in range(30)]
        reference = fit_healthy_reference(healthy)
        changed = healthy[0].copy()
        changed[-1] += 10
        self.assertTrue(reference.alarm(changed))
        with self.assertRaises(ActivationDiagnosticError):
            reference.score(np.ones((5, 4)))

    def test_model_exposes_only_its_own_hidden_layer_in_order(self):
        model = MLPClassifier(2, 4, 3, seed=3)
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        hidden = model.hidden_activations(x)
        self.assertEqual(hidden.shape, (2, 4))
        np.testing.assert_allclose(hidden, model._forward(x)[1][2])
        hidden[0, :] = 100
        self.assertFalse(np.allclose(hidden, model.hidden_activations(x)))

    def test_synthetic_demo_is_deterministic_and_has_both_outcome_classes(self):
        first = run_demo()
        self.assertEqual(first, run_demo())
        self.assertLess(0, first["test_errors"])
        self.assertLess(first["test_errors"], first["test_count"])
        self.assertGreater(first["disturbed_error_count"], 0)
        self.assertGreater(first["hard_correct_count"], 0)
        self.assertLessEqual(0, first["diagnostic_auc"])
        self.assertLessEqual(first["diagnostic_auc"], 1)
        self.assertLessEqual(0, first["confidence_baseline_auc"])
        self.assertLessEqual(first["confidence_baseline_auc"], 1)


if __name__ == "__main__":
    unittest.main()
