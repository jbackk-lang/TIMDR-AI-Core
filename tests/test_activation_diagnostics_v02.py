import unittest

import numpy as np

from activation_diagnostics import ActivationDiagnosticError
from activation_diagnostics_v02 import final_transition_features, fit_context_reference, prefinal_difficulty
from examples.activation_probe_v02 import run_v02


class ActivationDiagnosticV02Tests(unittest.TestCase):
    def test_final_perturbation_cannot_change_prefinal_difficulty(self):
        x = np.arange(16.0).reshape(8, 2)
        changed = x.copy()
        changed[-1] += 1000
        self.assertEqual(prefinal_difficulty(x), prefinal_difficulty(changed))

    def test_features_separate_mean_and_spread(self):
        base = np.ones((6, 4))
        shifted = base.copy()
        shifted[-1] += 2
        varied = base.copy()
        varied[-1] += [-2, 2, -2, 2]
        shift_features = final_transition_features(shifted)
        varied_features = final_transition_features(varied)
        self.assertAlmostEqual(shift_features[0], 2)
        self.assertAlmostEqual(shift_features[1], 0)
        self.assertAlmostEqual(varied_features[0], 0)
        self.assertGreater(varied_features[1], 0)

    def test_reference_requires_observable_context_and_shape(self):
        rng = np.random.default_rng(3)
        x = [rng.normal(0, 0.2 if i < 10 else 1, size=(8, 2)) for i in range(20)]
        a = [np.abs(rng.normal(size=(8, 4))) for _ in x]
        reference = fit_context_reference(a, x)
        with self.assertRaises(ActivationDiagnosticError):
            reference.score(a[0], context=None)
        with self.assertRaises(ActivationDiagnosticError):
            reference.score(np.ones((8, 3)), context="easy")
        self.assertIn(reference.context(x[0]), {"easy", "hard"})

    def test_fresh_synthetic_comparison_is_deterministic(self):
        first = run_v02()
        self.assertEqual(first, run_v02())
        self.assertGreater(first["hard_correct_count"], 0)
        self.assertGreater(first["disturbed_error_count"], 0)
        for name in ("v01", "v02_global", "v02_context"):
            self.assertLessEqual(0, first[f"{name}_hard_correct_fpr"])
            self.assertLessEqual(first[f"{name}_hard_correct_fpr"], 1)


if __name__ == "__main__":
    unittest.main()
