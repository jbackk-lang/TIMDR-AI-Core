import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from activation_diagnostics_v03 import transition_signature  # noqa: E402
from activation_probe_v04 import match_activation_delta, _group_report  # noqa: E402
from neural_network import MLPClassifier  # noqa: E402


class V04MechanicsTests(unittest.TestCase):
    def test_matching_uses_activation_magnitude(self):
        model = MLPClassifier(2, 12, 3, seed=51)
        previous = np.array([1.0, -1.0])
        direction = np.array([1.0, 0.0])
        target = 0.75
        final = match_activation_delta(model, previous, direction, target)
        self.assertIsNotNone(final)
        actual = transition_signature(model.hidden_activations(np.vstack((previous, final)))).delta_rms
        self.assertLess(abs(actual - target), 0.02)

    def test_ineligible_group_does_not_report_auc(self):
        report = _group_report([{"wrong": True, "shape_js": 0.5}])
        self.assertFalse(report["eligible"])
        self.assertNotIn("shape_js_auc", report)


if __name__ == "__main__":
    unittest.main()
