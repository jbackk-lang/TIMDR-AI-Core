import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from activation_probe_paderborn import (
    V05Error, _vibration_v07, load_split, power_counts, window_features,
)


class PaderbornActivationProbeTests(unittest.TestCase):
    def test_feature_shape_and_finiteness(self):
        signal = np.zeros(32000)
        features = window_features(signal)
        self.assertEqual(features.shape, (8,))
        self.assertTrue(np.isfinite(features).all())

    def test_v07_pads_two_missing_samples(self):
        with patch("activation_probe_paderborn._raw_vibration", return_value=np.ones(255998)):
            values, adjustment = _vibration_v07(Path("unused"), "K001.rar", "member")
        self.assertEqual(len(values), 256000)
        self.assertEqual(adjustment, -2)
        self.assertTrue(np.all(values[-2:] == 0))

    def test_holdout_refused_before_manifest_or_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("activation_probe_paderborn.committed_plan_matches", return_value=False):
                with self.assertRaisesRegex(V05Error, "Holdout locked"):
                    load_split(Path(directory), "holdout")

    def test_holdout_refused_if_calibration_has_no_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("activation_probe_paderborn.committed_plan_matches", return_value=True):
                with patch("activation_probe_paderborn.calibration_report", return_value={"calibration_power": {"passed": False}}):
                    with self.assertRaisesRegex(V05Error, "power gate"):
                        load_split(Path(directory), "holdout")

    def test_power_gate_requires_errors_and_distinct_measurements(self):
        correct = [{"error": 0, "member": f"c{i % 5}"} for i in range(336)]
        self.assertFalse(power_counts(correct)["passed"])
        errors = [{"error": 1, "member": f"e{i % 5}"} for i in range(30)]
        self.assertTrue(power_counts(correct + errors)["passed"])


if __name__ == "__main__":
    unittest.main()
