import unittest

import numpy as np

from activation_diagnostics_v03 import (
    auc_greater, first_alarm, ks_two_sided, lambda_entropy_series,
    nearest_rank, transition_signature,
)


class ActivationV03Tests(unittest.TestCase):
    def test_direction_reverses_and_locality(self):
        forward = transition_signature([[1, 2, 3], [2, 3, 4]])
        reverse = transition_signature([[2, 3, 4], [1, 2, 3]])
        self.assertAlmostEqual(forward.direction, 1)
        self.assertAlmostEqual(reverse.direction, -1)
        self.assertAlmostEqual(forward.locality, 1 / 3)

    def test_shape_identical_and_distinct(self):
        same = transition_signature([[1, 2, 3], [2, 4, 6]])
        different = transition_signature([[1, 0, 0], [0, 1, 0]])
        self.assertAlmostEqual(same.shape_js, 0)
        self.assertGreater(different.shape_js, 0)

    def test_lambda_entropy_bounds(self):
        series = lambda_entropy_series([[1, 0, 0], [1, 1, 1], [0, 0, 0]])
        self.assertAlmostEqual(series[0], 0)
        self.assertAlmostEqual(series[1], 1)
        self.assertTrue(np.all((series >= 0) & (series <= 1)))

    def test_threshold_and_first_alarm(self):
        self.assertEqual(nearest_rank([4, 1, 3, 2], 0.75), 3)
        self.assertEqual(first_alarm([0.1, 0.3, 0.5], 0.2), 2)
        self.assertIsNone(first_alarm([0.1, 0.2], 0.2))

    def test_auc_and_ks(self):
        self.assertEqual(auc_greater([2, 3], [0, 1]), 1)
        self.assertEqual(auc_greater([0, 1], [2, 3]), 0)
        self.assertEqual(ks_two_sided([0, 1], [2, 3]), 1)


if __name__ == "__main__":
    unittest.main()
