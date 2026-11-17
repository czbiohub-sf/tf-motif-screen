import unittest

import numpy as np
import pandas as pd

from export_tools.metric_bands import add_metric_bands


class MetricBandTests(unittest.TestCase):
    def test_cutoffs_are_inclusive_at_the_next_band(self):
        modes = pd.DataFrame(
            {
                "n_members": [0, 1, 2, 3, 10, None],
                "num_specs": [10] * 6,
                "homer_sim": [-0.1, 0.5999, 0.6, 0.7999, 0.8, None],
                "recovery": [0, 0.3999, 0.4, 0.6999, 0.7, None],
            }
        )
        result = add_metric_bands(modes)
        self.assertEqual(
            result.support_band.tolist(),
            ["Low", "Medium", "Medium", "High", "High", "Unavailable"],
        )
        for column in ["homer_sim_band", "recovery_band"]:
            self.assertEqual(
                result[column].tolist(),
                ["Low", "Low", "Medium", "Medium", "High", "No reference"],
            )

    def test_support_uses_each_rows_denominator(self):
        modes = pd.DataFrame(
            {
                "n_members": [4, 5, 14, 15, 5, 5],
                "num_specs": [47, 47, 47, 47, 10, 100],
                "homer_sim": [None] * 6,
                "recovery": [None] * 6,
            }
        )
        self.assertEqual(
            add_metric_bands(modes).support_band.tolist(),
            ["Low", "Medium", "Medium", "High", "High", "Low"],
        )

    def test_missing_and_invalid_scores_are_neutral(self):
        modes = pd.DataFrame(
            {
                "n_members": [1, 1, -1, 48, np.inf, pd.NA],
                "num_specs": [0, None, 47, 47, 47, 47],
                "homer_sim": [None, np.nan, pd.NA, np.inf, -np.inf, None],
                "recovery": [None, np.nan, pd.NA, np.inf, -np.inf, None],
            }
        )
        result = add_metric_bands(modes)
        self.assertEqual(result.support_band.tolist(), ["Unavailable"] * 6)
        self.assertEqual(result.homer_sim_band.tolist(), ["No reference"] * 6)
        self.assertEqual(result.recovery_band.tolist(), ["No reference"] * 6)

    def test_preserves_raw_values_indexes_and_modes_when_reapplied(self):
        modes = pd.DataFrame(
            {
                "n_members": [8, 3],
                "num_specs": [47, 47],
                "homer_sim": [0.8264, np.nan],
                "recovery": [0.6032, np.nan],
                "tier": ["primary", "alternate"],
                "pwm": [[[0.1, 0.2, 0.3, 0.4]], [[0.25] * 4]],
            },
            index=[9, 2],
        )
        original = modes.copy(deep=True)
        result = add_metric_bands(modes)
        pd.testing.assert_frame_equal(modes, original)
        pd.testing.assert_frame_equal(result[modes.columns], original)
        pd.testing.assert_frame_equal(add_metric_bands(result), result)


if __name__ == "__main__":
    unittest.main()
