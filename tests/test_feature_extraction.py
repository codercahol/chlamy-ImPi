import unittest

import pandas as pd
import numpy as np

from chlamy_impi.feature_extraction.rolling_features import rolling_avg_y2_value
from chlamy_impi.feature_extraction.trend_features import y2_exponential_decay_time, fit_exponential_decay


class TestRollingFeatures(unittest.TestCase):

    def test_rolling_avg_y2_value(self):
        df = pd.DataFrame(
            {
                "y2_1": [1, 2, 3, 4, 5],
                "y2_2": [6, 7, 8, 9, 10],
                "y2_3": [11, 12, 13, 14, 15],
                "y2_4": [16, 17, 18, 19, 20],
                "y2_5": [21, 22, 23, 24, 25],
                "y2_6": [26, 27, 28, 29, 30],
            }
        )
        result = rolling_avg_y2_value(df)
        expected = pd.DataFrame(
            {
                0: [np.nan, 16.0],
                1: [np.nan, 17.0],
                2: [np.nan, 18.0],
                3: [np.nan, 19.0],
                4: [np.nan, 20.0],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


class TestTrendFeatures(unittest.TestCase):

    def test_exponential_decay(self):
        x = np.array(range(10))
        y = np.exp(-x / 2)
        params, covariance = fit_exponential_decay(pd.Series(y))
        self.assertAlmostEqual(params[1], 2.0)

    def test_y2_exponential_decay_time(self):
        df = pd.DataFrame(
            {
                "y2_1": [10, 20],
                "y2_2": [5, 10],
                "y2_3": [2.5, 5],
                "y2_4": [1.25, 2.5],
                "y2_5": [0.625, 1.25],
            }
        )
        result = y2_exponential_decay_time(df)
        expected = pd.Series([1.44, 1.44])
        pd.testing.assert_series_equal(result, expected, atol=0.01, check_names=False)


