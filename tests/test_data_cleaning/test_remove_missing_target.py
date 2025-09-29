"""Tests for remove_missing_target function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import remove_missing_target


class TestRemoveMissingTarget:
    """Test remove_missing_target function."""

    def test_no_missing_target(self):
        """Test with no missing target values."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        result = remove_missing_target(df, "target")
        pd.testing.assert_frame_equal(result, df)

    def test_some_missing_target(self):
        """Test with some missing target values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [4, 5, 6, 7],
                "target": [0, np.nan, 1, 0],
            }
        )

        expected = pd.DataFrame(
            {"feature1": [1, 3, 4], "feature2": [4, 6, 7], "target": [0.0, 1.0, 0.0]},
            index=[0, 2, 3],
        )

        result = remove_missing_target(df, "target")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_all_missing_target(self):
        """Test with all target values missing."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "target": [np.nan, np.nan, np.nan],
            }
        )

        result = remove_missing_target(df, "target")
        assert len(result) == 0
        assert list(result.columns) == ["feature1", "feature2", "target"]
