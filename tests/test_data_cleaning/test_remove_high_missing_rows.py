import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.data_cleaning import remove_high_missing_rows


class TestRemoveHighMissingRows:
    """Test remove_high_missing_rows function."""

    def test_no_rows_removed(self):
        """Test when no rows exceed the threshold."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "target": [0, 1, 0],
            }
        )

        result = remove_high_missing_rows(df, "target", 0.5)
        pd.testing.assert_frame_equal(result, df)

    def test_remove_rows_above_threshold(self):
        """Test removing rows with missing ratio above threshold."""
        df = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan],  # 0, 1, 1 missing
                "feature2": [4, 5, np.nan],  # 0, 0, 1 missing
                "target": [0, 1, 0],
            }
        )
        # Row 0: 0/2 = 0.0 missing ratio
        # Row 1: 1/2 = 0.5 missing ratio
        # Row 2: 2/2 = 1.0 missing ratio

        result = remove_high_missing_rows(
            df, "target", 0.3
        )  # Remove rows with >30% missing
        expected = df.iloc[[0]].reset_index(drop=True)  # Keep only row 0

        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_threshold_zero(self):
        """Test with threshold 0 (remove any missing)."""
        df = pd.DataFrame(
            {"feature1": [1, np.nan, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        result = remove_high_missing_rows(df, "target", 0.0)
        expected = df.iloc[[0, 2]].reset_index(drop=True)

        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_threshold_one(self):
        """Test with threshold 1 (keep all rows)."""
        df = pd.DataFrame(
            {
                "feature1": [np.nan, np.nan, np.nan],
                "feature2": [np.nan, np.nan, np.nan],
                "target": [0, 1, 0],
            }
        )

        result = remove_high_missing_rows(df, "target", 1.0)
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})

        with pytest.raises(ValueError, match="max_missing_ratio must be in"):
            remove_high_missing_rows(df, "target", -0.1)

        with pytest.raises(ValueError, match="max_missing_ratio must be in"):
            remove_high_missing_rows(df, "target", 1.1)
