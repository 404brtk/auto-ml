"""Tests for remove_high_missing_rows function."""

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

    def test_all_rows_missing(self):
        """Test when all feature rows have missing values."""
        df = pd.DataFrame(
            {
                "feature1": [np.nan, np.nan, np.nan],
                "feature2": [np.nan, np.nan, np.nan],
                "target": [0, 1, 0],
            }
        )

        result = remove_high_missing_rows(df, "target", 0.9)
        # All rows should be removed since all have 100% missing
        assert len(result) == 0

    def test_single_feature_column(self):
        """Test with only one feature column."""
        df = pd.DataFrame({"feature1": [1, np.nan, 3], "target": [0, 1, 0]})

        result = remove_high_missing_rows(df, "target", 0.5)
        # Row with NaN should be removed (100% missing)
        assert len(result) == 2

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"target": []})

        result = remove_high_missing_rows(df, "target", 0.5)
        assert len(result) == 0

    def test_exact_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        df = pd.DataFrame(
            {
                "feature1": [1, np.nan],
                "feature2": [2, 3],
                "target": [0, 1],
            }
        )
        # Row 1: 1/2 = 0.5 missing ratio

        # At threshold 0.5, should keep the row (<=)
        result_inclusive = remove_high_missing_rows(df, "target", 0.5)
        assert len(result_inclusive) == 2

        # At threshold 0.49, should remove the row (>)
        result_exclusive = remove_high_missing_rows(df, "target", 0.49)
        assert len(result_exclusive) == 1
