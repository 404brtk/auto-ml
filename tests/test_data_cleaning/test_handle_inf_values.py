"""Tests for handle_inf_values function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import handle_inf_values


class TestHandleInfValues:
    """Test handle_inf_values function."""

    def test_positive_inf_replacement(self):
        """Test replacement of positive infinity with NaN."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.inf, 3.0],
                "col2": [4, 5, 6],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["col1"].iloc[1])
        assert result["col1"].iloc[0] == 1.0
        assert result["col1"].iloc[2] == 3.0
        assert result["col2"].tolist() == [4, 5, 6]

    def test_negative_inf_replacement(self):
        """Test replacement of negative infinity with NaN."""
        df = pd.DataFrame(
            {
                "col1": [1.0, -np.inf, 3.0],
                "col2": [4, 5, 6],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["col1"].iloc[1])
        assert result["col1"].iloc[0] == 1.0

    def test_both_inf_types(self):
        """Test replacement of both positive and negative infinity."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.inf, -np.inf, 4.0],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col1"].iloc[2])
        assert result["col1"].iloc[0] == 1.0
        assert result["col1"].iloc[3] == 4.0

    def test_multiple_columns(self):
        """Test handling of infinity in multiple columns."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.inf, 3.0],
                "col2": [4.0, 5.0, -np.inf],
                "col3": [np.inf, -np.inf, 9.0],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col2"].iloc[2])
        assert pd.isna(result["col3"].iloc[0])
        assert pd.isna(result["col3"].iloc[1])

    def test_no_inf_values(self):
        """Test that normal values are not affected."""
        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [4, 5, 6],
            }
        )

        result = handle_inf_values(df)

        assert result["col1"].tolist() == [1.0, 2.0, 3.0]
        assert result["col2"].tolist() == [4, 5, 6]

    def test_non_numeric_columns_not_affected(self):
        """Test that non-numeric columns are not processed."""
        df = pd.DataFrame(
            {
                "num_col": [1.0, np.inf, 3.0],
                "str_col": ["a", "b", "c"],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["num_col"].iloc[1])
        assert result["str_col"].tolist() == ["a", "b", "c"]

    def test_no_numeric_columns(self):
        """Test behavior when there are no numeric columns."""
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c"],
                "col2": ["x", "y", "z"],
            }
        )

        result = handle_inf_values(df)

        # Should return unchanged DataFrame
        assert result.equals(df)

    def test_integer_columns(self):
        """Test that integer columns are not affected (can't have inf)."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, np.inf, 3.0],
            }
        )

        result = handle_inf_values(df)

        assert result["int_col"].tolist() == [1, 2, 3]
        assert pd.isna(result["float_col"].iloc[1])

    def test_existing_nan_values(self):
        """Test handling of existing NaN values."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, np.inf, 4.0],
            }
        )

        result = handle_inf_values(df)

        assert pd.isna(result["col1"].iloc[1])  # Original NaN
        assert pd.isna(result["col1"].iloc[2])  # Replaced inf

    def test_large_but_finite_values(self):
        """Test that large finite values are not replaced."""
        df = pd.DataFrame(
            {
                "col1": [1e100, 1e308, -1e100, -1e308],
            }
        )

        result = handle_inf_values(df)

        # Large but finite values should not be replaced
        assert not result["col1"].isna().any()

    def test_inf_from_operations(self):
        """Test handling of infinity created from operations."""
        df = pd.DataFrame(
            {
                "col1": [1.0, 0.0, 3.0],
            }
        )
        df["col2"] = 1.0 / df["col1"]  # Will create inf at index 1

        result = handle_inf_values(df)

        assert pd.isna(result["col2"].iloc[1])
        assert not pd.isna(result["col2"].iloc[0])
        assert not pd.isna(result["col2"].iloc[2])
