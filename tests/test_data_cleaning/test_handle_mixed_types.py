"""Tests for handle_mixed_types function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import handle_mixed_types


class TestHandleMixedTypes:
    """Test handle_mixed_types function."""

    def test_coerce_strategy(self):
        """Test that coerce strategy converts mixed columns to string."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3.0, "four", 5],
                "normal_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = handle_mixed_types(df, strategy="coerce")

        # Mixed column should be converted to string
        assert result["mixed_col"].dtype == object
        assert result["mixed_col"].iloc[0] == "1"
        assert result["mixed_col"].iloc[1] == "two"

        # Normal column should remain unchanged
        assert result["normal_col"].dtype == df["normal_col"].dtype
        assert len(result.columns) == 3

    def test_drop_strategy(self):
        """Test that drop strategy removes mixed columns."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3.0, "four", 5],
                "normal_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = handle_mixed_types(df, strategy="drop")

        # Mixed column should be dropped
        assert "mixed_col" not in result.columns
        assert "normal_col" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_no_mixed_types(self):
        """Test behavior when there are no mixed types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
                "str_col": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result_coerce = handle_mixed_types(df, strategy="coerce")
        result_drop = handle_mixed_types(df, strategy="drop")

        # Nothing should change
        assert result_coerce.equals(df)
        assert result_drop.equals(df)

    def test_multiple_mixed_columns(self):
        """Test handling multiple mixed type columns."""
        df = pd.DataFrame(
            {
                "mixed1": [1, "two", 3],
                "mixed2": [4.0, "five", 6],
                "normal": [7, 8, 9],
                "target": [0, 1, 0],
            }
        )

        result_coerce = handle_mixed_types(df, strategy="coerce")
        result_drop = handle_mixed_types(df, strategy="drop")

        # Coerce: both mixed columns should be strings
        assert result_coerce["mixed1"].dtype == object
        assert result_coerce["mixed2"].dtype == object
        assert len(result_coerce.columns) == 4

        # Drop: both mixed columns should be removed
        assert "mixed1" not in result_drop.columns
        assert "mixed2" not in result_drop.columns
        assert "normal" in result_drop.columns
        assert len(result_drop.columns) == 2

    def test_mixed_with_nan(self):
        """Test mixed types with NaN values."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", np.nan, 4.0, "five"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = handle_mixed_types(df, strategy="coerce")

        # Should still detect as mixed and coerce
        assert result["mixed_col"].dtype == object
        assert pd.isna(result["mixed_col"].iloc[2])  # NaN should be preserved

    def test_all_nan_column(self):
        """Test that all-NaN columns are not detected as mixed."""
        df = pd.DataFrame(
            {
                "all_nan": [np.nan, np.nan, np.nan],
                "normal": [1, 2, 3],
                "target": [0, 1, 0],
            }
        )

        result = handle_mixed_types(df, strategy="drop")

        # All-NaN column should be kept (not detected as mixed)
        assert "all_nan" in result.columns
        assert len(result.columns) == 3

    def test_numeric_only_not_mixed(self):
        """Test that int/float mix is not considered mixed."""
        df = pd.DataFrame(
            {
                "int_float_mix": [1, 2.5, 3, 4.0],  # int and float
                "target": [0, 1, 0, 1],
            }
        )

        result = handle_mixed_types(df, strategy="drop")

        # Should not be detected as mixed (both are numeric)
        assert "int_float_mix" in result.columns

    def test_preserves_dataframe_copy(self):
        """Test that original dataframe is not modified."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3],
                "target": [0, 1, 0],
            }
        )

        original_dtypes = df.dtypes.copy()
        result = handle_mixed_types(df, strategy="coerce")

        # Original should be unchanged
        assert df.dtypes.equals(original_dtypes)
        assert not df.equals(result) or result["mixed_col"].dtype != object
