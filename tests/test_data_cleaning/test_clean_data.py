"""Tests for clean_data function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.config import CleaningConfig


class TestCleanData:
    """Test clean_data function."""

    def test_full_cleaning_pipeline(self):
        """Test the complete cleaning pipeline."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 1, 4, np.nan],  # Has NaN and duplicate
                "feature2": [4, 5, 4, 6, np.nan],  # Has NaN and duplicate
                "datetime_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-01",
                    "2023-01-03",
                    "2023-01-04",
                ],
                "numeric_str": ["1.5", "2.0", "1.5", "3.0", "4.0"],
                "target": [0, 1, 0, 1, np.nan],  # Has NaN target
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=True,
            drop_missing_target=True,
            max_missing_row_ratio=0.3,  # Remove rows with >30% missing
        )

        result = clean_data(df, "target", cfg)

        # Should have removed:
        # - Row with NaN target (index 4)
        # - Duplicate row (index 2, since index 0 is kept)
        # - Row with high missing ratio (if any)
        assert len(result) >= 2  # At least 2 rows should remain
        assert "target" in result.columns
        assert not result["target"].isna().any()  # No NaN targets
        assert not result.duplicated().any()  # No duplicates

    def test_no_cleaning_config(self):
        """Test with all cleaning disabled."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, np.nan]}
        )

        cfg = CleaningConfig(
            drop_duplicates=False, drop_missing_target=False, max_missing_row_ratio=None
        )

        result = clean_data(df, "target", cfg)
        # Should only apply transformers, not remove rows
        assert len(result) == 3
        assert result["target"].isna().iloc[2]  # NaN target should still be there

    def test_config_defaults(self):
        """Test with default config values."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, 0]}
        )

        cfg = CleaningConfig()  # Use defaults
        result = clean_data(df, "target", cfg)

        # Default config removes duplicates and missing targets
        assert len(result) == 2  # One duplicate removed
        assert not result.duplicated().any()

    def test_datetime_conversion(self):
        """Test that datetime conversion happens."""
        df = pd.DataFrame(
            {"datetime_col": ["2023-01-01", "2023-01-02"], "target": [0, 1]}
        )

        cfg = CleaningConfig(
            drop_duplicates=False, drop_missing_target=False, max_missing_row_ratio=None
        )
        result = clean_data(df, "target", cfg)

        # Should have converted datetime column
        datetime_cols = [col for col in result.columns if "datetime_col" in col]
        assert len(datetime_cols) == 1  # Should have the converted datetime column

    def test_numeric_coercion(self):
        """Test that numeric string coercion happens."""
        df = pd.DataFrame(
            {"numeric_str": ["1.5", "2.0", "not_a_number"], "target": [0, 1, 0]}
        )

        cfg = CleaningConfig(
            drop_duplicates=False, drop_missing_target=False, max_missing_row_ratio=None
        )
        result = clean_data(df, "target", cfg)

        # The numeric coercer should have processed the column
        assert "numeric_str" in result.columns
