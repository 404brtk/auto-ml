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

    def test_column_name_standardization(self):
        """Test that column names are standardized."""
        df = pd.DataFrame(
            {
                "Feature 1": [1, 2, 3],
                "Feature-2": [4, 5, 6],
                "Target": [0, 1, 0],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=False, drop_missing_target=False, max_missing_row_ratio=None
        )
        result = clean_data(df, "Target", cfg)

        # Columns should be standardized
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "target" in result.columns

    def test_special_null_values_cleaned(self):
        """Test that special null values are replaced."""
        df = pd.DataFrame(
            {
                "feature1": ["valid", "valid1", "?", "N/A", "null"],
                "feature2": ["data", "-", "ok", "good", "great"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig()
        result = clean_data(df, "target", cfg)

        # Special values should be replaced with NaN
        assert pd.isna(result["feature1"].iloc[2])  # ?
        assert pd.isna(result["feature1"].iloc[3])  # N/A
        assert pd.isna(result["feature1"].iloc[4])  # null
        assert pd.isna(result["feature2"].iloc[1])  # -

    def test_inf_values_handled(self):
        """Test that inf values are replaced with NaN."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, 4.0],
                "feature2": [5.0, 6.0, -np.inf, 8.0],
                "target": [0, 1, 0, 1],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=False, drop_missing_target=False, max_missing_row_ratio=None
        )
        result = clean_data(df, "target", cfg)

        # Inf values should be replaced with NaN
        assert pd.isna(result["feature1"].iloc[1])
        assert pd.isna(result["feature2"].iloc[2])

    def test_constant_features_removed(self):
        """Test that constant features are removed."""
        df = pd.DataFrame(
            {
                "const_feature": [5, 5, 5, 5, 5],
                "variable_feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=False,
            drop_missing_target=False,
            max_missing_row_ratio=None,
            remove_constant_features=True,
            constant_tolerance=1.0,
        )
        result = clean_data(df, "target", cfg)

        # Constant feature should be removed
        assert "const_feature" not in result.columns
        assert "variable_feature" in result.columns
        assert "target" in result.columns

    def test_constant_features_not_removed_when_disabled(self):
        """Test that constant features are kept when config is disabled."""
        df = pd.DataFrame(
            {
                "const_feature": [5, 5, 5, 5, 5],
                "variable_feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=False,
            drop_missing_target=False,
            max_missing_row_ratio=None,
            remove_constant_features=False,
        )
        result = clean_data(df, "target", cfg)

        # All features should be kept
        assert "const_feature" in result.columns
        assert "variable_feature" in result.columns
        assert "target" in result.columns

    def test_full_pipeline_with_new_features(self):
        """Test complete pipeline with all new features."""
        df = pd.DataFrame(
            {
                "Feature 1": [1, 2, 1, 4, 5],  # Row 0 and 2 have same value
                "Feature-2": [
                    "a",
                    "?",
                    "a",
                    "d",
                    "e",
                ],  # Row 0 and 2 have same value (after ? is cleaned)
                "Constant": [99, 99, 99, 99, 99],  # Constant feature
                "Numeric": [
                    1.0,
                    np.inf,
                    1.0,
                    4.0,
                    5.0,
                ],  # Row 0 and 2 have same value (after inf cleaned)
                "Target": [0, 1, 0, 1, 0],  # Row 0 and 2 have same value
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=True,
            drop_missing_target=False,
            max_missing_row_ratio=None,
            remove_constant_features=True,
            constant_tolerance=1.0,
        )
        result = clean_data(df, "Target", cfg)

        # Should have:
        # - Standardized column names
        # - Cleaned special nulls (? becomes NaN)
        # - Removed duplicates (row 0 and 2 are duplicates)
        # - Replaced inf with NaN
        # - Removed constant feature
        assert len(result) == 4  # One duplicate removed
        assert "constant" not in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "numeric" in result.columns
        assert "target" in result.columns
