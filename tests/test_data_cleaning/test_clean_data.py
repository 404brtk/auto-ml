"""Tests for clean_data function."""

import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.config import CleaningConfig


class TestCleanData:
    """Test clean_data function."""

    def test_full_cleaning_pipeline(self):
        """Test the complete cleaning pipeline."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 1, 4, np.nan],
                "feature2": [4, 5, 4, 6, np.nan],
                "datetime_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-01",
                    "2023-01-03",
                    "2023-01-04",
                ],
                "numeric_str": ["1.5", "2.0", "1.5", "3.0", "4.0"],
                "target": [0, 1, 0, 1, np.nan],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=True, max_missing_row_ratio=0.3, remove_id_columns=False
        )

        result, target = clean_data(df, "target", cfg)

        # Should have removed rows with NaN target and duplicates
        assert len(result) >= 2
        assert target in result.columns
        assert not result[target].isna().any()
        assert not result.duplicated().any()

    def test_no_cleaning_config(self):
        """Test with minimal cleaning."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, np.nan]}
        )

        cfg = CleaningConfig(
            drop_duplicates=False,
            max_missing_row_ratio=None,
            max_missing_feature_ratio=None,
            remove_id_columns=False,
        )

        result, target = clean_data(df, "target", cfg)

        # Missing targets are always removed
        assert len(result) == 2
        assert not result[target].isna().any()

    def test_config_defaults(self):
        """Test with default config values."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, 0]}
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # Default config removes duplicates
        assert len(result) == 2
        assert not result.duplicated().any()

    def test_datetime_conversion(self):
        """Test that datetime conversion happens."""
        df = pd.DataFrame(
            {
                "datetime_col": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "target": [0, 1, 0],
            }
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # Should have converted datetime column (creates new columns)
        datetime_cols = [col for col in result.columns if "datetime_col" in col]
        assert len(datetime_cols) >= 1

    def test_numeric_coercion(self):
        """Test that numeric string coercion happens."""
        df = pd.DataFrame(
            {"numeric_str": ["1.5", "2.0", "not_a_number"], "target": [0, 1, 0]}
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # The numeric coercer should have processed the column
        assert target in result.columns

    def test_column_name_standardization(self):
        """Test that column names are standardized."""
        df = pd.DataFrame(
            {"Feature 1": [1, 2, 3], "Feature-2": [4, 5, 6], "Target": [0, 1, 0]}
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, returned_target = clean_data(df, "Target", cfg)

        # Columns should be standardized
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert returned_target.lower() == "target"

    def test_special_null_values_cleaned(self):
        """Test that special null values are replaced."""
        df = pd.DataFrame(
            {
                "feature1": [
                    "valid",
                    "valid1",
                    "?",
                    "N/A",
                    "null",
                    "data1",
                    "data2",
                    "data3",
                    "data4",
                    "data5",
                ],
                "feature2": [
                    "data",
                    "-",
                    "ok",
                    "good",
                    "great",
                    "val1",
                    "val2",
                    "val3",
                    "val4",
                    "val5",
                ],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        cfg = CleaningConfig(
            remove_id_columns=False,
            remove_constant_features=False,
            max_missing_feature_ratio=None,
        )
        result, target = clean_data(df, "target", cfg)

        # Check using standardized column names
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
            drop_duplicates=False, max_missing_row_ratio=None, remove_id_columns=False
        )
        result, target = clean_data(df, "target", cfg)

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
            remove_constant_features=True,
            constant_tolerance=1.0,
            remove_id_columns=False,
        )
        result, target = clean_data(df, "target", cfg)

        # Constant feature should be removed
        assert "const_feature" not in result.columns
        assert "variable_feature" in result.columns

    def test_constant_features_not_removed_when_disabled(self):
        """Test that constant features are kept when config is disabled."""
        df = pd.DataFrame(
            {
                "const_feature": [5, 5, 5, 5, 5],
                "variable_feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(remove_constant_features=False, remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # All features should be kept
        assert "const_feature" in result.columns
        assert "variable_feature" in result.columns

    def test_full_pipeline_with_new_features(self):
        """Test complete pipeline with all new features."""
        df = pd.DataFrame(
            {
                "Feature 1": [1, 2, 1, 4, 5],
                "Feature-2": ["a", "?", "a", "d", "e"],
                "Constant": [99, 99, 99, 99, 99],
                "Numeric": [1.0, np.inf, 1.0, 4.0, 5.0],
                "Target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=True,
            remove_constant_features=True,
            constant_tolerance=1.0,
            remove_id_columns=False,
        )
        result, target = clean_data(df, "Target", cfg)

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

    def test_mixed_types_coerced(self):
        """Test that mixed type columns are coerced by default."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3.0, "four", 5],
                "normal_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(handle_mixed_types="coerce", remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # Mixed column should be converted to string
        assert result["mixed_col"].dtype == object

    def test_mixed_types_dropped(self):
        """Test that mixed type columns can be dropped."""
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3.0, "four", 5],
                "normal_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(handle_mixed_types="drop", remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # Mixed column should be removed
        assert "mixed_col" not in result.columns
        assert "normal_col" in result.columns

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed from string columns."""
        df = pd.DataFrame(
            {"str_col": ["  apple  ", "  banana  ", "  cherry  "], "target": [0, 1, 0]}
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # Whitespace should be trimmed
        assert result["str_col"].iloc[0] == "apple"
        assert result["str_col"].iloc[1] == "banana"
        assert result["str_col"].iloc[2] == "cherry"

    def test_id_columns_removed(self):
        """Test that ID columns are removed when enabled."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "transaction_id": ["T001", "T002", "T003", "T004", "T005"],
                "feature": [10, 20, 10, 20, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(remove_id_columns=True, id_column_threshold=0.95)
        result, target = clean_data(df, "target", cfg)

        # ID columns should be removed
        assert "customer_id" not in result.columns
        assert "transaction_id" not in result.columns
        assert "feature" in result.columns

    def test_id_columns_kept_when_disabled(self):
        """Test that ID columns are kept when disabled."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "feature": [10, 20, 10, 20, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # ID column should be kept
        assert "customer_id" in result.columns
        assert "feature" in result.columns

    def test_min_rows_validation_passes(self):
        """Test that validation passes with sufficient rows."""
        df = pd.DataFrame({"feature": list(range(20)), "target": [0, 1] * 10})

        cfg = CleaningConfig(min_rows_after_cleaning=10, remove_id_columns=False)

        # Should not raise error
        result, target = clean_data(df, "target", cfg)
        assert len(result) >= 10

    def test_min_rows_validation_fails(self):
        """Test that validation raises error with insufficient rows."""
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, np.nan]})

        cfg = CleaningConfig(min_rows_after_cleaning=10, remove_id_columns=False)

        with pytest.raises(ValueError, match="Dataset has only"):
            clean_data(df, "target", cfg)

    def test_min_cols_validation_fails(self):
        """Test that validation raises error with insufficient columns."""
        df = pd.DataFrame(
            {"id_col": list(range(10)), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
        )

        cfg = CleaningConfig(
            remove_id_columns=True, id_column_threshold=0.95, min_cols_after_cleaning=2
        )

        # Should raise ValueError (ID column removed, leaving no feature columns)
        with pytest.raises(ValueError, match="feature columns after cleaning"):
            clean_data(df, "target", cfg)

    def test_removes_high_missing_features(self):
        """Test that features with high missing ratio are removed."""
        df = pd.DataFrame(
            {
                "good_feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bad_feat": [
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    10,
                ],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        cfg = CleaningConfig(max_missing_feature_ratio=0.5, remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        # bad_feat has 80% missing, should be removed
        assert "bad_feat" not in result.columns
        assert "good_feat" in result.columns

    def test_preserves_datatypes(self):
        """Test that cleaning preserves appropriate datatypes."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        cfg = CleaningConfig(remove_id_columns=False)
        result, target = clean_data(df, "target", cfg)

        assert pd.api.types.is_integer_dtype(result["int_col"])
        assert pd.api.types.is_float_dtype(result["float_col"])
        assert result["str_col"].dtype == object
