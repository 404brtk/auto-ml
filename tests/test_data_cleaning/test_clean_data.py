"""Tests for clean_data function."""

import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.config import CleaningConfig


@pytest.fixture
def basic_config():
    return CleaningConfig(remove_id_columns=False)


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
    )


@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame(
        {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, 0]}
    )


class TestCleanData:
    def test_full_cleaning_pipeline(self):
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

        result, target, _ = clean_data(df, "target", cfg)

        assert len(result) >= 2
        assert not result[target].isna().any()
        assert not result.duplicated().any()

    def test_minimal_cleaning(self):
        df = pd.DataFrame(
            {"feature1": [1, 2, 1], "feature2": [4, 5, 4], "target": [0, 1, np.nan]}
        )
        cfg = CleaningConfig(
            drop_duplicates=False,
            max_missing_row_ratio=None,
            max_missing_feature_ratio=None,
            remove_id_columns=False,
        )

        result, target, _ = clean_data(df, "target", cfg)

        assert len(result) == 2
        assert not result[target].isna().any()

    def test_duplicate_removal(self, df_with_duplicates):
        cfg = CleaningConfig(drop_duplicates=True, remove_id_columns=False)
        result, target, _ = clean_data(df_with_duplicates, "target", cfg)

        assert len(result) == 2
        assert not result.duplicated().any()

    def test_column_name_standardization(self):
        df = pd.DataFrame(
            {"Feature 1": [1, 2, 3], "Feature-2": [4, 5, 6], "Target": [0, 1, 0]}
        )
        cfg = CleaningConfig(remove_id_columns=False)

        result, target, mapping = clean_data(df, "Target", cfg)

        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert target.lower() == "target"
        assert "Feature 1" in mapping

    def test_special_null_values_replaced(self):
        df = pd.DataFrame(
            {
                "feature1": ["valid", "valid1", "?", "N/A", "null"] + ["data"] * 5,
                "feature2": ["data", "-", "ok", "good", "great"] + ["val"] * 5,
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        cfg = CleaningConfig(
            remove_id_columns=False,
            remove_constant_features=False,
            max_missing_feature_ratio=None,
        )

        result, target, _ = clean_data(df, "target", cfg)

        assert pd.isna(result["feature1"].iloc[2])
        assert pd.isna(result["feature1"].iloc[3])
        assert pd.isna(result["feature2"].iloc[1])

    def test_inf_values_replaced_with_nan(self):
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

        result, target, _ = clean_data(df, "target", cfg)

        assert pd.isna(result["feature1"].iloc[1])
        assert pd.isna(result["feature2"].iloc[2])

    @pytest.mark.parametrize(
        "remove_constant,should_exist",
        [
            (True, False),
            (False, True),
        ],
    )
    def test_constant_feature_removal(self, remove_constant, should_exist):
        df = pd.DataFrame(
            {
                "const_feature": [5] * 5,
                "variable_feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        cfg = CleaningConfig(
            remove_constant_features=remove_constant,
            constant_tolerance=1.0,
            remove_id_columns=False,
        )

        result, target, _ = clean_data(df, "target", cfg)

        assert ("const_feature" in result.columns) == should_exist
        assert "variable_feature" in result.columns

    @pytest.mark.parametrize(
        "strategy,should_exist",
        [
            ("coerce", True),
            ("drop", False),
        ],
    )
    def test_mixed_types_handling(self, strategy, should_exist):
        df = pd.DataFrame(
            {
                "mixed_col": [1, "two", 3.0, "four", 5],
                "normal_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        cfg = CleaningConfig(handle_mixed_types=strategy, remove_id_columns=False)

        result, target, _ = clean_data(df, "target", cfg)

        assert ("mixed_col" in result.columns) == should_exist
        assert "normal_col" in result.columns

    def test_whitespace_trimmed_from_strings(self):
        df = pd.DataFrame(
            {"str_col": ["  apple  ", "  banana  ", "  cherry  "], "target": [0, 1, 0]}
        )
        cfg = CleaningConfig(remove_id_columns=False)

        result, target, _ = clean_data(df, "target", cfg)

        assert result["str_col"].tolist() == ["apple", "banana", "cherry"]

    @pytest.mark.parametrize(
        "remove_id,should_exist",
        [
            (True, False),
            (False, True),
        ],
    )
    def test_id_column_removal(self, remove_id, should_exist):
        df = pd.DataFrame(
            {
                "customer_id": [f"C{i:03d}" for i in range(5)],
                "transaction_id": [f"T{i:03d}" for i in range(5)],
                "feature": [10, 20, 10, 20, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )
        cfg = CleaningConfig(remove_id_columns=remove_id, id_column_threshold=0.95)

        result, target, _ = clean_data(df, "target", cfg)

        assert ("customer_id" in result.columns) == should_exist
        assert "feature" in result.columns

    def test_min_rows_validation_passes(self):
        df = pd.DataFrame({"feature": list(range(20)), "target": [0, 1] * 10})
        cfg = CleaningConfig(min_rows_after_cleaning=10, remove_id_columns=False)

        result, target, _ = clean_data(df, "target", cfg)

        assert len(result) >= 10

    def test_min_rows_validation_fails(self):
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, np.nan]})
        cfg = CleaningConfig(min_rows_after_cleaning=10, remove_id_columns=False)

        with pytest.raises(ValueError, match="Dataset has only"):
            clean_data(df, "target", cfg)

    def test_min_cols_validation_fails(self):
        df = pd.DataFrame({"id_col": list(range(10)), "target": [0, 1] * 5})
        cfg = CleaningConfig(
            remove_id_columns=True, id_column_threshold=0.95, min_cols_after_cleaning=2
        )

        with pytest.raises(ValueError, match="feature columns after cleaning"):
            clean_data(df, "target", cfg)

    def test_high_missing_features_removed(self):
        df = pd.DataFrame(
            {
                "good_feat": list(range(10)),
                "bad_feat": [1] + [np.nan] * 8 + [10],
                "target": [0, 1] * 5,
            }
        )
        cfg = CleaningConfig(max_missing_feature_ratio=0.5, remove_id_columns=False)

        result, target, _ = clean_data(df, "target", cfg)

        assert "bad_feat" not in result.columns
        assert "good_feat" in result.columns

    def test_datatypes_preserved(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )
        cfg = CleaningConfig(remove_id_columns=False)

        result, target, _ = clean_data(df, "target", cfg)

        assert pd.api.types.is_integer_dtype(result["int_col"])
        assert pd.api.types.is_float_dtype(result["float_col"])
        assert result["str_col"].dtype == object
