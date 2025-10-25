"""Tests for standardize_column_names function."""

import pandas as pd
import pytest
from auto_ml_pipeline.data_cleaning import standardize_column_names


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {
            "Feature 1": [1, 2, 3],
            "Feature-2": [4, 5, 6],
            "Feature.3": [7, 8, 9],
        }
    )


class TestStandardizeColumnNames:
    def test_basic_standardization(self, basic_df):
        result, target, mapping = standardize_column_names(basic_df, "Feature 1")

        assert list(result.columns) == ["feature_1", "feature_2", "feature_3"]
        assert target == "feature_1"
        assert mapping["Feature 1"] == "feature_1"

    @pytest.mark.parametrize("special_char", ["@", "#", "$", "%"])
    def test_special_characters_removed(self, special_char):
        df = pd.DataFrame({f"Feature{special_char}1": [1, 2, 3]})
        result, _, _ = standardize_column_names(df, f"Feature{special_char}1")

        assert result.columns[0] == "feature_1"

    def test_leading_number_gets_prefix(self):
        df = pd.DataFrame(
            {
                "1st_feature": [1, 2, 3],
                "2nd_feature": [4, 5, 6],
            }
        )
        result, target, _ = standardize_column_names(df, "1st_feature")

        assert result.columns[0] == "col_1st_feature"
        assert target == "col_1st_feature"

    def test_duplicate_names_get_suffix(self):
        df = pd.DataFrame(
            {
                "Feature": [1, 2, 3],
                "Feature ": [4, 5, 6],
                "Feature.": [7, 8, 9],
            }
        )
        result, target, _ = standardize_column_names(df, "Feature")

        assert len(set(result.columns)) == 3
        assert list(result.columns) == ["feature", "feature_1", "feature_2"]
        assert target == "feature"

    @pytest.mark.parametrize("empty_name", ["", " ", "@"])
    def test_empty_names_become_unnamed(self, empty_name):
        df = pd.DataFrame({empty_name: [1, 2, 3]})
        result, target, _ = standardize_column_names(df, empty_name)

        assert "unnamed" in result.columns[0]
        assert "unnamed" in target

    def test_case_conversion_to_lowercase(self):
        df = pd.DataFrame(
            {
                "FEATURE1": [1, 2, 3],
                "Feature2": [4, 5, 6],
                "FeAtUrE3": [7, 8, 9],
            }
        )
        result, target, _ = standardize_column_names(df, "FEATURE1")

        assert list(result.columns) == ["feature1", "feature2", "feature3"]
        assert target == "feature1"

    def test_underscores_stripped_and_deduplicated(self):
        df = pd.DataFrame(
            {
                "_feature": [1, 2, 3],
                "feature_": [4, 5, 6],
                "__feature__": [7, 8, 9],
            }
        )
        result, target, _ = standardize_column_names(df, "_feature")

        assert list(result.columns) == ["feature", "feature_1", "feature_2"]
        assert target == "feature"

    def test_preserves_already_valid_names(self):
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "target": [7, 8, 9],
            }
        )
        result, target, _ = standardize_column_names(df, "feature_1")

        assert list(result.columns) == ["feature_1", "feature_2", "target"]
        assert target == "feature_1"

    def test_data_values_unchanged(self, basic_df):
        result, _, _ = standardize_column_names(basic_df, "Feature 1")

        assert result["feature_1"].tolist() == [1, 2, 3]
        assert result["feature_2"].tolist() == [4, 5, 6]

    def test_numeric_column_names_get_prefix(self):
        df = pd.DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
        result, target, _ = standardize_column_names(df, 1)

        assert list(result.columns) == ["col_1", "col_2"]
        assert target == "col_1"

    def test_mapping_contains_all_columns(self, basic_df):
        _, _, mapping = standardize_column_names(basic_df, "Feature 1")

        assert len(mapping) == 3
        assert "Feature 1" in mapping
        assert "Feature-2" in mapping
        assert "Feature.3" in mapping
