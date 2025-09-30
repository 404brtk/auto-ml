"""Tests for standardize_column_names function."""

import pandas as pd
from auto_ml_pipeline.data_cleaning import standardize_column_names


class TestStandardizeColumnNames:
    """Test standardize_column_names function."""

    def test_basic_standardization(self):
        """Test basic column name standardization."""
        df = pd.DataFrame(
            {
                "Feature 1": [1, 2, 3],
                "Feature-2": [4, 5, 6],
                "Feature.3": [7, 8, 9],
            }
        )

        result = standardize_column_names(df)

        assert list(result.columns) == ["feature_1", "feature_2", "feature_3"]

    def test_special_characters(self):
        """Test handling of special characters."""
        df = pd.DataFrame(
            {
                "Feature@1": [1, 2, 3],
                "Feature#2": [4, 5, 6],
                "Feature$3": [7, 8, 9],
                "Feature%4": [10, 11, 12],
            }
        )

        result = standardize_column_names(df)

        for col in result.columns:
            # Should only contain lowercase letters, numbers, and underscores
            assert (
                col.replace("_", "").replace("col", "").replace("feature", "").isalnum()
                or col == "_"
            )

    def test_leading_number(self):
        """Test columns starting with numbers."""
        df = pd.DataFrame(
            {
                "1st_feature": [1, 2, 3],
                "2nd_feature": [4, 5, 6],
            }
        )

        result = standardize_column_names(df)

        # Should prefix with 'col_'
        assert result.columns[0] == "col_1st_feature"
        assert result.columns[1] == "col_2nd_feature"

    def test_duplicate_names(self):
        """Test handling of duplicate column names."""
        df = pd.DataFrame(
            {
                "Feature": [1, 2, 3],
                "Feature ": [4, 5, 6],  # Space will be removed, creating duplicate
                "Feature.": [7, 8, 9],  # Period will be removed, creating duplicate
            }
        )

        result = standardize_column_names(df)

        # Should have unique names with suffixes
        assert len(result.columns) == 3
        assert len(set(result.columns)) == 3  # All unique
        assert result.columns[0] == "feature"
        assert result.columns[1] == "feature_1"
        assert result.columns[2] == "feature_2"

    def test_empty_name(self):
        """Test handling of empty or invalid names."""
        df = pd.DataFrame(
            {
                "": [1, 2, 3],
                " ": [4, 5, 6],
                "@": [7, 8, 9],
            }
        )

        result = standardize_column_names(df)

        # Should replace with 'unnamed'
        assert "unnamed" in result.columns[0]

    def test_uppercase_to_lowercase(self):
        """Test conversion to lowercase."""
        df = pd.DataFrame(
            {
                "FEATURE1": [1, 2, 3],
                "Feature2": [4, 5, 6],
                "FeAtUrE3": [7, 8, 9],
            }
        )

        result = standardize_column_names(df)

        assert list(result.columns) == ["feature1", "feature2", "feature3"]

    def test_leading_trailing_underscores(self):
        """Test removal of leading/trailing underscores."""
        df = pd.DataFrame(
            {
                "_feature": [1, 2, 3],
                "feature_": [4, 5, 6],
                "__feature__": [7, 8, 9],
            }
        )

        result = standardize_column_names(df)

        # All three become "feature" after stripping underscores
        # So we get: "feature", "feature_1", "feature_2"
        assert result.columns[0] == "feature"
        assert result.columns[1] == "feature_1"
        assert result.columns[2] == "feature_2"

    def test_preserves_valid_names(self):
        """Test that already valid names are preserved."""
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "target": [7, 8, 9],
            }
        )

        result = standardize_column_names(df)

        assert list(result.columns) == ["feature_1", "feature_2", "target"]

    def test_no_data_modification(self):
        """Test that data values are not modified."""
        df = pd.DataFrame(
            {
                "Feature 1": [1, 2, 3],
                "Feature-2": [4, 5, 6],
            }
        )

        result = standardize_column_names(df)

        assert result["feature_1"].tolist() == [1, 2, 3]
        assert result["feature_2"].tolist() == [4, 5, 6]

    def test_numeric_column_names(self):
        """Test handling of numeric column names."""
        df = pd.DataFrame(
            {
                1: [1, 2, 3],
                2: [4, 5, 6],
            }
        )

        result = standardize_column_names(df)

        # Should convert to string and add prefix
        assert result.columns[0] == "col_1"
        assert result.columns[1] == "col_2"
