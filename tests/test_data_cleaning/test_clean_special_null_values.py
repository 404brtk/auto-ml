"""Tests for clean_special_null_values function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import clean_special_null_values


class TestCleanSpecialNullValues:
    """Test clean_special_null_values function."""

    def test_question_mark_replacement(self):
        """Test replacement of ? with NaN."""
        df = pd.DataFrame(
            {
                "col1": ["a", "?", "c"],
                "col2": [1, 2, 3],  # Numeric, should not be affected
            }
        )

        special_values = ["?"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[1])
        assert result["col1"].iloc[0] == "a"
        assert result["col2"].tolist() == [1, 2, 3]

    def test_multiple_special_values(self):
        """Test replacement of multiple special values."""
        df = pd.DataFrame(
            {
                "col1": ["a", "?", "N/A", "null", "c"],
                "col2": ["valid", "-", "--", "None", "data"],
            }
        )

        special_values = ["?", "N/A", "null", "-", "--", "None"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[1])  # ?
        assert pd.isna(result["col1"].iloc[2])  # N/A
        assert pd.isna(result["col1"].iloc[3])  # null
        assert pd.isna(result["col2"].iloc[1])  # -
        assert pd.isna(result["col2"].iloc[2])  # --
        assert pd.isna(result["col2"].iloc[3])  # None

    def test_case_insensitive(self):
        """Test that replacement is case-insensitive."""
        df = pd.DataFrame(
            {
                "col1": ["NULL", "null", "Null", "NuLl"],
            }
        )

        special_values = ["null"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[0])
        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col1"].iloc[2])
        assert pd.isna(result["col1"].iloc[3])

    def test_whitespace_handling(self):
        """Test handling of whitespace around values."""
        df = pd.DataFrame(
            {
                "col1": [" ? ", "  null  ", "N/A  ", "  -  "],
            }
        )

        special_values = ["?", "null", "N/A", "-"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[0])
        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col1"].iloc[2])
        assert pd.isna(result["col1"].iloc[3])

    def test_empty_string_replacement(self):
        """Test replacement of empty strings."""
        df = pd.DataFrame(
            {
                "col1": ["a", "", "c"],
                "col2": ["x", " ", "z"],  # Space
            }
        )

        special_values = ["", " "]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col2"].iloc[1])

    def test_numeric_columns_not_affected(self):
        """Test that numeric columns are not processed."""
        df = pd.DataFrame(
            {
                "num_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "?", "c"],
            }
        )

        special_values = ["?"]
        result = clean_special_null_values(df, special_values)

        assert result["num_col"].tolist() == [1, 2, 3]
        assert result["float_col"].tolist() == [1.5, 2.5, 3.5]
        assert pd.isna(result["str_col"].iloc[1])

    def test_no_object_columns(self):
        """Test behavior when there are no object columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4.5, 5.5, 6.5],
            }
        )

        special_values = ["?", "null"]
        result = clean_special_null_values(df, special_values)

        # Should return unchanged DataFrame
        assert result.equals(df)

    def test_already_nan_values(self):
        """Test handling of existing NaN values."""
        df = pd.DataFrame(
            {
                "col1": ["a", np.nan, "?", "c"],
            }
        )

        special_values = ["?"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["col1"].iloc[1])  # Original NaN
        assert pd.isna(result["col1"].iloc[2])  # Replaced ?

    def test_partial_match_not_replaced(self):
        """Test that partial matches are not replaced."""
        df = pd.DataFrame(
            {
                "col1": ["?question", "null_value", "N/A_test"],
            }
        )

        special_values = ["?", "null", "N/A"]
        result = clean_special_null_values(df, special_values)

        # Should not be replaced because they're not exact matches
        assert result["col1"].iloc[0] == "?question"
        assert result["col1"].iloc[1] == "null_value"
        assert result["col1"].iloc[2] == "N/A_test"

    def test_comprehensive_special_values(self):
        """Test with comprehensive list of special values."""
        df = pd.DataFrame(
            {
                "col1": [
                    "?",
                    "N/A",
                    "NA",
                    "null",
                    "None",
                    "nan",
                    "missing",
                    "-",
                    "--",
                    "---",
                ],
            }
        )

        special_values = [
            "?",
            "N/A",
            "n/a",
            "NA",
            "null",
            "NULL",
            "None",
            "none",
            "nan",
            "NaN",
            "NAN",
            "undefined",
            "missing",
            "MISSING",
            "-",
            "--",
            "---",
            "",
            " ",
        ]
        result = clean_special_null_values(df, special_values)

        # All should be NaN
        assert result["col1"].isna().all()

    def test_category_dtype(self):
        """Test handling of categorical columns."""
        df = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["a", "?", "c", "null"]),
            }
        )

        special_values = ["?", "null"]
        result = clean_special_null_values(df, special_values)

        assert pd.isna(result["cat_col"].iloc[1])
        assert pd.isna(result["cat_col"].iloc[3])
