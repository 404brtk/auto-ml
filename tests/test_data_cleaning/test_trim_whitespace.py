"""Tests for trim_whitespace function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import trim_whitespace


class TestTrimWhitespace:
    """Test trim_whitespace function."""

    def test_trim_leading_whitespace(self):
        """Test trimming leading whitespace."""
        df = pd.DataFrame(
            {
                "col1": ["  apple", "  banana", "  cherry"],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"
        assert result["col1"].iloc[2] == "cherry"

    def test_trim_trailing_whitespace(self):
        """Test trimming trailing whitespace."""
        df = pd.DataFrame(
            {
                "col1": ["apple  ", "banana  ", "cherry  "],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"
        assert result["col1"].iloc[2] == "cherry"

    def test_trim_both_whitespace(self):
        """Test trimming both leading and trailing whitespace."""
        df = pd.DataFrame(
            {
                "col1": ["  apple  ", "  banana  ", "  cherry  "],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"
        assert result["col1"].iloc[2] == "cherry"

    def test_preserve_nan_values(self):
        """Test that NaN values are preserved."""
        df = pd.DataFrame(
            {
                "col1": ["  apple  ", np.nan, "  cherry  "],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert pd.isna(result["col1"].iloc[1])
        assert result["col1"].iloc[2] == "cherry"

    def test_preserve_numeric_columns(self):
        """Test that numeric columns are not affected."""
        df = pd.DataFrame(
            {
                "str_col": ["  apple  ", "  banana  "],
                "int_col": [1, 2],
                "float_col": [1.5, 2.5],
                "target": [0, 1],
            }
        )

        result = trim_whitespace(df)

        # String column should be trimmed
        assert result["str_col"].iloc[0] == "apple"

        # Numeric columns should be unchanged
        assert result["int_col"].iloc[0] == 1
        assert result["float_col"].iloc[0] == 1.5

    def test_no_string_columns(self):
        """Test behavior when there are no string columns."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        # Should return unchanged dataframe
        assert result.equals(df)

    def test_already_trimmed(self):
        """Test behavior when strings are already trimmed."""
        df = pd.DataFrame(
            {
                "col1": ["apple", "banana", "cherry"],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        # Should return unchanged dataframe
        assert result.equals(df)

    def test_mixed_trimmed_and_untrimmed(self):
        """Test with mix of trimmed and untrimmed values."""
        df = pd.DataFrame(
            {
                "col1": ["apple", "  banana  ", "cherry", "  date"],
                "target": [0, 1, 0, 1],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"
        assert result["col1"].iloc[2] == "cherry"
        assert result["col1"].iloc[3] == "date"

    def test_empty_strings(self):
        """Test behavior with empty strings."""
        df = pd.DataFrame(
            {
                "col1": ["  ", "", "  apple  "],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == ""  # Trimmed spaces
        assert result["col1"].iloc[1] == ""  # Already empty
        assert result["col1"].iloc[2] == "apple"

    def test_preserves_dataframe_copy(self):
        """Test that original dataframe is not modified."""
        df = pd.DataFrame(
            {
                "col1": ["  apple  ", "  banana  "],
                "target": [0, 1],
            }
        )

        original = df.copy()
        result = trim_whitespace(df)

        # Original should be unchanged
        assert df.equals(original)
        assert df["col1"].iloc[0] == "  apple  "

        # Result should have trimmed values
        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"

    def test_tabs_and_newlines(self):
        """Test that only leading/trailing whitespace is trimmed."""
        df = pd.DataFrame(
            {
                "col1": ["\tapple\n", "  banana\t", "\n\ncherry  "],
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == "banana"
        assert result["col1"].iloc[2] == "cherry"

    def test_preserves_internal_whitespace(self):
        """Test that internal whitespace is preserved."""
        df = pd.DataFrame(
            {
                "col1": ["  hello world  ", "  foo  bar  "],
                "target": [0, 1],
            }
        )

        result = trim_whitespace(df)

        assert result["col1"].iloc[0] == "hello world"  # Internal space preserved
        assert result["col1"].iloc[1] == "foo  bar"  # Internal spaces preserved

    def test_non_string_objects(self):
        """Test handling of non-string objects in object columns."""
        df = pd.DataFrame(
            {
                "col1": ["  apple  ", 123, "  banana  "],  # Mixed with non-string
                "target": [0, 1, 0],
            }
        )

        result = trim_whitespace(df)

        # String values should be trimmed, non-strings preserved
        assert result["col1"].iloc[0] == "apple"
        assert result["col1"].iloc[1] == 123  # Not a string, preserved
        assert result["col1"].iloc[2] == "banana"
