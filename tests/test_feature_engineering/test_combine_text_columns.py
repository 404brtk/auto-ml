"""Tests for combine_text_columns function."""

import pandas as pd
import numpy as np

from auto_ml_pipeline.feature_engineering import combine_text_columns


class TestCombineTextColumns:
    """Test combine_text_columns helper function."""

    def test_combine_single_column(self):
        """Test combining when only one text column."""
        df = pd.DataFrame({"text": ["hello", "world", "test"]})
        result = combine_text_columns(df)
        expected = np.array(["hello", "world", "test"])
        np.testing.assert_array_equal(result, expected)

    def test_combine_multiple_columns(self):
        """Test combining multiple text columns."""
        df = pd.DataFrame({"title": ["Hello", "World"], "desc": ["world", "peace"]})
        result = combine_text_columns(df)
        expected = np.array(["Hello world", "World peace"])
        np.testing.assert_array_equal(result, expected)

    def test_combine_with_missing_values(self):
        """Test combining with NaN values."""
        df = pd.DataFrame(
            {"col1": ["Hello", np.nan, "Test"], "col2": ["world", "peace", np.nan]}
        )
        result = combine_text_columns(df)
        expected = np.array(["Hello world", "nan peace", "Test nan"])
        np.testing.assert_array_equal(result, expected)

    def test_combine_empty_dataframe(self):
        """Test combining empty DataFrame."""
        df = pd.DataFrame({"text": []})
        result = combine_text_columns(df)
        assert len(result) == 0

    def test_combine_single_row(self):
        """Test combining with single row."""
        df = pd.DataFrame({"col1": ["hello"], "col2": ["world"]})
        result = combine_text_columns(df)
        expected = np.array(["hello world"])
        np.testing.assert_array_equal(result, expected)

    def test_combine_with_numbers(self):
        """Test combining columns with numeric values."""
        df = pd.DataFrame({"col1": ["text", "more"], "col2": [123, 456]})
        result = combine_text_columns(df)
        # Should convert numbers to strings
        assert len(result) == 2
        assert "123" in result[0]
        assert "456" in result[1]
