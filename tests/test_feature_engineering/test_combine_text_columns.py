import pandas as pd
import numpy as np

from auto_ml_pipeline.feature_engineering import combine_text_columns


def test_combine_single_column():
    """Test combining when only one text column."""
    df = pd.DataFrame({"text": ["hello", "world", "test"]})
    result = combine_text_columns(df)
    expected = np.array(["hello", "world", "test"])
    np.testing.assert_array_equal(result, expected)


def test_combine_multiple_columns():
    """Test combining multiple text columns."""
    df = pd.DataFrame({"title": ["Hello", "World"], "desc": ["world", "peace"]})
    result = combine_text_columns(df)
    expected = np.array(["Hello world", "World peace"])
    np.testing.assert_array_equal(result, expected)


def test_combine_with_missing_values():
    """Test combining with NaN values."""
    df = pd.DataFrame(
        {"col1": ["Hello", np.nan, "Test"], "col2": ["world", "peace", np.nan]}
    )
    result = combine_text_columns(df)
    expected = np.array(["Hello world", "nan peace", "Test nan"])
    np.testing.assert_array_equal(result, expected)


def test_combine_empty_dataframe():
    """Test combining empty DataFrame."""
    df = pd.DataFrame({"text": []})
    result = combine_text_columns(df)
    assert len(result) == 0
