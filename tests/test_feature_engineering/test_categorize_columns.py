import pandas as pd
import numpy as np

from auto_ml_pipeline.feature_engineering import categorize_columns
from auto_ml_pipeline.config import FeatureEngineeringConfig


def test_categorize_numeric_columns():
    """Test categorization of numeric columns."""
    df = pd.DataFrame({"num1": [1, 2, 3], "num2": [1.1, 2.2, 3.3]})
    cfg = FeatureEngineeringConfig()
    result = categorize_columns(df, cfg)

    assert result.numeric == ["num1", "num2"]
    assert result.categorical_low == []
    assert result.categorical_high == []
    assert result.datetime == []
    assert result.time == []
    assert result.text == []


def test_categorize_datetime_columns():
    """Test categorization of datetime columns."""
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3)})
    cfg = FeatureEngineeringConfig()
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.datetime == ["date"]
    assert result.categorical_low == []
    assert result.categorical_high == []
    assert result.time == []
    assert result.text == []


def test_categorize_time_columns():
    """Test categorization of time columns."""
    df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "18:45:00"]})
    cfg = FeatureEngineeringConfig()
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.time == ["time"]
    assert result.categorical_low == []
    assert result.categorical_high == []
    assert result.datetime == []
    assert result.text == []


def test_categorize_text_columns():
    """Test categorization of text columns."""
    df = pd.DataFrame(
        {"description": ["This is a long description with many words"] * 3}
    )
    cfg = FeatureEngineeringConfig(
        text_length_threshold=10
    )  # Low threshold to trigger text
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.text == ["description"]
    assert result.categorical_low == []
    assert result.categorical_high == []
    assert result.datetime == []
    assert result.time == []


def test_categorize_categorical_low():
    """Test categorization of low-cardinality categorical columns."""
    df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"]})
    cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 10})
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.categorical_low == ["category"]
    assert result.categorical_high == []
    assert result.datetime == []
    assert result.time == []
    assert result.text == []


def test_categorize_categorical_high():
    """Test categorization of high-cardinality categorical columns."""
    # Create many unique values to trigger high cardinality
    unique_vals = [f"val_{i}" for i in range(100)]
    df = pd.DataFrame({"category": unique_vals})
    cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 50})
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.categorical_low == []
    assert result.categorical_high == ["category"]
    assert result.datetime == []
    assert result.time == []
    assert result.text == []


def test_categorize_short_unique_high_cardinality():
    """Test the short+unique heuristic for high cardinality."""
    # Short strings with high uniqueness ratio
    df = pd.DataFrame({"code": [f"AB{i}" for i in range(100)]})
    cfg = FeatureEngineeringConfig(
        encoding={"high_cardinality_threshold": 200}
    )  # High threshold
    result = categorize_columns(df, cfg)

    # Should be high cardinality due to short+unique heuristic
    assert result.categorical_high == ["code"]


def test_categorize_all_nan_column():
    """Test handling of columns with only NaN values."""
    df = pd.DataFrame(
        {"good_col": ["A", "B", "A"], "nan_col": [np.nan, np.nan, np.nan]}
    )
    cfg = FeatureEngineeringConfig()
    result = categorize_columns(df, cfg)

    # nan_col should be ignored
    assert "nan_col" not in result.categorical_low
    assert "nan_col" not in result.categorical_high
    assert "nan_col" not in result.text
    assert "nan_col" not in result.time


def test_categorize_empty_dataframe():
    """Test categorization of empty DataFrame."""
    df = pd.DataFrame()
    cfg = FeatureEngineeringConfig()
    result = categorize_columns(df, cfg)

    assert result.numeric == []
    assert result.categorical_low == []
    assert result.categorical_high == []
    assert result.datetime == []
    assert result.time == []
    assert result.text == []
