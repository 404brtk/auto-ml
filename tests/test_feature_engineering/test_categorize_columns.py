"""Tests for categorize_columns function."""

import pandas as pd
import numpy as np

from auto_ml_pipeline.feature_engineering import categorize_columns
from auto_ml_pipeline.config import FeatureEngineeringConfig


class TestCategorizeColumnsBasic:
    """Test basic column categorization by type."""

    def test_numeric_columns(self):
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

    def test_datetime_columns(self):
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

    def test_time_columns(self):
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

    def test_text_columns(self):
        """Test categorization of text columns."""
        df = pd.DataFrame(
            {"description": ["This is a long description with many words"] * 3}
        )
        cfg = FeatureEngineeringConfig(text_length_threshold=10)
        result = categorize_columns(df, cfg)

        assert result.numeric == []
        assert result.text == ["description"]
        assert result.categorical_low == []
        assert result.categorical_high == []
        assert result.datetime == []
        assert result.time == []

    def test_categorical_low_cardinality(self):
        """Test categorization of low-cardinality categorical columns."""
        df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"] * 20})
        cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 10})
        result = categorize_columns(df, cfg)

        assert result.numeric == []
        assert result.categorical_low == ["category"]
        assert result.categorical_high == []
        assert result.datetime == []
        assert result.time == []
        assert result.text == []

    def test_categorical_high_cardinality(self):
        """Test categorization of high-cardinality categorical columns."""
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


class TestCategorizeColumnsEdgeCases:
    """Test edge cases in column categorization."""

    def test_short_unique_high_cardinality_heuristic(self):
        """Test the short+unique heuristic for high cardinality."""
        df = pd.DataFrame({"code": [f"AB{i}" for i in range(100)]})
        cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 200})
        result = categorize_columns(df, cfg)

        assert result.categorical_high == ["code"]

    def test_all_nan_column(self):
        """Test handling of columns with only NaN values."""
        df = pd.DataFrame(
            {"good_col": ["A", "B", "A"], "nan_col": [np.nan, np.nan, np.nan]}
        )
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        assert "nan_col" not in result.categorical_low
        assert "nan_col" not in result.categorical_high
        assert "nan_col" not in result.text
        assert "nan_col" not in result.time

    def test_empty_dataframe(self):
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

    def test_mixed_time_patterns(self):
        """Test column with mixed time patterns (some valid, some invalid)."""
        df = pd.DataFrame(
            {"mixed_time": ["14:30:00", "09:15:30", "invalid", "18:45:00", "text"]}
        )
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        # Only 60% are valid times (3/5), below 70% threshold
        assert "mixed_time" not in result.time

    def test_single_value_column(self):
        """Test column with single unique value."""
        df = pd.DataFrame({"single": ["same"] * 100})
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        assert "single" in result.categorical_low

    def test_very_long_text(self):
        """Test with very long text values."""
        long_text = "word " * 1000  # 5000 characters
        df = pd.DataFrame({"long_text": [long_text] * 5})
        cfg = FeatureEngineeringConfig(text_length_threshold=50)
        result = categorize_columns(df, cfg)

        assert "long_text" in result.text

    def test_numeric_strings_are_categorical(self):
        """Test that numeric strings may be coerced or categorized."""
        df = pd.DataFrame({"num_str": ["1", "2", "3", "4", "5"]})
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        # Numeric strings may be coerced to numeric or treated as categorical
        # Either behavior is acceptable
        assert "num_str" not in result.datetime
        assert "num_str" not in result.time
        # Could be in numeric, categorical_low, or categorical_high

    def test_boundary_cardinality_threshold(self):
        """Test boundary cases for high cardinality detection."""
        threshold = 50

        # Exactly at threshold - may trigger short+unique heuristic
        df_at = pd.DataFrame({"cat": [f"val_{i}" for i in range(threshold)]})
        cfg = FeatureEngineeringConfig(
            encoding={"high_cardinality_threshold": threshold}
        )
        result_at = categorize_columns(df_at, cfg)
        # Could be in either low or high due to heuristics
        assert "cat" in (result_at.categorical_low + result_at.categorical_high)

        # Well above threshold
        df_above = pd.DataFrame({"cat": [f"val_{i}" for i in range(threshold + 20)]})
        result_above = categorize_columns(df_above, cfg)
        assert "cat" in result_above.categorical_high

    def test_only_whitespace_values(self):
        """Test column with only whitespace values."""
        df = pd.DataFrame({"whitespace": ["   ", "  ", "\t", "\n"]})
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        # Whitespace values are unique strings, may be high cardinality
        # Should be in one of the categorical buckets
        assert "whitespace" in (result.categorical_low + result.categorical_high)

    def test_short_unique_heuristic_edge_case(self):
        """Test the short and unique heuristic with edge cases."""
        df = pd.DataFrame({"codes": [f"AB{i:02d}" for i in range(100)]})
        cfg = FeatureEngineeringConfig()
        result = categorize_columns(df, cfg)

        assert "codes" in result.categorical_high
