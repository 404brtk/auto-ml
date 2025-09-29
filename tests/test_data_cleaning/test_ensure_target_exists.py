"""Tests for helper functions in data cleaning module."""

import pandas as pd
import pytest
from auto_ml_pipeline.data_cleaning import _ensure_target_exists


class TestEnsureTargetExists:
    """Test _ensure_target_exists validation function."""

    def test_target_exists(self):
        """Test with valid target column."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

        # Should not raise error
        _ensure_target_exists(df, "target")

    def test_target_missing(self):
        """Test with missing target column."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        with pytest.raises(KeyError, match="Target column 'target' not found"):
            _ensure_target_exists(df, "target")

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(KeyError, match="Target column 'target' not found"):
            _ensure_target_exists(df, "target")

    def test_case_sensitive(self):
        """Test that column name is case-sensitive."""
        df = pd.DataFrame({"Target": [0, 1, 0], "feature": [1, 2, 3]})

        # Should fail - looking for "target" not "Target"
        with pytest.raises(KeyError, match="Target column 'target' not found"):
            _ensure_target_exists(df, "target")

    def test_whitespace_in_name(self):
        """Test with whitespace in column name."""
        df = pd.DataFrame(
            {"target ": [0, 1, 0], "feature": [1, 2, 3]}  # Note trailing space
        )

        # Exact match required
        with pytest.raises(KeyError, match="Target column 'target' not found"):
            _ensure_target_exists(df, "target")

    def test_numeric_column_name(self):
        """Test with numeric column name."""
        df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})

        # Should work with numeric column names
        _ensure_target_exists(df, 1)

    def test_error_message_includes_available_columns(self):
        """Test that error message lists available columns."""
        df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "feat3": [7, 8, 9]})

        with pytest.raises(KeyError) as exc_info:
            _ensure_target_exists(df, "target")

        error_msg = str(exc_info.value)
        # Should mention available columns
        assert "Available columns" in error_msg
        assert "feat1" in error_msg
        assert "feat2" in error_msg
        assert "feat3" in error_msg
