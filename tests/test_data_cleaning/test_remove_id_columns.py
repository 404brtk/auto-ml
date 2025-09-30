"""Tests for remove_id_columns function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import remove_id_columns


class TestRemoveIdColumns:
    """Test remove_id_columns function."""

    def test_remove_perfect_id_column(self):
        """Test removing column with 100% unique values."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],  # 100% unique
                "feature": [10, 20, 10, 20, 10],  # Not unique (only 2 values)
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        assert "id" not in result.columns
        assert "feature" in result.columns
        assert "target" in result.columns

    def test_remove_high_cardinality_string(self):
        """Test removing string column with high uniqueness."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],  # 100% unique
                "feature": [10, 20, 10, 20, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        assert "customer_id" not in result.columns
        assert "feature" in result.columns
        assert "target" in result.columns

    def test_keep_low_cardinality_column(self):
        """Test keeping column with low uniqueness."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],  # Only 2 unique (40%)
                "feature": [10, 20, 10, 20, 10],  # Only 2 unique (40%)
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # All columns should remain (low uniqueness)
        assert "category" in result.columns
        assert "feature" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 3

    def test_threshold_96_percent(self):
        """Test with 96% unique values (above threshold)."""
        df = pd.DataFrame(
            {
                "semi_id": list(range(100)),  # 100 unique out of 100 = 100%
                "target": [0, 1] * 50,
            }
        )
        # Make one duplicate to get 99% uniqueness
        df.loc[99, "semi_id"] = df.loc[0, "semi_id"]

        result = remove_id_columns(df, "target", threshold=0.95)

        # 99% > 95%, should be removed
        assert "semi_id" not in result.columns

    def test_threshold_94_percent(self):
        """Test with 94% unique values (below threshold)."""
        df = pd.DataFrame(
            {
                "semi_id": list(range(94)) + [0] * 6,  # 94 unique out of 100 = 94%
                "target": [0, 1] * 50,
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # 94% < 95%, should be kept
        assert "semi_id" in result.columns

    def test_custom_threshold(self):
        """Test with custom threshold."""
        df = pd.DataFrame(
            {
                "col1": list(range(80)) + [0] * 20,  # 80% unique
                "col2": list(range(70)) + [0] * 30,  # 70% unique
                "target": [0, 1] * 50,
            }
        )

        result = remove_id_columns(df, "target", threshold=0.75)

        # 80% > 75%, col1 removed
        # 70% < 75%, col2 kept
        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_never_remove_target(self):
        """Test that target column is never removed even if unique."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],  # 100% unique
                "target": [0, 1, 2, 3, 4],  # Also 100% unique
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # ID should be removed, target should not
        assert "id" not in result.columns
        assert "target" in result.columns

    def test_multiple_id_columns(self):
        """Test removing multiple ID columns."""
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "transaction_id": ["T001", "T002", "T003", "T004", "T005"],
                "feature": [10, 20, 10, 20, 10],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # Both ID columns should be removed
        assert "customer_id" not in result.columns
        assert "transaction_id" not in result.columns
        assert "feature" in result.columns
        assert "target" in result.columns

    def test_numeric_id_columns_removed(self):
        """Test that numeric ID columns are also detected and removed."""
        df = pd.DataFrame(
            {
                "numeric_id": list(range(100)),  # 100% unique numeric
                "target": [0, 1] * 50,
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # Numeric ID columns should be removed (100% > 95%)
        assert "numeric_id" not in result.columns
        assert "target" in result.columns

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({"target": []})

        result = remove_id_columns(df, "target", threshold=0.95)

        assert "target" in result.columns
        assert len(result) == 0

    def test_preserves_dataframe_copy(self):
        """Test that original dataframe is not modified."""
        df = pd.DataFrame(
            {
                "id": ["ID1", "ID2", "ID3"],
                "feature": [1, 2, 1],
                "target": [0, 1, 0],
            }
        )

        original_cols = df.columns.tolist()
        result = remove_id_columns(df, "target", threshold=0.95)

        # Original should be unchanged
        assert df.columns.tolist() == original_cols
        assert "id" in df.columns

        # Result should have ID column removed but feature kept
        assert "id" not in result.columns
        assert "feature" in result.columns

    def test_with_nan_values(self):
        """Test ID detection with NaN values."""
        df = pd.DataFrame(
            {
                "id": ["ID1", "ID2", "ID3", np.nan, "ID5"],  # 4 unique + 1 NaN = 80%
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.75)

        # 80% unique > 75%, should be removed
        assert "id" not in result.columns

    def test_categorical_dtype(self):
        """Test with categorical dtype columns."""
        df = pd.DataFrame(
            {
                "cat_id": pd.Categorical(["C1", "C2", "C3", "C4", "C5"]),
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_id_columns(df, "target", threshold=0.95)

        # Categorical with 100% unique should be removed
        assert "cat_id" not in result.columns
