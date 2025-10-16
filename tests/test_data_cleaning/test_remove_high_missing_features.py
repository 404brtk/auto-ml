"""Tests for remove_high_missing_features function."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def df_with_missing():
    """Fixture for DataFrame with various missing ratios."""
    return pd.DataFrame(
        {
            "no_missing": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "low_missing": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # 10% missing
            "medium_missing": [
                1,
                2,
                np.nan,
                np.nan,
                np.nan,
                6,
                7,
                8,
                9,
                10,
            ],  # 30% missing
            "high_missing": [
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                10,
            ],  # 80% missing
            "all_missing": [np.nan] * 10,  # 100% missing
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


class TestRemoveHighMissingFeatures:
    """Test remove_high_missing_features function."""

    def test_removes_features_above_threshold(self, df_with_missing):
        """Test that features above threshold are removed."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        result = remove_high_missing_features(
            df_with_missing, "target", max_missing_ratio=0.5
        )

        # Features with >50% missing should be removed
        assert "high_missing" not in result.columns
        assert "all_missing" not in result.columns

        # Features with <=50% missing should be kept
        assert "no_missing" in result.columns
        assert "low_missing" in result.columns
        assert "medium_missing" in result.columns
        assert "target" in result.columns

    def test_keeps_features_below_threshold(self, df_with_missing):
        """Test that features below threshold are kept."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        result = remove_high_missing_features(
            df_with_missing, "target", max_missing_ratio=0.9
        )

        # Only features with >90% missing should be removed
        assert "all_missing" not in result.columns

        # All others should be kept
        assert "no_missing" in result.columns
        assert "low_missing" in result.columns
        assert "medium_missing" in result.columns
        assert "high_missing" in result.columns
        assert "target" in result.columns

    def test_preserves_target_column(self):
        """Test that target column is never removed."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "target": [np.nan, np.nan, np.nan, np.nan, 1],  # 80% missing
            }
        )

        result = remove_high_missing_features(df, "target", max_missing_ratio=0.5)

        # Target should still be present
        assert "target" in result.columns

    def test_no_features_removed_with_high_threshold(self, df_with_missing):
        """Test that no features are removed with threshold of 1.0."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        result = remove_high_missing_features(
            df_with_missing, "target", max_missing_ratio=1.0
        )

        # All columns should be kept
        assert set(result.columns) == set(df_with_missing.columns)

    def test_removes_features_with_any_missing(self):
        """Test behavior with threshold of 0.0."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        df = pd.DataFrame(
            {
                "feat_with_missing": [1, np.nan, 3, 4, 5],
                "feat_no_missing": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_high_missing_features(df, "target", max_missing_ratio=0.0)

        # feat_with_missing has >0% missing, should be removed
        assert "feat_with_missing" not in result.columns
        assert "feat_no_missing" in result.columns
        assert "target" in result.columns

    def test_invalid_threshold_raises_error(self, df_with_missing):
        """Test that invalid threshold raises ValueError."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        with pytest.raises(ValueError, match="max_missing_ratio must be in"):
            remove_high_missing_features(
                df_with_missing, "target", max_missing_ratio=1.5
            )

        with pytest.raises(ValueError, match="max_missing_ratio must be in"):
            remove_high_missing_features(
                df_with_missing, "target", max_missing_ratio=-0.1
            )

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        df = pd.DataFrame({"target": []})
        result = remove_high_missing_features(df, "target", max_missing_ratio=0.5)

        assert "target" in result.columns
        assert len(result) == 0

    def test_returns_copy_not_view(self, df_with_missing):
        """Test that function returns a copy, not modifying original."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        original_cols = df_with_missing.columns.tolist()
        result = remove_high_missing_features(
            df_with_missing, "target", max_missing_ratio=0.5
        )

        # Original should not be modified
        assert df_with_missing.columns.tolist() == original_cols
        # Result should have fewer columns
        assert len(result.columns) < len(df_with_missing.columns)

    def test_multiple_features_same_ratio(self):
        """Test with multiple features having same missing ratio."""
        from auto_ml_pipeline.data_cleaning import remove_high_missing_features

        df = pd.DataFrame(
            {
                "feat1": [1, np.nan, np.nan, 4, 5],  # 40% missing
                "feat2": [1, np.nan, np.nan, 4, 5],  # 40% missing
                "feat3": [1, 2, 3, 4, 5],  # 0% missing
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_high_missing_features(df, "target", max_missing_ratio=0.3)

        # Both feat1 and feat2 should be removed
        assert "feat1" not in result.columns
        assert "feat2" not in result.columns
        assert "feat3" in result.columns
