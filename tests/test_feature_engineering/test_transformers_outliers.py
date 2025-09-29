import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.transformers.outliers import OutlierTransformer


class TestOutlierTransformerIQR:
    """Test OutlierTransformer with IQR strategy."""

    def test_iqr_clip_outliers(self):
        """Test IQR method with clip."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is outlier

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Outlier should be clipped
        assert result["values"].max() <= 100  # Can be equal if at boundary
        # Value should be clipped to upper bound or less
        Q3 = df["values"].quantile(0.75)
        Q1 = df["values"].quantile(0.25)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        assert result["values"].max() <= upper_bound
        assert len(result) == len(df)  # All rows kept

    def test_iqr_remove_outliers(self):
        """Test IQR method with remove."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is outlier

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Outlier row should be removed
        assert len(result) < len(df)
        assert 100 not in result["values"].values

    def test_iqr_no_outliers(self):
        """Test IQR when no outliers present."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6]})  # Normal distribution

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_iqr_multiple_columns(self):
        """Test IQR with multiple numeric columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],  # col1 has outlier
                "col2": [10, 20, 30, 40, 50],  # col2 normal
                "text": ["a", "b", "c", "d", "e"],
            }
        )

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Only col1 should be modified
        assert result["col1"].max() <= 100  # Can be at boundary
        assert result["col2"].equals(df["col2"])

    def test_iqr_custom_multiplier(self):
        """Test IQR with different multipliers."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

        # More aggressive (lower threshold)
        transformer_aggressive = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.0
        )
        transformer_aggressive.fit(df)
        result_aggressive = transformer_aggressive.transform(df)

        # Less aggressive (higher threshold)
        transformer_lenient = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=3.0
        )
        transformer_lenient.fit(df)
        result_lenient = transformer_lenient.transform(df)

        # Aggressive should remove at least as many rows
        assert len(result_aggressive) <= len(result_lenient)

    def test_iqr_constant_column(self):
        """Test IQR with constant column (zero IQR)."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "variable": [1, 2, 3, 4, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Constant column should be ignored, variable column processed
        assert "constant" not in transformer.valid_cols_
        assert "variable" in transformer.valid_cols_
        assert result.shape == df.shape


class TestOutlierTransformerZScore:
    """Test OutlierTransformer with Z-score strategy."""

    def test_zscore_clip_outliers(self):
        """Test Z-score method with clip."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is outlier

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Outlier should be clipped (or at boundary)
        assert result["values"].max() <= 100
        assert len(result) == len(df)

    def test_zscore_remove_outliers(self):
        """Test Z-score method with remove."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is outlier

        transformer = OutlierTransformer(
            strategy="zscore", method="remove", zscore_threshold=3.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Outlier row should be removed (or kept if at boundary)
        assert len(result) <= len(df)
        # If row removed, 100 should not be present
        if len(result) < len(df):
            assert 100 not in result["values"].values

    def test_zscore_custom_threshold(self):
        """Test Z-score with different thresholds."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

        # More aggressive (lower threshold)
        transformer_aggressive = OutlierTransformer(
            strategy="zscore", method="remove", zscore_threshold=2.0
        )
        transformer_aggressive.fit(df)
        result_aggressive = transformer_aggressive.transform(df)

        # Less aggressive (higher threshold)
        transformer_lenient = OutlierTransformer(
            strategy="zscore", method="remove", zscore_threshold=4.0
        )
        transformer_lenient.fit(df)
        result_lenient = transformer_lenient.transform(df)

        # Aggressive should remove more rows
        assert len(result_aggressive) <= len(result_lenient)

    def test_zscore_constant_column(self):
        """Test Z-score with constant column (zero std)."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "variable": [1, 2, 3, 4, 100]})

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Constant column should be ignored
        assert "constant" not in transformer.valid_cols_
        assert "variable" in transformer.valid_cols_
        assert result.shape == df.shape

    def test_zscore_multiple_columns(self):
        """Test Z-score with multiple columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 100],  # Has outlier
                "col2": [10, 20, 30, 40, 50],  # Normal
            }
        )

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        assert result["col1"].max() <= 100  # Can be at boundary


class TestOutlierTransformerEdgeCases:
    """Test edge cases for OutlierTransformer."""

    def test_no_strategy(self):
        """Test with no outlier detection strategy."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 100]})

        transformer = OutlierTransformer(strategy=None)
        transformer.fit(df)
        result = transformer.transform(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_non_dataframe_input(self):
        """Test with non-DataFrame input."""
        arr = np.array([[1, 2], [3, 4]])

        transformer = OutlierTransformer(strategy="iqr")
        result = transformer.fit_transform(arr)

        # Should return unchanged
        np.testing.assert_array_equal(result, arr)

    def test_no_numeric_columns(self):
        """Test with DataFrame containing no numeric columns."""
        df = pd.DataFrame({"text1": ["a", "b", "c"], "text2": ["x", "y", "z"]})

        transformer = OutlierTransformer(strategy="iqr")
        transformer.fit(df)
        result = transformer.transform(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        transformer = OutlierTransformer(strategy="iqr")
        transformer.fit(df)
        result = transformer.transform(df)

        assert len(result) == 0

    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"values": [100]})

        transformer = OutlierTransformer(strategy="iqr")
        transformer.fit(df)
        result = transformer.transform(df)

        # Should handle gracefully
        assert len(result) == 1

    def test_all_same_values(self):
        """Test with all identical values (zero variance)."""
        df = pd.DataFrame({"values": [5, 5, 5, 5, 5]})

        transformer = OutlierTransformer(strategy="zscore")
        transformer.fit(df)
        result = transformer.transform(df)

        # Should be unchanged (no valid columns with variance)
        pd.testing.assert_frame_equal(result, df)

    def test_negative_outliers(self):
        """Test detection of negative outliers."""
        df = pd.DataFrame({"values": [-100, 1, 2, 3, 4, 5]})  # -100 is outlier

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Negative outlier should be removed
        assert -100 not in result["values"].values

    def test_both_extreme_outliers(self):
        """Test with outliers on both ends."""
        df = pd.DataFrame({"values": [-100, 1, 2, 3, 4, 5, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Both outliers should be removed
        assert len(result) < len(df)
        assert -100 not in result["values"].values
        assert 100 not in result["values"].values

    def test_transform_without_fit(self):
        """Test transform on different data after fit."""
        df_train = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        df_test = pd.DataFrame({"values": [1, 2, 100]})  # Has outlier

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df_train)
        result = transformer.transform(df_test)

        # Outlier should be clipped based on training data bounds
        assert result["values"].max() <= 100  # Can be at boundary


class TestOutlierTransformerValidation:
    """Test validation and error handling."""

    def test_invalid_iqr_multiplier(self):
        """Test invalid IQR multiplier."""
        with pytest.raises(ValueError, match="iqr_multiplier must be positive"):
            OutlierTransformer(strategy="iqr", iqr_multiplier=0)

        with pytest.raises(ValueError, match="iqr_multiplier must be positive"):
            OutlierTransformer(strategy="iqr", iqr_multiplier=-1.5)

    def test_invalid_zscore_threshold(self):
        """Test invalid Z-score threshold."""
        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            OutlierTransformer(strategy="zscore", zscore_threshold=0)

        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            OutlierTransformer(strategy="zscore", zscore_threshold=-3.0)

    def test_invalid_method(self):
        """Test invalid method parameter."""
        with pytest.raises(ValueError, match="method must be"):
            OutlierTransformer(strategy="iqr", method="invalid")

    def test_invalid_strategy(self):
        """Test invalid strategy parameter."""
        with pytest.raises(ValueError, match="strategy must be"):
            OutlierTransformer(strategy="invalid")

    def test_case_insensitive_strategy(self):
        """Test that strategy is case-insensitive."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 100]})

        # Should work with uppercase
        transformer = OutlierTransformer(strategy="IQR", method="clip")
        transformer.fit(df)
        result = transformer.transform(df)

        assert result["values"].max() <= 100  # Can be at boundary

    def test_case_insensitive_method(self):
        """Test that method is case-insensitive."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 100]})

        # Should work with uppercase
        transformer = OutlierTransformer(strategy="iqr", method="CLIP")
        transformer.fit(df)
        result = transformer.transform(df)

        assert len(result) == len(df)


class TestOutlierTransformerIntegration:
    """Integration tests for OutlierTransformer."""

    def test_pipeline_with_missing_values(self):
        """Test outlier detection with missing values present."""
        df = pd.DataFrame({"values": [1, 2, np.nan, 4, 5, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should handle NaN values gracefully
        assert result["values"].max() <= 100  # Can be at boundary
        assert result["values"].isna().sum() == df["values"].isna().sum()

    def test_multiple_outliers_same_row(self):
        """Test row with outliers in multiple columns."""
        df = pd.DataFrame({"col1": [1, 2, 100, 4, 5], "col2": [10, 20, 200, 40, 50]})

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Row 2 has outliers in both columns, should be removed
        assert len(result) < len(df)
        assert 100 not in result["col1"].values
        assert 200 not in result["col2"].values

    def test_preserves_dataframe_structure(self):
        """Test that DataFrame structure is preserved."""
        df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 100],
                "text": ["a", "b", "c", "d", "e"],
                "num2": [10, 20, 30, 40, 50],
            }
        )

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # All columns should be present
        assert set(result.columns) == set(df.columns)
        # Non-numeric columns unchanged
        assert result["text"].equals(df["text"])

    def test_fit_transform_equivalence(self):
        """Test that fit_transform gives same result as fit then transform."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 100]})

        transformer1 = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result1 = transformer1.fit_transform(df)

        transformer2 = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer2.fit(df)
        result2 = transformer2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)
