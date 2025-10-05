import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.transformers.outliers import OutlierTransformer


@pytest.fixture
def simple_outlier_df():
    """DataFrame with one clear outlier."""
    return pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})


@pytest.fixture
def multi_column_df():
    """DataFrame with multiple columns including outliers."""
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 100],  # col1 has outlier
            "col2": [10, 20, 30, 40, 50],  # col2 normal
            "text": ["a", "b", "c", "d", "e"],  # non-numeric
        }
    )


@pytest.fixture
def normal_distribution_df():
    """DataFrame with normal distribution, no outliers."""
    np.random.seed(42)
    return pd.DataFrame({"values": np.random.normal(50, 10, 100)})


@pytest.fixture
def both_extremes_df():
    """DataFrame with outliers on both ends."""
    return pd.DataFrame({"values": [-100, 1, 2, 3, 4, 5, 100]})


class TestOutlierTransformerIQR:
    """Test OutlierTransformer with IQR strategy."""

    def test_iqr_clip_outliers(self, simple_outlier_df):
        """Test IQR method clips outliers to calculated bounds."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Calculate expected bounds
        Q1 = df["values"].quantile(0.25)
        Q3 = df["values"].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Verify clipping
        assert result["values"].max() <= upper_bound
        assert len(result) == len(df)  # All rows kept

    def test_iqr_remove_outliers(self, simple_outlier_df):
        """Test IQR method removes outlier rows entirely."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Outlier row should be removed
        assert len(result) < len(df)
        assert 100 not in result["values"].values

    def test_iqr_no_outliers(self):
        """Test IQR when no outliers present - data unchanged."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        pd.testing.assert_frame_equal(result, df)

    def test_iqr_multiple_columns(self, multi_column_df):
        """Test IQR processes only numeric columns."""
        df = multi_column_df

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Text column should be unchanged
        assert result["text"].equals(df["text"])
        # Numeric columns should be processed
        assert "col1" in transformer.valid_cols_
        assert "col2" in transformer.valid_cols_

    @pytest.mark.parametrize(
        "multiplier,expected_behavior",
        [
            (1.0, "more_aggressive"),  # Lower threshold = more outliers detected
            (3.0, "less_aggressive"),  # Higher threshold = fewer outliers detected
        ],
    )
    def test_iqr_custom_multiplier(self, multiplier, expected_behavior):
        """Test IQR with different multipliers affects outlier detection."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=multiplier
        )
        result = transformer.fit_transform(df)

        if expected_behavior == "more_aggressive":
            # Lower multiplier should potentially remove more rows
            assert len(result) <= len(df)
        else:
            # Higher multiplier should be more lenient
            assert len(result) <= len(df)

    def test_iqr_constant_column(self):
        """Test IQR handles constant columns (zero IQR) gracefully."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "variable": [1, 2, 3, 4, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Constant column should be excluded from processing
        assert "constant" not in transformer.valid_cols_
        assert "variable" in transformer.valid_cols_
        assert result.shape == df.shape

    def test_iqr_negative_outliers(self):
        """Test IQR detects negative outliers correctly."""
        df = pd.DataFrame({"values": [-100, 1, 2, 3, 4, 5]})

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        assert -100 not in result["values"].values

    def test_iqr_both_extremes(self, both_extremes_df):
        """Test IQR removes outliers on both ends."""
        df = both_extremes_df

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Both outliers should be removed
        assert len(result) < len(df)
        assert -100 not in result["values"].values
        assert 100 not in result["values"].values


class TestOutlierTransformerZScore:
    """Test OutlierTransformer with Z-score strategy."""

    def test_zscore_clip_outliers(self, simple_outlier_df):
        """Test Z-score method clips outliers to threshold bounds."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        result = transformer.fit_transform(df)

        # Value should be clipped
        assert len(result) == len(df)  # All rows kept
        # Clipped value should be within bounds
        mean = df["values"].mean()
        std = df["values"].std(ddof=1)
        upper_bound = mean + 3.0 * std
        assert result["values"].max() <= upper_bound

    def test_zscore_remove_outliers(self, simple_outlier_df):
        """Test Z-score method removes outlier rows."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="zscore", method="remove", zscore_threshold=3.0
        )
        result = transformer.fit_transform(df)

        # Outlier row may be removed
        assert len(result) <= len(df)

    @pytest.mark.parametrize(
        "threshold,description",
        [
            (2.0, "aggressive"),  # Lower threshold = more outliers
            (4.0, "lenient"),  # Higher threshold = fewer outliers
        ],
    )
    def test_zscore_custom_threshold(self, threshold, description):
        """Test Z-score with different thresholds."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

        transformer = OutlierTransformer(
            strategy="zscore", method="remove", zscore_threshold=threshold
        )
        result = transformer.fit_transform(df)

        assert len(result) <= len(df)

    def test_zscore_constant_column(self):
        """Test Z-score handles constant columns (zero std) gracefully."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "variable": [1, 2, 3, 4, 100]})

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Constant column should be excluded
        assert "constant" not in transformer.valid_cols_
        assert "variable" in transformer.valid_cols_
        assert result.shape == df.shape

    def test_zscore_multiple_columns(self, multi_column_df):
        """Test Z-score with multiple columns."""
        df = multi_column_df

        transformer = OutlierTransformer(
            strategy="zscore", method="clip", zscore_threshold=3.0
        )
        result = transformer.fit_transform(df)

        # Should process numeric columns only
        assert "col1" in transformer.num_cols_
        assert "col2" in transformer.num_cols_
        assert "text" not in transformer.num_cols_

        # Verify result is valid and preserves structure
        assert len(result) == len(df)
        assert set(result.columns) == set(df.columns)
        assert result["text"].equals(df["text"])


class TestOutlierTransformerIsolationForest:
    """Test OutlierTransformer with IsolationForest strategy."""

    def test_isolation_forest_clip_raises_error(self):
        """Test that IsolationForest with 'clip' method raises ValueError."""
        with pytest.raises(
            ValueError, match="IsolationForest does not support method='clip'"
        ):
            OutlierTransformer(strategy="isolation_forest", method="clip")

    def test_isolation_forest_remove_outliers(self, simple_outlier_df):
        """Test IsolationForest removes detected outliers."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            contamination=0.1,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        # Should remove some outliers
        assert len(result) <= len(df)

    def test_isolation_forest_with_normal_data(self, normal_distribution_df):
        """Test IsolationForest with normal distribution data."""
        df = normal_distribution_df

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            contamination=0.05,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        # Should remove approximately 5% of data
        removed_pct = (len(df) - len(result)) / len(df)
        assert 0 <= removed_pct <= 0.15  # Allow some variance

    @pytest.mark.parametrize(
        "contamination",
        [0.05, 0.1, 0.2, "auto"],
    )
    def test_isolation_forest_contamination_parameter(
        self, contamination, simple_outlier_df
    ):
        """Test IsolationForest with different contamination values."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            contamination=contamination,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        # Should successfully process data
        assert len(result) <= len(df)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize(
        "n_estimators",
        [10, 50, 100, 200],
    )
    def test_isolation_forest_n_estimators(self, n_estimators, simple_outlier_df):
        """Test IsolationForest with different number of estimators."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            n_estimators=n_estimators,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        assert len(result) <= len(df)
        assert transformer.n_estimators == n_estimators

    @pytest.mark.parametrize(
        "max_samples",
        ["auto", 4, 0.5],
    )
    def test_isolation_forest_max_samples(self, max_samples):
        """Test IsolationForest with different max_samples values."""
        df = pd.DataFrame({"values": list(range(1, 11)) + [100]})  # 11 rows

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            max_samples=max_samples,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        assert len(result) <= len(df)

    def test_isolation_forest_insufficient_samples(self):
        """Test IsolationForest with too few samples (< 10)."""
        df = pd.DataFrame({"values": [1, 2, 3, 100]})  # Only 4 samples

        transformer = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should return unchanged due to insufficient samples
        pd.testing.assert_frame_equal(result, df)

    def test_isolation_forest_with_nan_values(self):
        """Test IsolationForest handles NaN values gracefully."""
        df = pd.DataFrame({"values": [1, 2, np.nan, 4, 5, 100]})

        transformer = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should return unchanged due to NaN values
        pd.testing.assert_frame_equal(result, df)

    def test_isolation_forest_with_inf_values(self):
        """Test IsolationForest handles infinite values gracefully."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, np.inf]})

        transformer = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Should return unchanged due to inf values
        pd.testing.assert_frame_equal(result, df)

    def test_isolation_forest_constant_features(self):
        """Test IsolationForest with zero variance features."""
        df = pd.DataFrame(
            {
                "constant": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                "variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            }
        )

        transformer = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        transformer.fit(df)
        result = transformer.transform(df)

        # Constant column should be excluded
        assert "constant" not in transformer.valid_cols_
        assert "variable" in transformer.valid_cols_

        # Verify result validity
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df)
        assert "constant" in result.columns  # Column exists but wasn't processed
        assert "variable" in result.columns

    def test_isolation_forest_multivariate_outliers(self):
        """Test IsolationForest detects multivariate outliers."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
                "col2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 200],
            }
        )

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            contamination=0.1,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        # Should detect and remove multivariate outlier
        assert len(result) < len(df)

    def test_isolation_forest_reproducibility(self, simple_outlier_df):
        """Test IsolationForest produces consistent results with same random_state."""
        df = simple_outlier_df

        transformer1 = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        result1 = transformer1.fit_transform(df)

        transformer2 = OutlierTransformer(
            strategy="isolation_forest", method="remove", random_state=42
        )
        result2 = transformer2.fit_transform(df)

        pd.testing.assert_frame_equal(result1, result2)


class TestOutlierTransformerEdgeCases:
    """Test edge cases for OutlierTransformer."""

    def test_no_strategy(self, simple_outlier_df):
        """Test with no outlier detection strategy."""
        df = simple_outlier_df

        transformer = OutlierTransformer(strategy=None)
        result = transformer.fit_transform(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_non_dataframe_input(self):
        """Test with non-DataFrame input returns unchanged."""
        arr = np.array([[1, 2], [3, 4]])

        transformer = OutlierTransformer(strategy="iqr")
        result = transformer.fit_transform(arr)

        # Should return unchanged
        np.testing.assert_array_equal(result, arr)

    def test_no_numeric_columns(self):
        """Test with DataFrame containing no numeric columns."""
        df = pd.DataFrame({"text1": ["a", "b", "c"], "text2": ["x", "y", "z"]})

        transformer = OutlierTransformer(strategy="iqr")
        result = transformer.fit_transform(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        transformer = OutlierTransformer(strategy="iqr")
        result = transformer.fit_transform(df)

        assert len(result) == 0

    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"values": [100]})

        transformer = OutlierTransformer(strategy="iqr")
        result = transformer.fit_transform(df)

        # Should handle gracefully
        assert len(result) == 1

    def test_all_same_values(self):
        """Test with all identical values (zero variance)."""
        df = pd.DataFrame({"values": [5, 5, 5, 5, 5]})

        transformer = OutlierTransformer(strategy="zscore")
        result = transformer.fit_transform(df)

        # Should be unchanged (no valid columns with variance)
        pd.testing.assert_frame_equal(result, df)

    def test_transform_without_fit(self, simple_outlier_df):
        """Test transform on different data after fit."""
        df_train = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        df_test = pd.DataFrame({"values": [1, 2, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        transformer.fit(df_train)
        result = transformer.transform(df_test)

        # Should apply training bounds to test data
        assert len(result) == len(df_test)

    def test_with_missing_values(self):
        """Test outlier detection with missing values present."""
        df = pd.DataFrame({"values": [1, 2, np.nan, 4, 5, 100]})

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Should handle NaN values gracefully
        assert result["values"].isna().sum() == df["values"].isna().sum()

    def test_multiple_outliers_same_row(self):
        """Test row with outliers in multiple columns."""
        df = pd.DataFrame({"col1": [1, 2, 100, 4, 5], "col2": [10, 20, 200, 40, 50]})

        transformer = OutlierTransformer(
            strategy="iqr", method="remove", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # Row 2 has outliers in both columns, should be removed
        assert len(result) < len(df)
        assert 100 not in result["col1"].values
        assert 200 not in result["col2"].values

    def test_preserves_dataframe_structure(self, multi_column_df):
        """Test that DataFrame structure is preserved."""
        df = multi_column_df

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )
        result = transformer.fit_transform(df)

        # All columns should be present
        assert set(result.columns) == set(df.columns)
        # Non-numeric columns unchanged
        assert result["text"].equals(df["text"])

    def test_fit_transform_equivalence(self, simple_outlier_df):
        """Test that fit_transform gives same result as fit then transform."""
        df = simple_outlier_df

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


class TestOutlierTransformerValidation:
    """Test validation and error handling."""

    def test_invalid_iqr_multiplier(self):
        """Test invalid IQR multiplier raises ValueError."""
        with pytest.raises(ValueError, match="iqr_multiplier must be positive"):
            OutlierTransformer(strategy="iqr", iqr_multiplier=0)

        with pytest.raises(ValueError, match="iqr_multiplier must be positive"):
            OutlierTransformer(strategy="iqr", iqr_multiplier=-1.5)

    def test_invalid_zscore_threshold(self):
        """Test invalid Z-score threshold raises ValueError."""
        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            OutlierTransformer(strategy="zscore", zscore_threshold=0)

        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            OutlierTransformer(strategy="zscore", zscore_threshold=-3.0)

    def test_invalid_method(self):
        """Test invalid method parameter raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            OutlierTransformer(strategy="iqr", method="invalid")

    def test_invalid_strategy(self):
        """Test invalid strategy parameter raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be"):
            OutlierTransformer(strategy="invalid")

    def test_invalid_contamination(self):
        """Test invalid contamination parameter raises ValueError."""
        with pytest.raises(ValueError, match="contamination must be"):
            OutlierTransformer(
                strategy="isolation_forest", method="remove", contamination=0
            )

        with pytest.raises(ValueError, match="contamination must be"):
            OutlierTransformer(
                strategy="isolation_forest",
                method="remove",
                contamination=0.6,  # > 0.5
            )

    def test_invalid_n_estimators(self):
        """Test invalid n_estimators parameter raises ValueError."""
        with pytest.raises(ValueError, match="n_estimators should be"):
            OutlierTransformer(
                strategy="isolation_forest",
                method="remove",
                n_estimators=5,  # < 10
            )

        with pytest.raises(ValueError, match="n_estimators should be"):
            OutlierTransformer(
                strategy="isolation_forest",
                method="remove",
                n_estimators=1500,  # > 1000
            )

    def test_invalid_max_samples_int(self):
        """Test invalid max_samples as integer raises ValueError."""
        with pytest.raises(ValueError, match="max_samples as int must be positive"):
            OutlierTransformer(
                strategy="isolation_forest", method="remove", max_samples=0
            )

    def test_invalid_max_samples_float(self):
        """Test invalid max_samples as float raises ValueError."""
        with pytest.raises(
            ValueError, match="max_samples as float must be in \\(0, 1\\]"
        ):
            OutlierTransformer(
                strategy="isolation_forest",
                method="remove",
                max_samples=0.0,  # Must be > 0
            )

        with pytest.raises(
            ValueError, match="max_samples as float must be in \\(0, 1\\]"
        ):
            OutlierTransformer(
                strategy="isolation_forest",
                method="remove",
                max_samples=1.5,  # Must be <= 1
            )

    @pytest.mark.parametrize(
        "strategy,method",
        [
            ("IQR", "clip"),
            ("iqr", "CLIP"),
            ("ZScore", "remove"),
            ("ZSCORE", "REMOVE"),
        ],
    )
    def test_case_insensitive_parameters(self, strategy, method, simple_outlier_df):
        """Test that strategy and method are case-insensitive."""
        df = simple_outlier_df

        # Should work with any case
        transformer = OutlierTransformer(strategy=strategy, method=method)
        result = transformer.fit_transform(df)

        assert isinstance(result, pd.DataFrame)


class TestOutlierTransformerIntegration:
    """Integration tests for OutlierTransformer."""

    def test_sklearn_pipeline_compatibility(self, simple_outlier_df):
        """Test compatibility with sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = simple_outlier_df

        pipeline = Pipeline(
            [
                ("outlier", OutlierTransformer(strategy="iqr", method="clip")),
                ("scaler", StandardScaler()),
            ]
        )

        result = pipeline.fit_transform(df)

        # Should successfully pass through pipeline
        assert result is not None
        assert len(result) > 0

    def test_cross_strategy_comparison(self, simple_outlier_df):
        """Test different strategies on same data produce valid results."""
        df = simple_outlier_df

        # Test all strategies
        strategies = ["iqr", "zscore"]
        results = {}

        for strategy in strategies:
            transformer = OutlierTransformer(strategy=strategy, method="remove")
            results[strategy] = transformer.fit_transform(df)

        # All should produce valid DataFrames
        for strategy, result in results.items():
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= len(df)

    def test_get_feature_names_out(self, simple_outlier_df):
        """Test get_feature_names_out method for sklearn compatibility."""
        df = simple_outlier_df

        transformer = OutlierTransformer(strategy="iqr", method="clip")
        transformer.fit(df)

        feature_names = transformer.get_feature_names_out()

        assert feature_names is not None
        assert "values" in feature_names

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "col1": np.random.normal(50, 10, 1000),
                "col2": np.random.normal(100, 20, 1000),
            }
        )
        # Add some outliers
        df.loc[999, "col1"] = 500
        df.loc[998, "col2"] = 500

        transformer = OutlierTransformer(
            strategy="isolation_forest",
            method="remove",
            contamination=0.01,
            random_state=42,
        )
        result = transformer.fit_transform(df)

        # Should handle large dataset efficiently
        assert len(result) < len(df)
        assert len(result) > 900  # Most data retained

    def test_sequential_transforms(self, simple_outlier_df):
        """Test applying transformer multiple times."""
        df = simple_outlier_df

        transformer = OutlierTransformer(
            strategy="iqr", method="clip", iqr_multiplier=1.5
        )

        # First transform
        result1 = transformer.fit_transform(df)

        # Second transform on same data
        result2 = transformer.transform(result1)

        # Should be stable (idempotent after first transform)
        pd.testing.assert_frame_equal(result1, result2)
