import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from auto_ml_pipeline.feature_selection import build_selector
from auto_ml_pipeline.config import FeatureSelectionConfig, TaskType


class TestBuildSelector:
    """Test build_selector function."""

    def test_no_selection_returns_none(self):
        """Test that no feature selection returns None."""
        cfg = FeatureSelectionConfig(
            remove_constant=False,
            variance_threshold=None,
            correlation_threshold=None,
            mutual_info_k=None,
            pca_components=None,
        )

        selector = build_selector(cfg, TaskType.classification)
        assert selector is None

    def test_constant_feature_removal(self):
        """Test constant feature removal."""
        cfg = FeatureSelectionConfig(remove_constant=True, constant_tolerance=0.01)

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert isinstance(selector, Pipeline)
        assert "constant" in dict(selector.steps)

    def test_variance_threshold(self):
        """Test variance threshold selection."""
        cfg = FeatureSelectionConfig(variance_threshold=0.1)

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert "variance" in dict(selector.steps)

    def test_variance_threshold_negative_raises_error(self):
        """Test that negative variance threshold raises validation error."""
        from pydantic import ValidationError

        # Pydantic should prevent negative variance threshold
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(variance_threshold=-0.1)

    def test_correlation_filtering(self):
        """Test correlation-based feature selection."""
        cfg = FeatureSelectionConfig(
            correlation_threshold=0.9, correlation_method="pearson"
        )

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert "correlation" in dict(selector.steps)

    def test_correlation_invalid_threshold_raises_error(self):
        """Test that invalid correlation thresholds raise validation error."""
        from pydantic import ValidationError

        # Threshold = 0 (invalid - should be > 0)
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(correlation_threshold=0.0)

        # Threshold = 1.0 (invalid - should be < 1)
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(correlation_threshold=1.0)

    def test_mutual_info_classification(self):
        """Test mutual information for classification."""
        cfg = FeatureSelectionConfig(mutual_info_k=10)

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert "mutual_info" in dict(selector.steps)

    def test_mutual_info_regression(self):
        """Test mutual information for regression."""
        cfg = FeatureSelectionConfig(mutual_info_k=10)

        selector = build_selector(cfg, TaskType.regression)
        assert selector is not None
        assert "mutual_info" in dict(selector.steps)

    def test_mutual_info_zero_k_raises_error(self):
        """Test that k=0 raises validation error."""
        from pydantic import ValidationError

        # k must be > 0
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(mutual_info_k=0)

    def test_pca_with_variance_ratio(self):
        """Test PCA with variance ratio."""
        cfg = FeatureSelectionConfig(pca_components=0.95)

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert "pca" in dict(selector.steps)

    def test_pca_with_n_components(self):
        """Test PCA with number of components."""
        cfg = FeatureSelectionConfig(pca_components=10)

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        assert "pca" in dict(selector.steps)

    def test_pca_invalid_variance_ratio_raises_error(self):
        """Test that invalid PCA variance ratios raise validation error."""
        from pydantic import ValidationError

        # Too high (> 1.0)
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(pca_components=1.5)

        # Too low (<= 0.0)
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(pca_components=0.0)

    def test_pca_invalid_n_components_raises_error(self):
        """Test that invalid PCA n_components raise validation error."""
        from pydantic import ValidationError

        # n_components must be > 0 when integer
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(pca_components=0)

    def test_multiple_selectors_combined(self):
        """Test combining multiple feature selection methods."""
        cfg = FeatureSelectionConfig(
            remove_constant=True,
            variance_threshold=0.1,
            correlation_threshold=0.9,
            mutual_info_k=10,
        )

        selector = build_selector(cfg, TaskType.classification)
        assert selector is not None
        steps_dict = dict(selector.steps)

        # Should have all four steps
        assert "constant" in steps_dict
        assert "variance" in steps_dict
        assert "correlation" in steps_dict
        assert "mutual_info" in steps_dict

    def test_pipeline_ordering(self):
        """Test that selection steps are in correct order."""
        cfg = FeatureSelectionConfig(
            remove_constant=True,
            variance_threshold=0.1,
            correlation_threshold=0.9,
            mutual_info_k=10,
            pca_components=5,
        )

        selector = build_selector(cfg, TaskType.classification)
        step_names = [name for name, _ in selector.steps]

        # Check expected ordering
        expected_order = ["constant", "variance", "correlation", "mutual_info", "pca"]
        assert step_names == expected_order

    def test_correlation_methods(self):
        """Test different correlation methods."""
        for method in ["pearson", "spearman", "kendall"]:
            cfg = FeatureSelectionConfig(
                correlation_threshold=0.9, correlation_method=method
            )

            selector = build_selector(cfg, TaskType.classification)
            assert selector is not None


class TestFeatureSelectorIntegration:
    """Integration tests for feature selection with real data."""

    def test_constant_removal_integration(self):
        """Test constant feature removal with real data."""
        X = pd.DataFrame({"const": [1, 1, 1, 1], "var": [1, 2, 3, 4]})
        y = np.array([0, 1, 0, 1])

        cfg = FeatureSelectionConfig(remove_constant=True)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Constant column should be removed
        assert X_transformed.shape[1] == 1

    def test_variance_threshold_integration(self):
        """Test variance threshold with real data."""
        X = pd.DataFrame(
            {
                "low_var": [1.0, 1.0, 1.1, 1.0],  # Very low variance
                "high_var": [1, 2, 3, 4],  # Higher variance
            }
        )
        y = np.array([0, 1, 0, 1])

        cfg = FeatureSelectionConfig(variance_threshold=0.05)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Low variance column should be removed
        assert X_transformed.shape[1] == 1

    def test_correlation_filtering_integration(self):
        """Test correlation filtering with real data."""
        X = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],  # Perfectly correlated with feat1
                "feat3": [5, 4, 3, 2, 1],  # Independent
            }
        )
        y = np.array([0, 1, 0, 1, 0])

        cfg = FeatureSelectionConfig(correlation_threshold=0.95)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # One of the correlated features should be removed
        assert X_transformed.shape[1] < X.shape[1]

    def test_mutual_info_integration(self):
        """Test mutual information selection with real data."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "informative1": np.random.randn(100),
                "informative2": np.random.randn(100),
                "noise1": np.random.randn(100),
                "noise2": np.random.randn(100),
                "noise3": np.random.randn(100),
            }
        )
        # Make first two features informative
        y = (X["informative1"] + X["informative2"] > 0).astype(int)

        cfg = FeatureSelectionConfig(mutual_info_k=2)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Should select top 2 features
        assert X_transformed.shape[1] == 2

    def test_pca_integration(self):
        """Test PCA dimensionality reduction with real data."""
        X = pd.DataFrame(np.random.randn(100, 20))
        y = np.random.randint(0, 2, 100)

        cfg = FeatureSelectionConfig(pca_components=5)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Should reduce to 5 components
        assert X_transformed.shape[1] == 5

    def test_pca_variance_ratio_integration(self):
        """Test PCA with variance ratio."""
        X = pd.DataFrame(np.random.randn(100, 20))
        y = np.random.randint(0, 2, 100)

        cfg = FeatureSelectionConfig(pca_components=0.95)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Should reduce dimensions while keeping 95% variance
        assert X_transformed.shape[1] < X.shape[1]

    def test_full_pipeline_integration(self):
        """Test complete feature selection pipeline."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "const": [1] * 100,  # Constant
                "low_var": [1.0] * 95 + [1.1] * 5,  # Low variance
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100) * 0.01,  # Correlated with noise
                "feat3": np.random.randn(100),
                "feat4": np.random.randn(100),
                "feat5": np.random.randn(100),
            }
        )
        y = (X["feat1"] + X["feat3"] > 0).astype(int)

        cfg = FeatureSelectionConfig(
            remove_constant=True,
            variance_threshold=0.05,
            correlation_threshold=0.95,
            mutual_info_k=3,
        )
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Should progressively reduce features
        assert X_transformed.shape[1] < X.shape[1]

    def test_selector_preserves_arrays(self):
        """Test that selector works with numpy arrays."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        cfg = FeatureSelectionConfig(variance_threshold=0.1)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, np.ndarray)

    def test_fit_transform_equivalence(self):
        """Test that fit_transform equals fit then transform."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.random.randint(0, 2, 100)

        cfg = FeatureSelectionConfig(remove_constant=True, variance_threshold=0.1)

        selector1 = build_selector(cfg, TaskType.classification)
        X_fit_transform = selector1.fit_transform(X, y)

        selector2 = build_selector(cfg, TaskType.classification)
        selector2.fit(X, y)
        X_fit_then_transform = selector2.transform(X)

        np.testing.assert_array_almost_equal(X_fit_transform, X_fit_then_transform)

    def test_transform_on_new_data(self):
        """Test transform on different data than training."""
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = np.random.randint(0, 2, 100)
        X_test = pd.DataFrame(np.random.randn(50, 10))

        cfg = FeatureSelectionConfig(variance_threshold=0.1)
        selector = build_selector(cfg, TaskType.classification)

        selector.fit(X_train, y_train)
        X_test_transformed = selector.transform(X_test)

        # Should apply same transformations
        assert X_test_transformed.shape[0] == X_test.shape[0]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pd.DataFrame()
        y = np.array([])

        cfg = FeatureSelectionConfig(remove_constant=True)
        selector = build_selector(cfg, TaskType.classification)

        # Empty dataframe may cause errors in sklearn, so skip if selector exists
        # This is an expected edge case - empty data should be handled before feature selection
        if selector is not None:
            # sklearn requires at least 1 feature, so this may raise ValueError
            # which is acceptable behavior for edge case
            try:
                X_transformed = selector.fit_transform(X, y)
                assert X_transformed.shape[0] == 0
            except ValueError as e:
                # Expected for empty DataFrame
                assert "minimum of 1" in str(e) or "0 feature" in str(e)

    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame({"feat": np.random.randn(100)})
        y = np.random.randint(0, 2, 100)

        cfg = FeatureSelectionConfig(variance_threshold=0.01)
        selector = build_selector(cfg, TaskType.classification)

        X_transformed = selector.fit_transform(X, y)

        # Should keep the single feature if it has variance
        assert X_transformed.shape[1] == 1
