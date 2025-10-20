"""Tests for metrics system."""

import pytest
import numpy as np
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.metrics import (
    MetricsRegistry,
    compute_metrics,
    get_optimization_metric,
    get_optimization_metric_from_config,
    get_evaluation_metrics_from_config,
)


@pytest.fixture
def binary_classification_data():
    """Binary classification test data."""
    return {
        "y_true": np.array([0, 1, 0, 1, 0, 1]),
        "y_pred": np.array([0, 1, 0, 1, 0, 0]),
        "y_proba": np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.6, 0.4],
            ]
        ),
    }


@pytest.fixture
def multiclass_classification_data():
    """Multiclass classification test data."""
    return {
        "y_true": np.array([0, 1, 2, 0, 1, 2]),
        "y_pred": np.array([0, 1, 2, 0, 1, 1]),
        "y_proba": np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.9, 0.05, 0.05],
                [0.1, 0.7, 0.2],
                [0.2, 0.7, 0.1],
            ]
        ),
    }


@pytest.fixture
def regression_data():
    """Regression test data."""
    return {
        "y_true": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "y_pred": np.array([1.1, 2.1, 2.9, 4.2, 4.8]),
    }


@pytest.fixture
def mock_metrics_config():
    """Mock MetricsConfig for testing."""
    from unittest.mock import Mock

    config = Mock()
    config.classification_optimization_metric = "f1_macro"
    config.classification_evaluation_metrics = ["accuracy", "f1_macro"]
    config.regression_optimization_metric = "rmse"
    config.regression_evaluation_metrics = ["rmse", "mae"]
    return config


class TestMetricsRegistry:
    """Test MetricsRegistry class."""

    def test_get_available_metrics_classification(self):
        """Test getting available classification metrics."""
        metrics = MetricsRegistry.get_available_metrics(TaskType.classification)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_weighted" in metrics
        assert "roc_auc" in metrics

        # Should not have regression metrics
        assert "rmse" not in metrics
        assert "mae" not in metrics

    def test_get_available_metrics_regression(self):
        """Test getting available regression metrics."""
        metrics = MetricsRegistry.get_available_metrics(TaskType.regression)

        assert "rmse" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "r2" in metrics

        # Should not have classification metrics
        assert "accuracy" not in metrics
        assert "f1_macro" not in metrics

    def test_get_metric_names(self):
        """Test getting metric names as a set."""
        names = MetricsRegistry.get_metric_names(TaskType.classification)

        assert isinstance(names, set)
        assert "accuracy" in names
        assert len(names) == 8  # 8 classification metrics

    def test_get_default_metrics(self):
        """Test getting default metrics."""
        defaults_clf = MetricsRegistry.get_default_metrics(TaskType.classification)
        assert defaults_clf == [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ]

        defaults_reg = MetricsRegistry.get_default_metrics(TaskType.regression)
        assert defaults_reg == ["rmse", "mae", "r2"]

    def test_get_default_optimization_metric(self):
        """Test getting default optimization metric."""
        assert (
            MetricsRegistry.get_default_optimization_metric(TaskType.classification)
            == "f1_macro"
        )
        assert (
            MetricsRegistry.get_default_optimization_metric(TaskType.regression)
            == "rmse"
        )

    def test_get_metric(self):
        """Test getting a specific metric definition."""
        metric_def = MetricsRegistry.get_metric(TaskType.classification, "accuracy")

        assert metric_def is not None
        assert metric_def.name == "accuracy"
        assert metric_def.display_name == "Accuracy"
        assert metric_def.sklearn_scorer_name == "accuracy"
        assert not metric_def.requires_proba
        assert metric_def.higher_is_better

    def test_validate_metrics_valid(self):
        """Test validation with valid metrics."""
        # Should not raise any exception
        MetricsRegistry.validate_metrics(
            TaskType.classification, ["accuracy", "f1_macro"]
        )

        MetricsRegistry.validate_metrics(TaskType.regression, ["rmse", "mae", "r2"])

    def test_validate_metrics_invalid(self):
        """Test validation with invalid metrics."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            MetricsRegistry.validate_metrics(
                TaskType.classification,
                ["accuracy", "rmse"],  # rmse is regression metric
            )

        with pytest.raises(ValueError, match="Invalid metrics"):
            MetricsRegistry.validate_metrics(
                TaskType.regression,
                ["mae", "accuracy"],  # accuracy is classification metric
            )


class TestClassificationMetrics:
    """Test classification metrics computation."""

    def test_basic_metrics(self, binary_classification_data):
        """Test basic classification metrics."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
        )

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_with_probabilities(self, binary_classification_data):
        """Test classification with probability scores for ROC AUC."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
            binary_classification_data["y_proba"],
            metric_names=[
                "accuracy",
                "f1_macro",
                "roc_auc",
            ],
        )

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_custom_metrics_list(self, binary_classification_data):
        """Test computing only specific metrics."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
            metric_names=["accuracy", "f1_weighted"],
        )

        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        # Should not have other default metrics
        assert "precision_macro" not in metrics
        assert "recall_macro" not in metrics

    def test_multiclass(self, multiclass_classification_data):
        """Test multiclass classification metrics."""
        metrics = compute_metrics(
            TaskType.classification,
            multiclass_classification_data["y_true"],
            multiclass_classification_data["y_pred"],
        )

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_multiclass_no_roc_auc(self, multiclass_classification_data):
        """Test multiclass doesn't compute ROC AUC."""
        metrics = compute_metrics(
            TaskType.classification,
            multiclass_classification_data["y_true"],
            multiclass_classification_data["y_pred"],
            multiclass_classification_data["y_proba"],
        )

        # ROC AUC should not be computed for multiclass
        assert "roc_auc" not in metrics

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = compute_metrics(TaskType.classification, y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong_predictions(self):
        """Test with all wrong predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        metrics = compute_metrics(TaskType.classification, y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_roc_auc_without_probabilities(self, binary_classification_data):
        """Test that ROC AUC is skipped when probabilities not provided."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
            metric_names=["accuracy", "roc_auc"],
        )

        assert "accuracy" in metrics
        # ROC AUC should be skipped (logged warning, not computed)
        assert "roc_auc" not in metrics

    def test_invalid_metric_names(self, binary_classification_data):
        """Test that invalid metrics raise ValueError."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            compute_metrics(
                TaskType.classification,
                binary_classification_data["y_true"],
                binary_classification_data["y_pred"],
                metric_names=["accuracy", "rmse"],  # rmse is regression
            )


class TestRegressionMetrics:
    """Test regression metrics computation."""

    def test_basic_metrics(self, regression_data):
        """Test basic regression metrics."""
        metrics = compute_metrics(
            TaskType.regression,
            regression_data["y_true"],
            regression_data["y_pred"],
        )

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_custom_metrics_list(self, regression_data):
        """Test computing only specific regression metrics."""
        metrics = compute_metrics(
            TaskType.regression,
            regression_data["y_true"],
            regression_data["y_pred"],
            metric_names=["mae", "mape"],
        )

        assert "mae" in metrics
        assert "mape" in metrics
        # Should not have other metrics
        assert "rmse" not in metrics
        assert "r2" not in metrics

    def test_perfect_predictions(self):
        """Test regression with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_metrics(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_poor_predictions(self):
        """Test regression with poor predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        metrics = compute_metrics(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] < 1

    def test_with_negative_values(self):
        """Test regression with negative values."""
        y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        y_pred = np.array([-4.8, -3.2, -0.9, 1.1, 2.8, 5.2])

        metrics = compute_metrics(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_with_large_values(self):
        """Test regression with large values."""
        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        y_pred = np.array([1100.0, 2100.0, 2900.0, 4200.0, 4800.0])

        metrics = compute_metrics(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_ignores_probabilities(self, regression_data):
        """Test that regression ignores y_proba parameter."""
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        metrics = compute_metrics(
            TaskType.regression,
            regression_data["y_true"],
            regression_data["y_pred"],
            y_proba,
        )

        # Should only have regression metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "accuracy" not in metrics
        assert "roc_auc" not in metrics

    def test_invalid_metric_names(self, regression_data):
        """Test that invalid metrics raise ValueError."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            compute_metrics(
                TaskType.regression,
                regression_data["y_true"],
                regression_data["y_pred"],
                metric_names=["rmse", "accuracy"],  # accuracy is classification
            )


class TestGetOptimizationMetric:
    """Test get_optimization_metric function."""

    def test_default_classification(self):
        """Test default optimization metric for classification."""
        scorer = get_optimization_metric(TaskType.classification, None)
        assert scorer == "f1_macro"

    def test_default_regression(self):
        """Test default optimization metric for regression."""
        scorer = get_optimization_metric(TaskType.regression, None)
        assert scorer == "neg_root_mean_squared_error"

    def test_custom_classification_metric(self):
        """Test custom classification optimization metric."""
        scorer = get_optimization_metric(TaskType.classification, "accuracy")
        assert scorer == "accuracy"

    def test_custom_regression_metric(self):
        """Test custom regression optimization metric."""
        scorer = get_optimization_metric(TaskType.regression, "mae")
        assert scorer == "neg_mean_absolute_error"

    def test_invalid_metric(self):
        """Test invalid optimization metric raises error."""
        with pytest.raises(ValueError, match="Invalid optimization metric"):
            get_optimization_metric(TaskType.classification, "rmse")

    def test_case_insensitive(self):
        """Test that metric names are case-insensitive."""
        scorer = get_optimization_metric(TaskType.classification, "ACCURACY")
        assert scorer == "accuracy"


class TestConfigHelpers:
    """Test config helper functions."""

    def test_get_optimization_metric_from_config_classification(
        self, mock_metrics_config
    ):
        """Test getting optimization metric from config for classification."""
        metric = get_optimization_metric_from_config(
            TaskType.classification, mock_metrics_config
        )
        assert metric == "f1_macro"

    def test_get_optimization_metric_from_config_regression(self, mock_metrics_config):
        """Test getting optimization metric from config for regression."""
        metric = get_optimization_metric_from_config(
            TaskType.regression, mock_metrics_config
        )
        assert metric == "rmse"

    def test_get_evaluation_metrics_from_config_classification(
        self, mock_metrics_config
    ):
        """Test getting evaluation metrics from config for classification."""
        metrics = get_evaluation_metrics_from_config(
            TaskType.classification, mock_metrics_config
        )
        assert metrics == ["accuracy", "f1_macro"]

    def test_get_evaluation_metrics_from_config_regression(self, mock_metrics_config):
        """Test getting evaluation metrics from config for regression."""
        metrics = get_evaluation_metrics_from_config(
            TaskType.regression, mock_metrics_config
        )
        assert metrics == ["rmse", "mae"]

    def test_none_values_passthrough(self):
        """Test that None values are passed through correctly."""
        from unittest.mock import Mock

        config = Mock()
        config.classification_optimization_metric = None
        config.regression_evaluation_metrics = None

        # Should return None (which downstream functions handle)
        assert (
            get_optimization_metric_from_config(TaskType.classification, config) is None
        )
        assert get_evaluation_metrics_from_config(TaskType.regression, config) is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metric_list_uses_defaults(self, binary_classification_data):
        """Test that empty metric list uses defaults."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
            metric_names=[],
        )

        # Should use default metrics
        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_none_metric_list_uses_defaults(self, binary_classification_data):
        """Test that None metric list uses defaults."""
        metrics = compute_metrics(
            TaskType.classification,
            binary_classification_data["y_true"],
            binary_classification_data["y_pred"],
            metric_names=None,
        )

        # Should use default metrics
        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_single_sample(self):
        """Test with single sample (edge case)."""
        y_true = np.array([1.0])
        y_pred = np.array([1.5])

        # Should not crash, though R2 will be undefined
        metrics = compute_metrics(
            TaskType.regression, y_true, y_pred, metric_names=["mae"]
        )

        # At least MAE should work
        assert "mae" in metrics
