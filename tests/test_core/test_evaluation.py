"""Tests for evaluation module."""

import numpy as np
from auto_ml_pipeline.evaluation import evaluate_predictions
from auto_ml_pipeline.config import TaskType


class TestEvaluatePredictionsClassification:
    """Test evaluate_predictions for classification tasks."""

    def test_binary_classification_basic_metrics(self):
        """Test basic metrics for binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_binary_classification_with_probabilities(self):
        """Test binary classification with probability scores for ROC AUC."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_proba = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]]
        )

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_multiclass_classification(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong_predictions(self):
        """Test with all wrong predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_single_class_in_true_labels(self):
        """Test when only one class is present (ROC AUC may be NaN)."""
        import warnings

        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0])
        y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])

        # Catch and verify the expected warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = evaluate_predictions(
                TaskType.classification, y_true, y_pred, y_proba
            )
            # Verify warning was raised (expected behavior)
            assert len(w) >= 1
            assert "ROC AUC" in str(w[0].message)

        # ROC AUC may be included but with NaN value, or skipped
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        # If roc_auc is present, it should be NaN for single class
        if "roc_auc" in metrics:
            assert np.isnan(metrics["roc_auc"])

    def test_multiclass_with_probabilities(self):
        """Test multiclass with probabilities (no ROC AUC)."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.9, 0.05, 0.05],
                [0.1, 0.7, 0.2],
                [0.2, 0.7, 0.1],
            ]
        )

        metrics = evaluate_predictions(TaskType.classification, y_true, y_pred, y_proba)

        # ROC AUC not computed for multiclass (shape[1] != 2)
        assert "roc_auc" not in metrics
        assert "accuracy" in metrics


class TestEvaluatePredictionsRegression:
    """Test evaluate_predictions for regression tasks."""

    def test_regression_basic_metrics(self):
        """Test basic regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_perfect_regression_predictions(self):
        """Test regression with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_poor_regression_predictions(self):
        """Test regression with poor predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] < 1

    def test_constant_predictions(self):
        """Test regression with constant predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        # R2 = 0 for constant predictions equal to mean

    def test_regression_with_large_values(self):
        """Test regression with large values."""
        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        y_pred = np.array([1100.0, 2100.0, 2900.0, 4200.0, 4800.0])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_regression_ignores_probabilities(self):
        """Test that regression ignores probability parameter."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred, y_proba)

        # Should only have regression metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "accuracy" not in metrics
        assert "roc_auc" not in metrics

    def test_regression_with_negative_values(self):
        """Test regression with negative values."""
        y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        y_pred = np.array([-4.8, -3.2, -0.9, 1.1, 2.8, 5.2])

        metrics = evaluate_predictions(TaskType.regression, y_true, y_pred)

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1
