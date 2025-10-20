from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MetricDefinition:
    name: str
    display_name: str
    sklearn_scorer_name: str  # name used by sklearn's cross_val_score
    compute_fn: Callable
    requires_proba: bool = False  # whether metric needs probability predictions
    higher_is_better: bool = True


class MetricsRegistry:
    _classification_metrics = {
        "accuracy": MetricDefinition(
            name="accuracy",
            display_name="Accuracy",
            sklearn_scorer_name="accuracy",
            compute_fn=lambda y_true, y_pred, y_proba=None: accuracy_score(
                y_true, y_pred
            ),
            requires_proba=False,
            higher_is_better=True,
        ),
        "precision_macro": MetricDefinition(
            name="precision_macro",
            display_name="Precision (Macro)",
            sklearn_scorer_name="precision_macro",
            compute_fn=lambda y_true, y_pred, y_proba=None: precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            requires_proba=False,
            higher_is_better=True,
        ),
        "recall_macro": MetricDefinition(
            name="recall_macro",
            display_name="Recall (Macro)",
            sklearn_scorer_name="recall_macro",
            compute_fn=lambda y_true, y_pred, y_proba=None: recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            requires_proba=False,
            higher_is_better=True,
        ),
        "f1_macro": MetricDefinition(
            name="f1_macro",
            display_name="F1 Score (Macro)",
            sklearn_scorer_name="f1_macro",
            compute_fn=lambda y_true, y_pred, y_proba=None: f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            requires_proba=False,
            higher_is_better=True,
        ),
        "f1_weighted": MetricDefinition(
            name="f1_weighted",
            display_name="F1 Score (Weighted)",
            sklearn_scorer_name="f1_weighted",
            compute_fn=lambda y_true, y_pred, y_proba=None: f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            requires_proba=False,
            higher_is_better=True,
        ),
        "roc_auc": MetricDefinition(
            name="roc_auc",
            display_name="ROC AUC",
            sklearn_scorer_name="roc_auc",
            compute_fn=lambda y_true, y_pred, y_proba=None: (
                roc_auc_score(y_true, y_proba[:, 1])
                if y_proba is not None and y_proba.shape[1] == 2
                else None  # Explicitly for binary classification
            ),
            requires_proba=True,
            higher_is_better=True,
        ),
        "roc_auc_macro": MetricDefinition(
            name="roc_auc_macro",
            display_name="ROC AUC (Macro)",
            sklearn_scorer_name="roc_auc_ovr",
            compute_fn=lambda y_true, y_pred, y_proba=None: (
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                if y_proba is not None and len(np.unique(y_true)) > 2
                else None
            ),
            requires_proba=True,
            higher_is_better=True,
        ),
        "roc_auc_weighted": MetricDefinition(
            name="roc_auc_weighted",
            display_name="ROC AUC (Weighted)",
            sklearn_scorer_name="roc_auc_ovr_weighted",
            compute_fn=lambda y_true, y_pred, y_proba=None: (
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                if y_proba is not None and len(np.unique(y_true)) > 2
                else None
            ),
            requires_proba=True,
            higher_is_better=True,
        ),
    }

    _regression_metrics = {
        "rmse": MetricDefinition(
            name="rmse",
            display_name="RMSE",
            sklearn_scorer_name="neg_root_mean_squared_error",
            compute_fn=lambda y_true, y_pred, y_proba=None: np.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            requires_proba=False,
            higher_is_better=False,
        ),
        "mse": MetricDefinition(
            name="mse",
            display_name="MSE",
            sklearn_scorer_name="neg_mean_squared_error",
            compute_fn=lambda y_true, y_pred, y_proba=None: mean_squared_error(
                y_true, y_pred
            ),
            requires_proba=False,
            higher_is_better=False,
        ),
        "mae": MetricDefinition(
            name="mae",
            display_name="MAE",
            sklearn_scorer_name="neg_mean_absolute_error",
            compute_fn=lambda y_true, y_pred, y_proba=None: mean_absolute_error(
                y_true, y_pred
            ),
            requires_proba=False,
            higher_is_better=False,
        ),
        "mape": MetricDefinition(
            name="mape",
            display_name="MAPE",
            sklearn_scorer_name="neg_mean_absolute_percentage_error",
            compute_fn=lambda y_true,
            y_pred,
            y_proba=None: mean_absolute_percentage_error(y_true, y_pred),
            requires_proba=False,
            higher_is_better=False,
        ),
        "r2": MetricDefinition(
            name="r2",
            display_name="R² Score",
            sklearn_scorer_name="r2",
            compute_fn=lambda y_true, y_pred, y_proba=None: r2_score(y_true, y_pred),
            requires_proba=False,
            higher_is_better=True,
        ),
    }

    _defaults = {
        TaskType.classification: [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ],
        TaskType.regression: ["rmse", "mae", "r2"],
    }

    _optimization_defaults = {
        TaskType.classification: "f1_macro",
        TaskType.regression: "rmse",
    }

    @classmethod
    def get_available_metrics(cls, task: TaskType) -> Dict[str, MetricDefinition]:
        if task == TaskType.classification:
            return cls._classification_metrics.copy()
        return cls._regression_metrics.copy()

    @classmethod
    def get_metric_names(cls, task: TaskType) -> Set[str]:
        return set(cls.get_available_metrics(task).keys())

    @classmethod
    def get_default_metrics(cls, task: TaskType) -> List[str]:
        return cls._defaults[task].copy()

    @classmethod
    def get_default_optimization_metric(cls, task: TaskType) -> str:
        return cls._optimization_defaults[task]

    @classmethod
    def get_metric(cls, task: TaskType, metric_name: str) -> Optional[MetricDefinition]:
        return cls.get_available_metrics(task).get(metric_name)

    @classmethod
    def validate_metrics(cls, task: TaskType, metric_names: List[str]) -> None:
        available = cls.get_metric_names(task)
        invalid = set(metric_names) - available

        if invalid:
            raise ValueError(
                f"Invalid metrics for {task.value}: {sorted(invalid)}. "
                f"Available metrics: {sorted(available)}"
            )


def compute_metrics(
    task: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    # Use default metrics if none specified
    if metric_names is None or len(metric_names) == 0:
        metric_names = MetricsRegistry.get_default_metrics(task)
        logger.info(f"Using default metrics for {task.value}: {metric_names}")

    MetricsRegistry.validate_metrics(task, metric_names)

    results = {}
    available_metrics = MetricsRegistry.get_available_metrics(task)

    for metric_name in metric_names:
        metric_def = available_metrics[metric_name]

        try:
            # Skip if metric requires probabilities but they weren't provided
            if metric_def.requires_proba and y_proba is None:
                logger.warning(
                    f"Skipping metric '{metric_name}' - requires probability predictions "
                    "but none were provided"
                )
                continue

            value = metric_def.compute_fn(y_true, y_pred, y_proba)

            # Skip if computation returned None (e.g., ROC AUC for multiclass)
            if value is None:
                logger.warning(
                    f"Skipping metric '{metric_name}' - not applicable for this data"
                )
                continue

            results[metric_name] = float(value)

        except Exception as e:
            logger.warning(f"Failed to compute metric '{metric_name}': {e}. Skipping.")
            continue

    return results


def get_optimization_metric(
    task: TaskType, requested_metric: Optional[str] = None
) -> str:
    if requested_metric is None:
        metric_name = MetricsRegistry.get_default_optimization_metric(task)
    else:
        metric_name = requested_metric.lower()

        available = MetricsRegistry.get_metric_names(task)
        if metric_name not in available:
            raise ValueError(
                f"Invalid optimization metric '{metric_name}' for {task.value}. "
                f"Available: {sorted(available)}"
            )

    metric_def = MetricsRegistry.get_metric(task, metric_name)

    # This should probably never be None due to validation above,
    # but mypy needs it to be satisfied
    if metric_def is None:
        raise ValueError(
            f"Invalid optimization metric '{metric_name}' for {task.value}. "
            f"Available: {sorted(available)}"
        )
    logger.info(f"Using optimization metric: {metric_def.display_name} ({metric_name})")

    return metric_def.sklearn_scorer_name


def get_optimization_metric_from_config(
    task: TaskType, metrics_config
) -> Optional[str]:
    if task == TaskType.classification:
        return metrics_config.classification_optimization_metric
    return metrics_config.regression_optimization_metric


def get_evaluation_metrics_from_config(
    task: TaskType, metrics_config
) -> Optional[List[str]]:
    if task == TaskType.classification:
        return metrics_config.classification_evaluation_metrics
    return metrics_config.regression_evaluation_metrics
