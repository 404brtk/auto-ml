from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from auto_ml_pipeline.config import TaskType


def evaluate_predictions(
    task: TaskType, y_true, y_pred, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Evaluate predictions and return metrics."""

    if task == TaskType.classification:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

        if y_proba is not None and y_proba.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except ValueError:
                # roc_auc_score can raise ValueError when only one class is present
                # in y_true or inputs are not valid; in that case, skip ROC AUC.
                pass

    else:  # regression
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    return metrics
