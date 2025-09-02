from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from auto_ml_pipeline.config import TaskType


def evaluate_predictions(
    task: TaskType, y_true, y_pred, y_proba=None
) -> Dict[str, float]:
    res: Dict[str, float] = {}
    if task == TaskType.classification:
        res["accuracy"] = skm.accuracy_score(y_true, y_pred)
        if len(np.unique(y_true)) == 2:
            # try ROC-AUC
            if y_proba is not None and y_proba.ndim == 1:
                res["roc_auc"] = skm.roc_auc_score(y_true, y_proba)
            elif y_proba is not None and y_proba.shape[1] == 2:
                res["roc_auc"] = skm.roc_auc_score(y_true, y_proba[:, 1])
        res["f1_macro"] = skm.f1_score(y_true, y_pred, average="macro")
    elif task == TaskType.regression:
        res["mae"] = skm.mean_absolute_error(y_true, y_pred)
        res["mse"] = skm.mean_squared_error(y_true, y_pred)
        res["rmse"] = float(np.sqrt(res["mse"]))
        res["r2"] = skm.r2_score(y_true, y_pred)
    return {k: float(v) for k, v in res.items()}


def confusion_matrix_df(y_true, y_pred) -> Optional[pd.DataFrame]:
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
