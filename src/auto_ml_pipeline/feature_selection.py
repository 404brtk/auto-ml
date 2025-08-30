from typing import Optional, List
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline
from auto_ml_pipeline.config import FeatureSelectionConfig, TaskType
from auto_ml_pipeline.selection_utils import correlation_threshold_columns
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.drop_cols_: List[str] = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.drop_cols_ = correlation_threshold_columns(X, self.threshold)
        else:  # best-effort: if array, skip
            self.drop_cols_ = []
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.drop_cols_:
            return X.drop(columns=self.drop_cols_)
        return X


def build_selector(cfg: FeatureSelectionConfig, task: TaskType) -> Optional[Pipeline]:
    steps = []
    if cfg.correlation_threshold is not None:
        steps.append(("corr", CorrelationFilter(threshold=cfg.correlation_threshold)))
    if cfg.pca:
        steps.append(("pca", PCA(n_components=cfg.pca_variance, svd_solver="full")))
    if cfg.mutual_info_k:
        if task == TaskType.classification:
            steps.append(
                ("mi", SelectKBest(score_func=mutual_info_classif, k=cfg.mutual_info_k))
            )
        elif task == TaskType.regression:
            steps.append(
                (
                    "mi",
                    SelectKBest(score_func=mutual_info_regression, k=cfg.mutual_info_k),
                )
            )
    if not steps:
        return None
    return Pipeline(steps)
