from typing import Optional, List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from auto_ml_pipeline.config import FeatureSelectionConfig, TaskType
from auto_ml_pipeline.selection_utils import correlation_threshold_columns
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from auto_ml_pipeline.logging_utils import get_logger


logger = get_logger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.drop_cols_: List[str] = []
        self.drop_idx_: List[int] = []

    def fit(self, X, y=None):
        # DataFrame path: use column names
        if isinstance(X, pd.DataFrame):
            self.drop_cols_ = correlation_threshold_columns(X, self.threshold)
            self.drop_idx_ = []
        else:
            # ndarray path: compute correlation and mark indices to drop
            X_arr = np.asarray(X)
            self.drop_idx_ = []
            self.drop_cols_ = []
            if X_arr.ndim == 2 and X_arr.shape[1] > 1:
                # protect against NaNs in constant columns
                with np.errstate(invalid="ignore"):
                    corr = np.corrcoef(X_arr, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                # greedy: keep first occurrence, drop later highly-correlated
                n = corr.shape[0]
                to_drop = set()
                for i in range(n):
                    if i in to_drop:
                        continue
                    for j in range(i + 1, n):
                        if j in to_drop:
                            continue
                        if abs(corr[i, j]) >= self.threshold:
                            to_drop.add(j)
                self.drop_idx_ = sorted(to_drop)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.drop_cols_:
                return X.drop(columns=self.drop_cols_)
            return X
        # ndarray path
        if self.drop_idx_:
            return np.delete(np.asarray(X), self.drop_idx_, axis=1)
        return X


def build_selector(cfg: FeatureSelectionConfig, task: TaskType) -> Optional[Pipeline]:
    steps = []
    if cfg.variance_threshold is not None:
        vt = cfg.variance_threshold
        if vt is None:
            pass
        elif vt < 0:
            logger.warning("selection.variance_threshold %.4f < 0; setting to 0.0", vt)
            vt = 0.0
        steps.append(("var", VarianceThreshold(threshold=vt)))
    if cfg.correlation_threshold is not None:
        ct = cfg.correlation_threshold
        if ct is not None and not (0.0 <= ct <= 0.999999):
            logger.warning(
                "selection.correlation_threshold %.4f out of [0,1); clipping to 0.9999",
                ct,
            )
            ct = 0.9999
        steps.append(("corr", CorrelationFilter(threshold=ct)))
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
