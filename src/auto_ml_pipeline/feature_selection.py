from typing import Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from auto_ml_pipeline.config import FeatureSelectionConfig, TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated features, keeping the first occurrence."""

    def __init__(self, threshold: float = 0.95):
        if not (0.0 <= threshold < 1.0):
            raise ValueError(f"threshold must be in [0, 1), got {threshold}")
        self.threshold = threshold
        self.columns_to_drop_: List[str] = []
        self.indices_to_drop_: List[int] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the correlation filter on training data."""
        if isinstance(X, pd.DataFrame):
            self.columns_to_drop_ = self._find_correlated_columns_df(X)
            self.indices_to_drop_ = []
        else:
            self.columns_to_drop_ = []
            self.indices_to_drop_ = self._find_correlated_indices_array(X)
        return self

    def _find_correlated_columns_df(self, X: pd.DataFrame) -> List[str]:
        """Find correlated columns in DataFrame."""
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty or X_numeric.shape[1] <= 1:
            return []

        try:
            corr = X_numeric.corr().abs().fillna(0.0)
            cols = list(X_numeric.columns)
            to_drop = set()

            for i, col1 in enumerate(cols):
                if col1 in to_drop:
                    continue
                for col2 in cols[i + 1 :]:
                    if col2 not in to_drop and corr.loc[col1, col2] > self.threshold:
                        to_drop.add(col2)

            dropped = list(to_drop)
            if dropped:
                logger.info(
                    "[CorrFilter] Found %d correlated features to drop (threshold=%.3f)",
                    len(dropped),
                    self.threshold,
                )
            return dropped

        except Exception as e:
            logger.warning("Error computing correlations: %s", e)
            return []

    def _find_correlated_indices_array(self, X: np.ndarray) -> List[int]:
        """Find correlated feature indices in numpy array."""
        if X.ndim != 2 or X.shape[1] <= 1:
            return []

        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(X, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)

            n_features = corr.shape[0]
            to_drop = set()

            for i in range(n_features):
                if i in to_drop:
                    continue
                for j in range(i + 1, n_features):
                    if j not in to_drop and abs(corr[i, j]) >= self.threshold:
                        to_drop.add(j)

            dropped = sorted(to_drop)
            if dropped:
                logger.info(
                    "[CorrFilter] Found %d correlated features to drop (threshold=%.3f)",
                    len(dropped),
                    self.threshold,
                )
            return dropped

        except Exception as e:
            logger.warning("Error computing correlations: %s", e)
            return []

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply correlation filtering."""
        if isinstance(X, pd.DataFrame) and self.columns_to_drop_:
            cols_to_drop = [col for col in self.columns_to_drop_ if col in X.columns]
            return X.drop(columns=cols_to_drop) if cols_to_drop else X
        elif isinstance(X, np.ndarray) and self.indices_to_drop_:
            return np.delete(X, self.indices_to_drop_, axis=1)
        return X


def build_selector(cfg: FeatureSelectionConfig, task: TaskType) -> Optional[Pipeline]:
    """Build feature selection pipeline based on configuration."""
    steps = []

    # Variance threshold
    if cfg.variance_threshold is not None:
        threshold = max(0.0, cfg.variance_threshold)
        steps.append(("variance", VarianceThreshold(threshold=threshold)))

    # Correlation filtering
    if cfg.correlation_threshold is not None:
        threshold = min(cfg.correlation_threshold, 0.999)
        steps.append(("correlation", CorrelationFilter(threshold=threshold)))

    # PCA
    if getattr(cfg, "pca_components", None) is not None:
        steps.append(("pca", PCA(n_components=cfg.pca_components)))

    # Mutual information
    if cfg.mutual_info_k is not None:
        score_func = (
            mutual_info_classif
            if task == TaskType.classification
            else mutual_info_regression
        )
        # Ensure k doesn't exceed available features (will be handled by pipeline)
        steps.append(
            ("mutual_info", SelectKBest(score_func=score_func, k=cfg.mutual_info_k))
        )

    return Pipeline(steps) if steps else None
