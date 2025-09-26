from typing import Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
)
from sklearn.pipeline import Pipeline
from auto_ml_pipeline.config import FeatureSelectionConfig, TaskType
from auto_ml_pipeline.logging_utils import get_logger
from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures

logger = get_logger(__name__)


def build_selector(cfg: FeatureSelectionConfig, task: TaskType) -> Optional[Pipeline]:
    """Build feature selection pipeline based on configuration."""
    steps = []

    # Constant feature removal
    if cfg.remove_constant:
        steps.append(("constant", DropConstantFeatures(tol=cfg.constant_tolerance)))

    # Variance threshold
    if cfg.variance_threshold is not None and cfg.variance_threshold >= 0:
        steps.append(("variance", VarianceThreshold(threshold=cfg.variance_threshold)))

    # Correlation filtering
    if cfg.correlation_threshold is not None and 0 < cfg.correlation_threshold < 1:
        steps.append(
            (
                "correlation",
                DropCorrelatedFeatures(
                    method=cfg.correlation_method, threshold=cfg.correlation_threshold
                ),
            )
        )

    # Mutual information
    if cfg.mutual_info_k is not None and cfg.mutual_info_k > 0:
        score_func = (
            mutual_info_classif
            if task == TaskType.classification
            else mutual_info_regression
        )
        steps.append(
            ("mutual_info", SelectKBest(score_func=score_func, k=cfg.mutual_info_k))
        )

    # PCA - dimensionality reduction
    if cfg.pca_components is not None:
        # Validate PCA components
        if isinstance(cfg.pca_components, float):
            if not (0 < cfg.pca_components <= 1):
                logger.warning(
                    "PCA variance ratio should be in (0,1], got %s", cfg.pca_components
                )
                return Pipeline(steps) if steps else None
        elif isinstance(cfg.pca_components, int):
            if cfg.pca_components < 1:
                logger.warning(
                    "PCA n_components should be >= 1, got %s", cfg.pca_components
                )
                return Pipeline(steps) if steps else None

        steps.append(("pca", PCA(n_components=cfg.pca_components)))

    return Pipeline(steps) if steps else None
