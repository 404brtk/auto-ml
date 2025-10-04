import pandas as pd
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def infer_task(
    df: pd.DataFrame,
    target: str,
    classification_cardinality_threshold: int = 30,
) -> TaskType:
    """Infer whether a ML task is classification or regression based on target variable."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")

    y = df[target].copy()

    # Handle edge cases
    if len(y) == 0:
        logger.warning("Empty target column, defaulting to classification")
        return TaskType.classification

    # Remove NaN for analysis
    y_clean = y.dropna()
    if len(y_clean) == 0:
        logger.warning(
            "Target column contains only NaN values, defaulting to classification"
        )
        return TaskType.classification

    # Warn if very few samples
    if len(y_clean) < 10:
        logger.warning(
            "Very few samples (%d), task inference may be unreliable", len(y_clean)
        )

    # Single unique value -> classification (constant prediction)
    if y_clean.nunique() <= 1:
        logger.info("Target has single unique value, treating as classification")
        return TaskType.classification

    # Boolean dtype -> classification
    if y.dtype.kind == "b":
        logger.info("Boolean target detected -> classification")
        return TaskType.classification

    # Categorical dtype -> classification
    if isinstance(y.dtype, pd.CategoricalDtype):
        logger.info("Categorical target detected -> classification")
        return TaskType.classification

    # Object dtypes
    if y.dtype.kind == "O":
        logger.info(
            "Object target detected after cleaning - treating as classification"
        )
        return TaskType.classification

    # Numeric dtypes -> decide based on cardinality and dtype
    return _infer_from_numeric_column(y_clean, classification_cardinality_threshold)


def _infer_from_numeric_column(y: pd.Series, cardinality_threshold: int) -> TaskType:
    """Infer task type from numeric column."""

    nunique = y.nunique(dropna=True)
    dtype_kind = y.dtype.kind

    # Float types -> regression (continuous by nature)
    if dtype_kind in {"f"}:
        logger.info(
            "Float target (dtype=%s) with %d unique values -> regression",
            y.dtype,
            nunique,
        )
        return TaskType.regression

    # Integer types: check cardinality and uniqueness ratio
    if dtype_kind in {"i", "u"}:
        uniqueness_ratio = nunique / len(y) if len(y) > 0 else 1.0

        # Classification if: low cardinality AND not too many unique values relative to samples
        if nunique <= cardinality_threshold and uniqueness_ratio < 0.5:
            logger.info(
                "Integer target with %d unique values (%.1f%% of samples) -> classification",
                nunique,
                100 * uniqueness_ratio,
            )
            return TaskType.classification
        else:
            logger.info(
                "Integer target with %d unique values (%.1f%% of samples, threshold: %d) -> regression",
                nunique,
                100 * uniqueness_ratio,
                cardinality_threshold,
            )
            return TaskType.regression

    # Fallback for other numeric types
    logger.info("Numeric target -> regression (fallback)")
    return TaskType.regression
