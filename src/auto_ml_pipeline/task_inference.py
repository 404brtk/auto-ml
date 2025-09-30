import pandas as pd
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def infer_task(
    df: pd.DataFrame,
    target: str,
    numeric_coercion_threshold: float = 0.9,
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

    # Object dtype -> try numeric conversion
    if y.dtype.kind == "O":
        return _infer_from_object_column(
            y_clean, numeric_coercion_threshold, classification_cardinality_threshold
        )

    # Numeric dtypes -> decide based on cardinality and dtype
    return _infer_from_numeric_column(y_clean, classification_cardinality_threshold)


def _infer_from_object_column(
    y: pd.Series, numeric_threshold: float, cardinality_threshold: int
) -> TaskType:
    """Infer task type from object/string column."""

    # Attempt numeric coercion with common formatting
    y_str = y.astype(str).str.strip()

    # Remove common thousand separators but preserve other punctuation
    y_cleaned = y_str.str.replace(",", "", regex=False)  # Remove commas
    y_cleaned = y_cleaned.str.replace(" ", "", regex=False)  # Remove spaces

    y_numeric = pd.to_numeric(y_cleaned, errors="coerce")
    numeric_ratio = y_numeric.notna().mean() if len(y_numeric) > 0 else 0.0

    if numeric_ratio >= numeric_threshold:
        # Mostly numeric-like strings - check cardinality and uniqueness ratio
        nunique_numeric = y_numeric.nunique(dropna=True)
        uniqueness_ratio = nunique_numeric / len(y) if len(y) > 0 else 1.0

        logger.info(
            "Object column is %.1f%% numeric-coercible with %d unique values (%.1f%% of samples, threshold: %d)",
            100 * numeric_ratio,
            nunique_numeric,
            100 * uniqueness_ratio,
            cardinality_threshold,
        )

        if nunique_numeric <= cardinality_threshold and uniqueness_ratio < 0.5:
            return TaskType.classification
        else:
            return TaskType.regression
    else:
        # Mostly non-numeric strings -> classification
        logger.info(
            "Object column is %.1f%% numeric-coercible -> classification",
            100 * numeric_ratio,
        )
        return TaskType.classification


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
