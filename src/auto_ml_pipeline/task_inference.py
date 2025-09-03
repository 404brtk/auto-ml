import pandas as pd
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def infer_task(
    df: pd.DataFrame,
    target: str,
    numeric_coercion_threshold: float = 0.95,
    classification_cardinality_threshold: int = 20,
    min_samples_for_inference: int = 10,
) -> TaskType:
    """
    Infer whether a ML task is classification or regression based on target variable.

    Args:
        df: DataFrame containing the target
        target: Name of target column
        numeric_coercion_threshold: Minimum ratio of values that must be numeric-coercible
        classification_cardinality_threshold: Max unique values for classification
        min_samples_for_inference: Minimum samples needed for reliable inference

    Returns:
        TaskType.classification or TaskType.regression

    Examples:
        >>> df = pd.DataFrame({'target': [1, 2, 3, 4, 5]})
        >>> infer_task(df, 'target')  # Many unique integers
        TaskType.regression

        >>> df = pd.DataFrame({'target': ['A', 'B', 'A', 'C']})
        >>> infer_task(df, 'target')  # String categories
        TaskType.classification
    """
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

    if len(y_clean) < min_samples_for_inference:
        logger.warning(
            "Very few samples (%d), inference may be unreliable", len(y_clean)
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
    if pd.api.types.is_categorical_dtype(y):
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
    # This avoids misclassifying strings like "product_v1.2" as numeric
    y_cleaned = y_str.str.replace(",", "", regex=False)  # Remove commas
    y_cleaned = y_cleaned.str.replace(" ", "", regex=False)  # Remove spaces

    y_numeric = pd.to_numeric(y_cleaned, errors="coerce")
    numeric_ratio = y_numeric.notna().mean() if len(y_numeric) > 0 else 0.0

    if numeric_ratio >= numeric_threshold:
        # Mostly numeric-like strings
        nunique_numeric = y_numeric.nunique(dropna=True)
        logger.info(
            "Object column is %.1f%% numeric-coercible with %d unique values",
            100 * numeric_ratio,
            nunique_numeric,
        )

        if nunique_numeric <= cardinality_threshold:
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

    # Integer types with low cardinality -> likely classification
    if dtype_kind in {"i", "u"} and nunique <= cardinality_threshold:
        logger.info("Integer target with %d unique values -> classification", nunique)
        return TaskType.classification

    # Float types or high-cardinality integers -> regression
    if dtype_kind in {"f"} or nunique > cardinality_threshold:
        logger.info(
            "Numeric target (dtype=%s) with %d unique values -> regression",
            y.dtype,
            nunique,
        )
        return TaskType.regression

    # Fallback for other numeric types
    logger.info("Numeric target -> regression (fallback)")
    return TaskType.regression
