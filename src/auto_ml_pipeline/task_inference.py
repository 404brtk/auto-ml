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
    # This avoids misclassifying strings like "product_v1.2" as numeric
    y_cleaned = y_str.str.replace(",", "", regex=False)  # Remove commas
    y_cleaned = y_cleaned.str.replace(" ", "", regex=False)  # Remove spaces

    y_numeric = pd.to_numeric(y_cleaned, errors="coerce")
    numeric_ratio = y_numeric.notna().mean() if len(y_numeric) > 0 else 0.0

    if numeric_ratio >= numeric_threshold:
        # Mostly numeric-like strings - use consistent thresholds
        nunique_numeric = y_numeric.nunique(dropna=True)

        # Use same dynamic threshold as feature_engineering.py
        dynamic_threshold = min(100, len(y) * 0.1)
        effective_threshold = max(cardinality_threshold, dynamic_threshold)

        logger.info(
            "Object column is %.1f%% numeric-coercible with %d unique values (threshold: %.1f)",
            100 * numeric_ratio,
            nunique_numeric,
            effective_threshold,
        )

        if (
            nunique_numeric <= cardinality_threshold
            and nunique_numeric <= dynamic_threshold
        ):
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

    # Use same heuristic as feature_engineering.py for consistency
    # Dynamic threshold based on sample size, with minimum of 100
    dynamic_threshold = min(100, len(y) * 0.1)
    effective_threshold = max(cardinality_threshold, dynamic_threshold)

    # Float types -> regression (continuous by nature)
    if dtype_kind in {"f"}:
        logger.info(
            "Float target (dtype=%s) with %d unique values -> regression",
            y.dtype,
            nunique,
        )
        return TaskType.regression

    # Integer types: use dynamic threshold for better detection
    if dtype_kind in {"i", "u"}:
        if nunique <= cardinality_threshold and nunique <= dynamic_threshold:
            logger.info(
                "Integer target with %d unique values -> classification", nunique
            )
            return TaskType.classification
        else:
            logger.info(
                "Integer target with %d unique values (threshold: %.1f) -> regression",
                nunique,
                effective_threshold,
            )
            return TaskType.regression

    # Fallback for other numeric types
    logger.info("Numeric target -> regression (fallback)")
    return TaskType.regression
