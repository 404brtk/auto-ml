import pandas as pd
from auto_ml_pipeline.config import TaskType
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def infer_task(
    df: pd.DataFrame,
    target: str,
    uniqueness_ratio_threshold: float = 0.05,
    max_categories_absolute: int = 20,
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
    nunique = y_clean.nunique()
    if nunique <= 1:
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

    # Object dtypes -> classification
    if y.dtype.kind == "O":
        logger.info(
            "Object target detected (dtype=%s) with %d unique values -> classification",
            y.dtype,
            nunique,
        )
        return TaskType.classification

    # Numeric dtypes -> decide based on cardinality and dtype
    return _infer_from_numeric_column(
        y_clean, uniqueness_ratio_threshold, max_categories_absolute
    )


def _infer_from_numeric_column(
    y: pd.Series,
    uniqueness_ratio_threshold: float = 0.05,
    max_categories_absolute: int = 20,
) -> TaskType:
    """Infer task type from numeric column."""
    n_samples = len(y)
    nunique_full = y.nunique()
    dtype_kind = y.dtype.kind

    # Binary classification
    if nunique_full == 2:
        logger.info("Binary target with 2 unique values -> classification")
        return TaskType.classification

    # Small multiclass (absolute cardinality check)
    # for low cardinality, always prefer classification for integers
    if nunique_full <= max_categories_absolute:
        if dtype_kind in {"i", "u"}:
            logger.info(
                "Integer target with %d unique values (<= %d threshold) -> classification",
                nunique_full,
                max_categories_absolute,
            )
            return TaskType.classification

        # Float types with low cardinality - check if integer-like
        if dtype_kind in {"f"}:
            # modulo check: (value % 1 == 0) means it's an integer
            if (y % 1 == 0).all():
                logger.info(
                    "Float target with %d integer-like values (<= %d threshold) -> classification",
                    nunique_full,
                    max_categories_absolute,
                )
                return TaskType.classification
            else:
                # Float with few unique values but not integers -> likely regression
                logger.info(
                    "Float target with %d unique continuous values -> regression",
                    nunique_full,
                )
                return TaskType.regression

    # for larger cardinality, use sampling-based uniqueness ratio
    # use at least 1000 samples or 10% of data
    sample_size = min(n_samples, max(1000, int(0.1 * n_samples)))

    if sample_size < n_samples:
        # Sample for large datasets
        y_sample = y.sample(sample_size, random_state=42)
        nunique_sample = y_sample.nunique()
        uniqueness_ratio = nunique_sample / sample_size
    else:
        # Use full data for small datasets
        nunique_sample = nunique_full
        uniqueness_ratio = nunique_full / n_samples

    # Float types - default to regression
    if dtype_kind in {"f"}:
        logger.info(
            "Float target (dtype=%s) with %d unique values (%.1f%% uniqueness) -> regression",
            y.dtype,
            nunique_sample,
            100 * uniqueness_ratio,
        )
        return TaskType.regression

    # Integer types - use uniqueness ratio
    if dtype_kind in {"i", "u"}:
        if uniqueness_ratio < uniqueness_ratio_threshold:
            logger.info(
                "Integer target with %d unique values (%.1f%% uniqueness < %.1f%% threshold) -> classification",
                nunique_sample,
                100 * uniqueness_ratio,
                100 * uniqueness_ratio_threshold,
            )
            return TaskType.classification
        else:
            logger.info(
                "Integer target with %d unique values (%.1f%% uniqueness >= %.1f%% threshold) -> regression",
                nunique_sample,
                100 * uniqueness_ratio,
                100 * uniqueness_ratio_threshold,
            )
            return TaskType.regression

    # Fallback for any other numeric types
    logger.info(
        "Numeric target (dtype=%s) with %d unique values -> regression (fallback)",
        y.dtype,
        nunique_sample,
    )
    return TaskType.regression
