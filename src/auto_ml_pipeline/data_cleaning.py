import pandas as pd
import numpy as np
import re

from auto_ml_pipeline.config import CleaningConfig
from auto_ml_pipeline.logging_utils import get_logger
from auto_ml_pipeline.transformers import (
    TimeConverter,
    DateTimeConverter,
    NumericLikeCoercer,
)
from feature_engine.selection import DropConstantFeatures

logger = get_logger(__name__)


def _ensure_target_exists(df: pd.DataFrame, target: str) -> None:
    """Validate target column exists in DataFrame."""
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def handle_mixed_types(df: pd.DataFrame, strategy: str = "coerce") -> pd.DataFrame:
    df_clean = df.copy()
    mixed_type_cols = []

    for col in df.columns:
        if df[col].isna().all():
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        # Get unique types in the column
        types = col_data.apply(type).unique()

        # Check if column has mixed numeric and non-numeric types
        if len(types) > 1:
            has_numeric = any(
                isinstance(val, (int, float, np.number)) for val in col_data.head(100)
            )
            has_string = any(isinstance(val, str) for val in col_data.head(100))

            if has_numeric and has_string:
                mixed_type_cols.append(col)
                type_distribution = col_data.apply(type).value_counts()
                logger.warning(
                    "Column '%s' has mixed types: %s",
                    col,
                    {t.__name__: count for t, count in type_distribution.items()},
                )

    if not mixed_type_cols:
        return df_clean

    logger.warning(
        "Found %d columns with mixed data types: %s",
        len(mixed_type_cols),
        ", ".join(mixed_type_cols[:5])
        + (
            f" ... (+{len(mixed_type_cols) - 5} more)"
            if len(mixed_type_cols) > 5
            else ""
        ),
    )

    # Apply strategy
    if strategy == "coerce":
        for col in mixed_type_cols:
            # Preserve NaN values when converting to string
            mask = df_clean[col].notna()
            df_clean.loc[mask, col] = df_clean.loc[mask, col].astype(str)
        logger.info("Coerced %d mixed-type columns to string", len(mixed_type_cols))
    elif strategy == "drop":
        df_clean = df_clean.drop(columns=mixed_type_cols)
        logger.info("Dropped %d mixed-type columns", len(mixed_type_cols))

    return df_clean


def standardize_column_names(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    df_clean = df.copy()

    # Store original names for logging
    original_names = df_clean.columns.tolist()

    # Verify target exists before processing
    if target not in original_names:
        raise ValueError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {original_names}"
        )

    # Standardize names
    new_names = []
    for col in original_names:
        # Convert to string and lowercase
        name = str(col).lower()
        # Replace spaces and special chars with underscore
        name = re.sub(r"[^a-z0-9_]", "_", name)
        # Remove leading/trailing underscores
        name = name.strip("_")
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = "col_" + name
        # Handle empty names
        if not name:
            name = "unnamed"
        new_names.append(name)

    # Handle duplicates by appending suffix
    seen: dict[str, int] = {}
    final_names = []
    for name in new_names:
        if name in seen:
            seen[name] += 1
            final_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            final_names.append(name)

    name_mapping = {
        original: final for original, final in zip(original_names, final_names)
    }

    df_clean.columns = final_names

    changes = [
        (old, new) for old, new in zip(original_names, final_names) if old != new
    ]
    if changes:
        logger.info("Standardized %d column names", len(changes))

    # Track target column transformation
    target_idx = original_names.index(target)
    standardized_target = final_names[target_idx]

    if target != standardized_target:
        logger.info(
            "Target column standardized: '%s' -> '%s'",
            target,
            standardized_target,
        )

    return df_clean, standardized_target, name_mapping


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading and trailing whitespace from string columns."""
    df_clean = df.copy()

    # Get object/string columns
    obj_cols = df_clean.select_dtypes(include=["object"]).columns

    if len(obj_cols) == 0:
        return df_clean

    trimmed_cols = 0

    for col in obj_cols:
        try:
            # Only trim if column contains strings
            if df_clean[col].dtype == object:
                # Store original for comparison
                original = df_clean[col].copy()
                # Trim whitespace, preserving NaN
                df_clean[col] = df_clean[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

                # Check if anything changed
                changed = (original != df_clean[col]).sum()
                if changed > 0:
                    trimmed_cols += 1
        except Exception as e:
            logger.warning("Error trimming whitespace in column '%s': %s", col, e)
            continue

    if trimmed_cols > 0:
        logger.info("Trimmed whitespace from %d string columns", trimmed_cols)

    return df_clean


def clean_special_null_values(df: pd.DataFrame, special_values: list) -> pd.DataFrame:
    df_clean = df.copy()

    # Get object/string columns
    obj_cols = df_clean.select_dtypes(include=["object", "category"]).columns

    if len(obj_cols) == 0:
        return df_clean

    total_replaced = 0

    for col in obj_cols:
        # Convert to string and strip whitespace for comparison
        col_values = df_clean[col].astype(str).str.strip().str.lower()

        # Create mask for special values
        mask = col_values.isin([str(v).lower() for v in special_values])

        replaced_count = mask.sum()
        if replaced_count > 0:
            df_clean.loc[mask, col] = np.nan
            total_replaced += replaced_count

    if total_replaced > 0:
        logger.info(
            "Replaced %d special null values with NaN across %d columns",
            total_replaced,
            len(obj_cols),
        )

    return df_clean


def handle_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # Get numeric columns
    num_cols = df_clean.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        return df_clean

    total_inf = 0

    for col in num_cols:
        inf_mask = np.isinf(df_clean[col])
        inf_count = inf_mask.sum()

        if inf_count > 0:
            df_clean.loc[inf_mask, col] = np.nan
            total_inf += inf_count

    if total_inf > 0:
        logger.info(
            "Replaced %d inf values with NaN across numeric columns",
            total_inf,
        )

    return df_clean


def remove_constant_features(
    df: pd.DataFrame,
    target: str,
    constant_tolerance: float = 1.0,
    min_features_to_keep: int = 1,
) -> pd.DataFrame:
    X = df.drop(columns=[target])

    if X.shape[1] == 0:
        logger.info("No features to analyze for constant feature removal.")
        return df

    dropper = DropConstantFeatures(tol=constant_tolerance, missing_values="ignore")

    try:
        dropper.fit(X)
        features_to_drop = dropper.features_to_drop_.copy()

        # if all features are constant (or quasi-constant) and min_features_to_keep=0, skip removal
        if min_features_to_keep == 0 and len(features_to_drop) == X.shape[1]:
            logger.info(
                "All features identified as constant and min_features_to_keep is 0. "
                "Skipping constant feature removal."
            )
            return df

        # calculate how many features would remain
        n_features_remaining = X.shape[1] - len(features_to_drop)

        # rescue features if we'd drop too many
        if n_features_remaining < min_features_to_keep:
            n_to_rescue = min_features_to_keep - n_features_remaining

            logger.warning(
                "%d features identified as constant (tol=%.2f), leaving only %d. "
                "Rescuing %d least-constant features to maintain minimum of %d.",
                len(features_to_drop),
                constant_tolerance,
                n_features_remaining,
                n_to_rescue,
                min_features_to_keep,
            )

            # sort dropped features by performance (variance) and rescue the least constant
            performance_of_dropped = {
                f: dropper.feature_performance_[f]
                for f in features_to_drop
                if f in dropper.feature_performance_
            }

            # keep features with highest variance (lowest performance score = more variance)
            features_to_rescue = sorted(
                performance_of_dropped.items(), key=lambda item: item[1]
            )[:n_to_rescue]

            features_to_rescue = [f[0] for f in features_to_rescue]
            features_to_drop = [
                f for f in features_to_drop if f not in features_to_rescue
            ]

        if features_to_drop:
            X_clean = X.drop(columns=features_to_drop)
            logger.info(
                "Removed %d constant/quasi-constant features (tol=%.2f): %s",
                len(features_to_drop),
                constant_tolerance,
                ", ".join(features_to_drop[:5])
                + (
                    f" ... (+{len(features_to_drop) - 5} more)"
                    if len(features_to_drop) > 5
                    else ""
                ),
            )
        else:
            X_clean = X
            logger.info("No constant features detected to remove.")

        df_clean = pd.concat([X_clean, df[[target]]], axis=1)
        return df_clean

    except Exception as e:
        logger.warning("Error removing constant features: %s. Skipping.", e)
        return df


def remove_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove rows with missing target values."""
    before = len(df)
    mask = df[target].notna()
    df_clean = df[mask].copy()
    removed = before - len(df_clean)

    if removed > 0:
        logger.info(
            "Removed %d rows (%.2f%%) with missing target",
            removed,
            100 * removed / before,
        )

    return df_clean


def remove_high_missing_rows(
    df: pd.DataFrame, target: str, max_missing_ratio: float
) -> pd.DataFrame:
    """Remove rows with missing ratio above max_missing_ratio."""
    if not (0 <= max_missing_ratio <= 1):
        raise ValueError(f"max_missing_ratio must be in [0,1], got {max_missing_ratio}")

    before = len(df)

    # Separate target from features
    X = df.drop(columns=[target])

    # Handle edge case: no features
    if X.shape[1] == 0:
        logger.warning(
            "No features found (only target column). Returning original DataFrame."
        )
        return df.copy()

    # Count missing values per row, excluding target column
    missing_per_row = X.isnull().sum(axis=1)
    missing_ratios = missing_per_row / X.shape[1]

    # Keep rows with missing ratio <= threshold
    mask = missing_ratios <= max_missing_ratio
    df_clean = df[mask].copy()
    removed = before - len(df_clean)

    if removed > 0:
        logger.info(
            "Removed %d rows (%.2f%%) with >%.1f%% missing features",
            removed,
            100 * removed / before,
            100 * max_missing_ratio,
        )

    return df_clean


def remove_high_missing_features(
    df: pd.DataFrame, target: str, max_missing_ratio: float
) -> pd.DataFrame:
    """Remove features with missing ratio above max_missing_ratio."""
    if not (0 <= max_missing_ratio <= 1):
        raise ValueError(f"max_missing_ratio must be in [0,1], got {max_missing_ratio}")

    df_clean = df.copy()

    missing_ratios = df_clean.isnull().sum() / len(df_clean)

    features_to_remove = missing_ratios[
        (missing_ratios > max_missing_ratio) & (missing_ratios.index != target)
    ].index.tolist()

    if features_to_remove:
        df_clean = df_clean.drop(columns=features_to_remove)
        logger.info(
            "Removed %d features with >%.1f%% missing values: %s",
            len(features_to_remove),
            100 * max_missing_ratio,
            ", ".join(features_to_remove[:5])
            + (
                f" ... (+{len(features_to_remove) - 5} more)"
                if len(features_to_remove) > 5
                else ""
            ),
        )

    return df_clean


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows."""
    before = len(df)
    df_clean = df.drop_duplicates(keep="first")
    removed = before - len(df_clean)

    if removed > 0:
        logger.info(
            "Dropped %d duplicate rows (%.2f%%)", removed, 100 * removed / before
        )

    return df_clean


def remove_id_columns(
    df: pd.DataFrame, target: str, threshold: float = 0.95
) -> pd.DataFrame:
    """Remove columns that are likely ID columns based on uniqueness ratio."""
    df_clean = df.copy()
    id_cols_to_remove = []

    for col in df.columns:
        if col == target:
            continue

        # Check both numeric and categorical columns for ID-like behavior
        nunique = df_clean[col].nunique()
        total = len(df_clean)
        uniqueness_ratio = nunique / total if total > 0 else 0

        # Check if column is essentially unique (potential ID)
        if uniqueness_ratio > threshold:
            id_cols_to_remove.append(col)
            logger.info(
                "Removing ID column '%s' (%d unique / %d total = %.1f%%)",
                col,
                nunique,
                total,
                100 * uniqueness_ratio,
            )

    if id_cols_to_remove:
        df_clean = df_clean.drop(columns=id_cols_to_remove)
        logger.info(
            "Removed %d ID columns with >%.0f%% unique values",
            len(id_cols_to_remove),
            100 * threshold,
        )

    return df_clean


def validate_dataset_size(
    df: pd.DataFrame, initial_shape: tuple, min_rows: int = 10, min_cols: int = 1
) -> None:
    current_rows, current_cols = df.shape
    initial_rows, initial_cols = initial_shape

    # Check absolute minimums
    if current_rows < min_rows:
        raise ValueError(
            f"Dataset has only {current_rows} rows after cleaning (minimum: {min_rows}). "
            f"This is insufficient for reliable model training. "
            f"Consider relaxing cleaning parameters or lowering min_rows parameter."
        )

    if current_cols < min_cols:
        raise ValueError(
            f"Dataset has only {current_cols} feature columns after cleaning (minimum: {min_cols}). "
            f"This is insufficient for reliable model training. "
            f"Consider relaxing cleaning parameters or lowering min_cols parameter."
        )

    # Check for excessive data loss
    row_loss_pct = (
        100 * (initial_rows - current_rows) / initial_rows if initial_rows > 0 else 0
    )
    col_loss_pct = (
        100 * (initial_cols - current_cols) / initial_cols if initial_cols > 0 else 0
    )

    if row_loss_pct > 50:
        logger.warning(
            "Lost %.1f%% of rows during cleaning (%d -> %d). "
            "Consider relaxing cleaning parameters if this is excessive.",
            row_loss_pct,
            initial_rows,
            current_rows,
        )

    if col_loss_pct > 50:
        logger.warning(
            "Lost %.1f%% of columns during cleaning (%d -> %d). "
            "This may indicate overly aggressive feature removal.",
            col_loss_pct,
            initial_cols,
            current_cols,
        )


def clean_data(
    df: pd.DataFrame, target: str, cfg: CleaningConfig
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    """Clean data with pre-split operations."""
    _ensure_target_exists(df, target)

    result_df = df.copy()
    initial_shape = result_df.shape

    logger.info(
        "Starting data cleaning: %d rows, %d columns",
        initial_shape[0],
        initial_shape[1],
    )

    # 1. Standardize column names (avoid issues with special characters)
    result_df, target, name_mapping = standardize_column_names(result_df, target)

    # 2. Handle mixed data types
    result_df = handle_mixed_types(result_df, strategy=cfg.handle_mixed_types)

    # 3. Trim whitespace from string columns
    result_df = trim_whitespace(result_df)

    # 4. Clean special null value representations
    result_df = clean_special_null_values(result_df, cfg.special_null_values)

    # 5. Drop duplicates
    if cfg.drop_duplicates:
        result_df = drop_duplicates(result_df)

    # 6. Convert time-only columns
    time_converter = TimeConverter()
    result_df = time_converter.fit_transform(result_df)

    # 7. Convert datetime columns
    datetime_converter = DateTimeConverter()
    result_df = datetime_converter.fit_transform(result_df)

    # 8. Convert numeric-like strings to actual numbers
    numeric_coercer = NumericLikeCoercer(
        threshold=cfg.numeric_coercion_threshold, target_col=target
    )
    result_df = numeric_coercer.fit_transform(result_df)

    # 9. Handle inf values in numeric columns
    result_df = handle_inf_values(result_df)

    # 10. Remove rows with missing target
    result_df = remove_missing_target(result_df, target)

    # 11. Remove rows with too many missing features
    if cfg.max_missing_row_ratio is not None:
        result_df = remove_high_missing_rows(
            result_df, target, cfg.max_missing_row_ratio
        )

    # 12. Remove features with too many missing values
    if cfg.max_missing_feature_ratio is not None:
        result_df = remove_high_missing_features(
            result_df, target, cfg.max_missing_feature_ratio
        )

    # 13. Remove constant/quasi-constant features
    if cfg.remove_constant_features:
        result_df = remove_constant_features(
            result_df,
            target,
            cfg.constant_tolerance,
            cfg.min_features_after_constant_removal,
        )

    # 14. Remove ID columns
    if cfg.remove_id_columns:
        result_df = remove_id_columns(result_df, target, cfg.id_column_threshold)

    # 15. Validate dataset size after cleaning (raises error if insufficient)
    validate_dataset_size(
        result_df,
        initial_shape,
        min_rows=cfg.min_rows_after_cleaning,
        min_cols=cfg.min_cols_after_cleaning,
    )

    # Reset index to avoid potential issues
    result_df = result_df.reset_index(drop=True)

    logger.info(
        "Data cleaning completed. Shape: (%d, %d) -> (%d, %d)",
        initial_shape[0],
        initial_shape[1],
        result_df.shape[0],
        result_df.shape[1],
    )
    return result_df, target, name_mapping
