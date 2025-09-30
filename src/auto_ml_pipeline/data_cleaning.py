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


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # Store original names for logging
    original_names = df_clean.columns.tolist()

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

    df_clean.columns = final_names

    # Log changes
    changes = [
        (old, new) for old, new in zip(original_names, final_names) if old != new
    ]
    if changes:
        logger.info(
            "Standardized %d column names (e.g., '%s' -> '%s')",
            len(changes),
            changes[0][0],
            changes[0][1],
        )

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


# TODO: when tolerance is low and all features are removed, it's being skipped
# maybe leave some of featueres with the highest variance
def remove_constant_features(
    df: pd.DataFrame, target: str, constant_tolerance: float = 1.0
) -> pd.DataFrame:

    # Separate target from features
    X = df.drop(columns=[target])

    # Use feature-engine's DropConstantFeatures
    dropper = DropConstantFeatures(tol=constant_tolerance, missing_values="ignore")

    try:
        X_clean = dropper.fit_transform(X)

        removed_features = dropper.features_to_drop_

        # Recombine with target
        df_clean = pd.concat([X_clean, df[[target]]], axis=1)

        if removed_features:
            logger.info(
                "Removed %d constant/quasi-constant features (tol=%.2f): %s",
                len(removed_features),
                constant_tolerance,
                ", ".join(removed_features[:5])
                + (
                    f" ... (+{len(removed_features)-5} more)"
                    if len(removed_features) > 5
                    else ""
                ),
            )
        else:
            logger.info("No constant features detected")

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


def clean_data(df: pd.DataFrame, target: str, cfg: CleaningConfig) -> pd.DataFrame:
    """Clean data with pre-split operations."""
    _ensure_target_exists(df, target)

    result_df = df.copy()
    initial_shape = result_df.shape

    # 1. Standardize column names (avoid issues with special characters)
    result_df = standardize_column_names(result_df)
    # Update target name if it was changed
    target_lower = str(target).lower()
    target = re.sub(r"[^a-z0-9_]", "_", target_lower).strip("_")

    # 2. Clean special null value representations
    special_null_values = [
        "?",
        "N/A",
        "n/a",
        "NA",
        "null",
        "NULL",
        "None",
        "none",
        "nan",
        "NaN",
        "NAN",
        "undefined",
        "missing",
        "MISSING",
        "-",
        "--",
        "---",
        "",
        " ",
    ]
    result_df = clean_special_null_values(result_df, special_null_values)

    # 3. Drop duplicates
    if cfg.drop_duplicates:
        result_df = drop_duplicates(result_df)

    # 4. Convert time-only columns
    time_converter = TimeConverter()
    result_df = time_converter.fit_transform(result_df)

    # 5. Convert datetime columns
    datetime_converter = DateTimeConverter()
    result_df = datetime_converter.fit_transform(result_df)

    # 6. Convert numeric-like strings to actual numbers
    numeric_coercer = NumericLikeCoercer(threshold=0.95)
    result_df = numeric_coercer.fit_transform(result_df)

    # 7. Handle inf values in numeric columns
    result_df = handle_inf_values(result_df)

    # 8. Remove rows with missing target
    if cfg.drop_missing_target:
        result_df = remove_missing_target(result_df, target)

    # 9. Remove rows with too many missing features
    if cfg.max_missing_row_ratio is not None:
        result_df = remove_high_missing_rows(
            result_df, target, cfg.max_missing_row_ratio
        )

    # 10. Remove constant/quasi-constant features
    if cfg.remove_constant_features:
        result_df = remove_constant_features(result_df, target, cfg.constant_tolerance)

    # Reset index to avoid potential issues
    result_df = result_df.reset_index(drop=True)

    logger.info(
        "Data cleaning completed. Shape: (%d, %d) -> (%d, %d)",
        initial_shape[0],
        initial_shape[1],
        result_df.shape[0],
        result_df.shape[1],
    )
    return result_df
