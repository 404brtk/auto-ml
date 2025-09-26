import pandas as pd

from auto_ml_pipeline.config import CleaningConfig
from auto_ml_pipeline.logging_utils import get_logger
from auto_ml_pipeline.transformers import (
    TimeConverter,
    DateTimeConverter,
    NumericLikeCoercer,
)

logger = get_logger(__name__)


def _ensure_target_exists(df: pd.DataFrame, target: str) -> None:
    """Validate target column exists in DataFrame."""
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


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
    df: pd.DataFrame, target: str, max_missing: int
) -> pd.DataFrame:
    """Remove rows with more than max_missing missing features."""
    if max_missing < 0:
        raise ValueError(f"max_missing must be non-negative, got {max_missing}")

    before = len(df)

    # Count missing values per row, excluding target column
    X = df.drop(columns=[target])
    missing_per_row = X.isnull().sum(axis=1)

    # Keep rows with missing count <= max_missing
    mask = missing_per_row <= max_missing
    df_clean = df[mask].copy()
    removed = before - len(df_clean)

    if removed > 0:
        logger.info(
            "Removed %d rows (%.2f%%) with >%d missing features",
            removed,
            100 * removed / before,
            max_missing,
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

    # Drop duplicates
    if cfg.drop_duplicates:
        result_df = drop_duplicates(result_df)

    # Convert time-only columns
    time_converter = TimeConverter()
    result_df = time_converter.fit_transform(result_df)

    # Convert datetime columns
    datetime_converter = DateTimeConverter()
    result_df = datetime_converter.fit_transform(result_df)

    # Convert numeric-like strings to actual numbers
    numeric_coercer = NumericLikeCoercer(threshold=0.95)
    result_df = numeric_coercer.fit_transform(result_df)

    # Remove rows with missing target
    if cfg.drop_missing_target:
        result_df = remove_missing_target(result_df, target)

    # Remove rows with too many missing features
    if cfg.max_missing_features_per_row is not None:
        result_df = remove_high_missing_rows(
            result_df, target, cfg.max_missing_features_per_row
        )

    # Reset index to avoid potential issues
    result_df = result_df.reset_index(drop=True)

    logger.info("Data cleaning completed. Rows: %d -> %d", len(df), len(result_df))
    return result_df
