import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.config import CleaningConfig
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin


logger = get_logger(__name__)


def _validate_config(cfg: CleaningConfig) -> None:
    """Validate cleaning configuration parameters."""
    if (
        hasattr(cfg, "feature_missing_threshold")
        and cfg.feature_missing_threshold is not None
    ):
        if not (0 <= cfg.feature_missing_threshold <= 1):
            raise ValueError(
                f"feature_missing_threshold must be in [0,1], got {cfg.feature_missing_threshold}"
            )

    if (
        hasattr(cfg, "max_missing_features_per_row")
        and cfg.max_missing_features_per_row is not None
    ):
        if cfg.max_missing_features_per_row < 0:
            raise ValueError(
                f"max_missing_features_per_row must be non-negative, got {cfg.max_missing_features_per_row}"
            )

    if (
        hasattr(cfg, "outlier_iqr_multiplier")
        and cfg.outlier_iqr_multiplier is not None
    ):
        if cfg.outlier_iqr_multiplier <= 0:
            raise ValueError(
                f"outlier_iqr_multiplier must be positive, got {cfg.outlier_iqr_multiplier}"
            )

    if (
        hasattr(cfg, "outlier_zscore_threshold")
        and cfg.outlier_zscore_threshold is not None
    ):
        if cfg.outlier_zscore_threshold <= 0:
            raise ValueError(
                f"outlier_zscore_threshold must be positive, got {cfg.outlier_zscore_threshold}"
            )


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


def _compute_high_missing_cols(X: pd.DataFrame, threshold: float) -> List[str]:
    """Identify columns with high missing value ratio."""
    if not (0 <= threshold <= 1):
        logger.warning(
            "feature_missing_threshold %.3f is out of [0,1]; skipping", threshold
        )
        return []

    if X.empty:
        return []
    miss_ratio = X.isnull().mean()
    high_missing = miss_ratio[miss_ratio > threshold]
    return high_missing.index.tolist()


def drop_high_missing_features(
    df: pd.DataFrame, threshold: float, target: str
) -> pd.DataFrame:
    """Drop features with high missing value ratio."""
    if threshold is None or threshold <= 0:
        return df

    X = df.drop(columns=[target])
    to_drop = _compute_high_missing_cols(X, threshold)

    if to_drop:
        show = ", ".join(to_drop[:10]) + (
            f" ... (+{len(to_drop)-10} more)" if len(to_drop) > 10 else ""
        )
        logger.info(
            "Dropping %d features with missingness > %.2f: %s",
            len(to_drop),
            threshold,
            show,
        )
        X = X.drop(columns=to_drop)

    # Maintain column order
    kept_cols = [c for c in df.columns if c != target and c in X.columns]
    return pd.concat([X[kept_cols], df[[target]]], axis=1)


def _compute_constant_cols(X: pd.DataFrame) -> List[str]:
    """Identify constant columns."""
    if X.empty:
        return []
    constant_cols = []
    for col in X.columns:
        try:
            nunique = X[col].nunique(dropna=True)
            if nunique <= 1:
                constant_cols.append(col)
        except Exception as e:
            logger.warning("Error checking column %s for constancy: %s", col, e)

    return constant_cols


def remove_constant_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove constant features."""
    X = df.drop(columns=[target])
    to_drop = _compute_constant_cols(X)

    if to_drop:
        show = ", ".join(to_drop[:10]) + (
            f" ... (+{len(to_drop)-10} more)" if len(to_drop) > 10 else ""
        )
        logger.info("Removed %d constant features: %s", len(to_drop), show)
        X = X.drop(columns=to_drop)

    kept_cols = [c for c in df.columns if c != target and c in X.columns]
    return pd.concat([X[kept_cols], df[[target]]], axis=1)


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
    """Clean data with basic pre-split operations."""
    _validate_config(cfg)
    _ensure_target_exists(df, target)

    result_df = df.copy()

    # Drop duplicates first (affects all columns)
    if cfg.drop_duplicates:  # Use the renamed config field
        result_df = drop_duplicates(result_df)

    # Only perform row-wise target cleaning pre-split to avoid leakage
    if cfg.drop_missing_target:
        result_df = remove_missing_target(result_df, target)

    # Remove rows with too many missing features (pre-split to avoid leakage)
    if cfg.max_missing_features_per_row is not None:
        result_df = remove_high_missing_rows(
            result_df, target, cfg.max_missing_features_per_row
        )

    # Reset index to avoid potential issues
    result_df = result_df.reset_index(drop=True)

    logger.info("Data cleaning completed. Rows: %d -> %d", len(df), len(result_df))
    return result_df


# sklearn transformers


class FeatureMissingnessDropper(BaseEstimator, TransformerMixin):
    """Drop columns with high missing ratio based on training data."""

    def __init__(self, threshold: float = 0.5):
        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.threshold = float(threshold)
        self.drop_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            self.drop_cols_ = []
            logger.warning("FeatureMissingnessDropper received non-DataFrame; skipping")
            return self

        self.drop_cols_ = _compute_high_missing_cols(X, self.threshold)

        if self.drop_cols_:
            show = ", ".join(self.drop_cols_[:10]) + (
                f" ... (+{len(self.drop_cols_)-10} more)"
                if len(self.drop_cols_) > 10
                else ""
            )
            logger.info(
                "[Dropper] High-missing features to drop (> %.2f): %d -> %s",
                self.threshold,
                len(self.drop_cols_),
                show,
            )
        else:
            logger.info("[Dropper] No high-missing features to drop")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame) and self.drop_cols_:
            # Use intersection to handle cases where columns might not exist
            cols_to_drop = list(set(self.drop_cols_) & set(X.columns))
            if cols_to_drop:
                return X.drop(columns=cols_to_drop)
        return X


class ConstantFeatureDropper(BaseEstimator, TransformerMixin):
    """Drop constant features based on training data."""

    def __init__(self):
        self.drop_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            self.drop_cols_ = []
            logger.warning("ConstantFeatureDropper received non-DataFrame; skipping")
            return self

        self.drop_cols_ = _compute_constant_cols(X)

        if self.drop_cols_:
            show = ", ".join(self.drop_cols_[:10]) + (
                f" ... (+{len(self.drop_cols_)-10} more)"
                if len(self.drop_cols_) > 10
                else ""
            )
            logger.info(
                "[Dropper] Constant features to drop: %d -> %s",
                len(self.drop_cols_),
                show,
            )
        else:
            logger.info("[Dropper] No constant features to drop")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame) and self.drop_cols_:
            cols_to_drop = list(set(self.drop_cols_) & set(X.columns))
            if cols_to_drop:
                return X.drop(columns=cols_to_drop)
        return X


class NumericLikeCoercer(BaseEstimator, TransformerMixin):
    """Numeric coercion with better number format detection."""

    def __init__(
        self,
        threshold: float = 0.95,
        thousand_sep: Optional[str] = None,
        sample_size: int = 10000,
    ):
        if not (0 < threshold <= 1):
            raise ValueError(f"threshold must be in (0,1], got {threshold}")

        self.threshold = float(threshold)
        self.thousand_sep = thousand_sep
        self.sample_size = max(1000, sample_size)  # Minimum sample size
        self.convert_cols_: List[str] = []
        self.conversion_stats_: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _detect_number_format(series: pd.Series) -> Dict[str, Any]:
        """Detect number format with confidence scoring."""
        sample = series.dropna().astype(str).str.strip()
        if len(sample) == 0:
            return {"confidence": 0.0, "format": "unknown"}

        # Count format patterns
        patterns = {
            "comma_decimal": 0,  # 1.234,56
            "dot_decimal": 0,  # 1,234.56
            "simple": 0,  # 1234.56 or 1234
            "scientific": 0,  # 1.23e10
        }

        for val in sample.head(min(1000, len(sample))):
            val = val.replace(" ", "").replace("'", "")
            if not val or val in ["nan", "null", "none"]:
                continue

            # Scientific notation
            if "e" in val.lower():
                patterns["scientific"] += 1
            # Contains both comma and dot
            elif "," in val and "." in val:
                last_comma = val.rfind(",")
                last_dot = val.rfind(".")
                if last_comma > last_dot:
                    patterns["comma_decimal"] += 1
                else:
                    patterns["dot_decimal"] += 1
            # Only comma
            elif "," in val:
                # Heuristic: if 1-2 digits after last comma, likely decimal
                after_comma = len(val) - val.rfind(",") - 1
                if val.count(",") == 1 and 1 <= after_comma <= 2:
                    patterns["comma_decimal"] += 1
                else:
                    patterns["dot_decimal"] += 1  # Grouping comma
            # Only dot or no separators
            else:
                patterns["simple"] += 1

        total = sum(patterns.values())
        if total == 0:
            return {"confidence": 0.0, "format": "unknown"}

        best_format = max(patterns.items(), key=lambda x: x[1])
        confidence = best_format[1] / total

        return {
            "confidence": confidence,
            "format": best_format[0],
            "patterns": patterns,
        }

    def _normalize_number_string(self, s: str, format_info: Dict[str, Any]) -> str:
        """Normalize number string based on detected format."""
        s = s.strip().replace(" ", "").replace("'", "")
        if not s:
            return s

        format_type = format_info.get("format", "simple")

        try:
            if format_type == "scientific":
                return s  # Keep as-is for scientific notation
            elif format_type == "comma_decimal":
                # European: 1.234,56 -> 1234.56
                s = s.replace(".", "")  # Remove grouping dots
                s = s.replace(",", ".")  # Convert decimal comma to dot
            elif format_type == "dot_decimal":
                # US: 1,234.56 -> 1234.56
                s = s.replace(",", "")  # Remove grouping commas
            # For "simple", no changes needed

            return s
        except Exception:
            return s

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            self.convert_cols_ = []
            self.conversion_stats_ = {}
            logger.warning("NumericLikeCoercer received non-DataFrame; skipping")
            return self

        obj_cols = X.select_dtypes(include=["object", "category"]).columns
        convert: List[str] = []

        for col in obj_cols:
            try:
                # Sample large columns for efficiency
                series = X[col]
                if len(series) > self.sample_size:
                    series = series.sample(n=self.sample_size, random_state=42)

                # Detect number format
                format_info = self._detect_number_format(series)

                if self.thousand_sep:
                    # Legacy path with explicit separator
                    cleaned = (
                        series.astype(str).str.replace(" ", "").str.replace("'", "")
                    )
                    cleaned = cleaned.str.replace(self.thousand_sep, "")
                else:
                    # Use format detection
                    cleaned = series.astype(str).apply(
                        lambda x: self._normalize_number_string(x, format_info)
                    )

                # Test conversion
                numeric = pd.to_numeric(cleaned, errors="coerce")
                if len(numeric) == 0:
                    continue

                conversion_rate = float(numeric.notna().mean())

                if conversion_rate >= self.threshold:
                    convert.append(col)
                    self.conversion_stats_[col] = {
                        "conversion_rate": conversion_rate,
                        "format_info": format_info,
                        "sample_size": len(series),
                    }

            except Exception as e:
                logger.warning(
                    "Error analyzing column %s for numeric conversion: %s", col, e
                )
                continue

        self.convert_cols_ = convert

        if convert:
            logger.info(
                "[Coercer] Converting %d columns to numeric (threshold=%.2f): %s",
                len(convert),
                self.threshold,
                ", ".join(convert[:10])
                + (f" ... (+{len(convert)-10} more)" if len(convert) > 10 else ""),
            )
        else:
            logger.info("[Coercer] No numeric-like columns detected")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame) or not self.convert_cols_:
            return X

        X_out = X.copy()

        for col in self.convert_cols_:
            if col not in X_out.columns:
                continue

            try:
                format_info = self.conversion_stats_.get(col, {}).get("format_info", {})

                if self.thousand_sep:
                    cleaned = (
                        X_out[col].astype(str).str.replace(" ", "").str.replace("'", "")
                    )
                    cleaned = cleaned.str.replace(self.thousand_sep, "")
                else:
                    cleaned = (
                        X_out[col]
                        .astype(str)
                        .apply(lambda x: self._normalize_number_string(x, format_info))
                    )

                X_out[col] = pd.to_numeric(cleaned, errors="coerce")

            except Exception as e:
                logger.warning("Error converting column %s: %s", col, e)
                continue

        return X_out


# outlier utilities


def fit_outlier_params(
    df: pd.DataFrame, target: str, cfg: CleaningConfig
) -> Dict[str, Any]:
    """Fit outlier detection parameters with error handling."""
    _validate_config(cfg)

    strategy = (cfg.outlier_strategy or "").lower()
    if strategy in {None, "", "none"}:
        return {"strategy": strategy}

    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    params: Dict[str, Any] = {
        "strategy": strategy,
        "num_cols": num_cols,
        "total_samples": len(X),
    }

    if len(num_cols) == 0:
        logger.info("No numeric columns found for outlier detection")
        return params

    try:
        if strategy == "iqr":
            Q1 = X[num_cols].quantile(0.25)
            Q3 = X[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            k = cfg.outlier_iqr_multiplier

            # Handle zero IQR (constant columns)
            valid_cols = IQR[IQR > 0].index.tolist()
            if not valid_cols:
                logger.warning(
                    "All numeric columns have zero IQR; skipping outlier detection"
                )
                return params

            lower = Q1[valid_cols] - k * IQR[valid_cols]
            upper = Q3[valid_cols] + k * IQR[valid_cols]
            params.update({"lower": lower, "upper": upper, "valid_cols": valid_cols})

        elif strategy == "zscore":
            mean = X[num_cols].mean()
            std = X[num_cols].std(ddof=1)  # Use sample std

            # Filter out columns with zero/near-zero std
            valid_cols = std[std > 1e-10].index.tolist()
            if not valid_cols:
                logger.warning(
                    "All numeric columns have zero std; skipping outlier detection"
                )
                return params

            params.update(
                {
                    "mean": mean[valid_cols],
                    "std": std[valid_cols],
                    "thr": cfg.outlier_zscore_threshold,
                    "valid_cols": valid_cols,
                }
            )

    except Exception as e:
        logger.error(
            "Error fitting outlier parameters for strategy '%s': %s", strategy, e
        )
        return {"strategy": "none"}  # Fallback to no outlier detection

    logger.info(
        "Fitted outlier detection: strategy=%s, valid_cols=%d",
        strategy,
        len(params.get("valid_cols", num_cols)),
    )
    return params


def apply_outliers(
    df: pd.DataFrame,
    target: str,
    cfg: CleaningConfig,
    params: Dict[str, Any],
    scope: str,
) -> pd.DataFrame:
    """Apply outlier treatment with error handling."""
    strategy = params.get("strategy")
    if not strategy or strategy == "none":
        return df

    valid_cols = params.get("valid_cols", [])
    if not valid_cols:
        return df

    # Ensure indices are aligned
    df_work = df.copy().reset_index(drop=True)
    X = df_work.drop(columns=[target])
    y = df_work[target]

    method = (cfg.outlier_method or "clip").lower()

    try:
        if strategy == "iqr" and "lower" in params and "upper" in params:
            lower = params["lower"]
            upper = params["upper"]

            if method == "clip":
                X[valid_cols] = X[valid_cols].clip(lower=lower, upper=upper, axis=1)
                outliers_handled = (
                    (
                        (df.drop(columns=[target])[valid_cols] < lower)
                        | (df.drop(columns=[target])[valid_cols] > upper)
                    )
                    .sum()
                    .sum()
                )
                logger.info(
                    "Clipped %d outliers via IQR on %s data", outliers_handled, scope
                )
            else:  # remove
                mask = ~(
                    ((X[valid_cols] < lower) | (X[valid_cols] > upper)).any(axis=1)
                )
                before = len(X)
                X = X[mask].reset_index(drop=True)
                y = y[mask].reset_index(drop=True)
                outliers_handled = before - len(X)
                logger.info(
                    "Removed %d outliers via IQR on %s data", outliers_handled, scope
                )

        elif strategy == "zscore" and "mean" in params and "std" in params:
            mean = params["mean"]
            std = params["std"]
            thr = params.get("thr", 3.0)

            z_scores = (X[valid_cols] - mean) / std

            if method == "clip":
                clipped = mean + thr * np.sign(z_scores) * std
                X[valid_cols] = X[valid_cols].where(z_scores.abs() <= thr, clipped)
                outliers_handled = (z_scores.abs() > thr).sum().sum()
                logger.info(
                    "Clipped %d outliers via Z-score on %s data (thr=%.2f)",
                    outliers_handled,
                    scope,
                    thr,
                )
            else:
                mask = (z_scores.abs() <= thr).all(axis=1)
                before = len(X)
                X = X[mask].reset_index(drop=True)
                y = y[mask].reset_index(drop=True)
                outliers_handled = before - len(X)
                logger.info(
                    "Removed %d outliers via Z-score on %s data (thr=%.2f)",
                    outliers_handled,
                    scope,
                    thr,
                )

    except Exception as e:
        logger.error("Error applying outlier treatment (%s): %s", strategy, e)
        return df

    # Reconstruct DataFrame maintaining column order
    result = pd.concat([X, y], axis=1)
    return result
