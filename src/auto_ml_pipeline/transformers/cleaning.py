import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin

logger = get_logger(__name__)


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
    """Numeric coercion with number format detection."""

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


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Detect and handle outliers in numeric features."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        method: str = "clip",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ):
        self.strategy = strategy.lower() if strategy else None
        self.method = (method or "clip").lower()
        self.iqr_multiplier = float(iqr_multiplier)
        self.zscore_threshold = float(zscore_threshold)

        # Fitted parameters
        self.num_cols_: List[str] = []
        self.valid_cols_: List[str] = []
        self.outlier_params_: Dict[str, Any] = {}

        # Validation
        if self.iqr_multiplier <= 0:
            raise ValueError(
                f"iqr_multiplier must be positive, got {self.iqr_multiplier}"
            )
        if self.zscore_threshold <= 0:
            raise ValueError(
                f"zscore_threshold must be positive, got {self.zscore_threshold}"
            )
        if self.method not in ["clip", "remove"]:
            raise ValueError(f"method must be 'clip' or 'remove', got {self.method}")
        if self.strategy not in [None, "iqr", "zscore"]:
            raise ValueError(
                f"strategy must be None, 'iqr', or 'zscore', got {self.strategy}"
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit outlier detection parameters."""
        if not isinstance(X, pd.DataFrame):
            logger.warning("OutlierTransformer received non-DataFrame; skipping")
            return self

        if self.strategy is None:
            logger.info("[OutlierTransformer] No outlier detection requested")
            return self

        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(self.num_cols_) == 0:
            logger.info(
                "[OutlierTransformer] No numeric columns found for outlier detection"
            )
            return self

        try:
            if self.strategy == "iqr":
                Q1 = X[self.num_cols_].quantile(0.25)
                Q3 = X[self.num_cols_].quantile(0.75)
                IQR = Q3 - Q1

                # Handle zero IQR (constant columns)
                self.valid_cols_ = IQR[IQR > 0].index.tolist()
                if not self.valid_cols_:
                    logger.warning(
                        "[OutlierTransformer] All numeric columns have zero IQR; skipping outlier detection"
                    )
                    return self

                lower = (
                    Q1[self.valid_cols_] - self.iqr_multiplier * IQR[self.valid_cols_]
                )
                upper = (
                    Q3[self.valid_cols_] + self.iqr_multiplier * IQR[self.valid_cols_]
                )
                self.outlier_params_ = {"lower": lower, "upper": upper}

            elif self.strategy == "zscore":
                mean = X[self.num_cols_].mean()
                std = X[self.num_cols_].std(ddof=1)  # Use sample std

                # Filter out columns with zero/near-zero std
                self.valid_cols_ = std[std > 1e-10].index.tolist()
                if not self.valid_cols_:
                    logger.warning(
                        "[OutlierTransformer] All numeric columns have zero std; skipping outlier detection"
                    )
                    return self

                self.outlier_params_ = {
                    "mean": mean[self.valid_cols_],
                    "std": std[self.valid_cols_],
                }

        except Exception as e:
            logger.error(
                "[OutlierTransformer] Error fitting outlier parameters for strategy '%s': %s",
                self.strategy,
                e,
            )
            self.strategy = "none"  # Fallback to no outlier detection
            return self

        if self.valid_cols_:
            logger.info(
                "[OutlierTransformer] Fitted outlier detection: strategy=%s, method=%s, valid_cols=%d",
                self.strategy,
                self.method,
                len(self.valid_cols_),
            )
        else:
            logger.info("[OutlierTransformer] No outlier detection applied")
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply outlier treatment."""
        if not isinstance(X, pd.DataFrame):
            return X

        if self.strategy is None or not self.valid_cols_:
            return X

        X_work = X.copy()
        original_shape = X_work.shape

        try:
            if (
                self.strategy == "iqr"
                and "lower" in self.outlier_params_
                and "upper" in self.outlier_params_
            ):
                lower = self.outlier_params_["lower"]
                upper = self.outlier_params_["upper"]

                if self.method == "clip":
                    X_work[self.valid_cols_] = X_work[self.valid_cols_].clip(
                        lower=lower, upper=upper, axis=1
                    )
                    outliers_handled = (
                        ((X[self.valid_cols_] < lower) | (X[self.valid_cols_] > upper))
                        .sum()
                        .sum()
                    )
                    if outliers_handled > 0:
                        logger.info(
                            "[OutlierTransformer] Clipped %d outliers via IQR",
                            outliers_handled,
                        )
                else:  # remove
                    mask = ~(
                        (
                            (X_work[self.valid_cols_] < lower)
                            | (X_work[self.valid_cols_] > upper)
                        ).any(axis=1)
                    )
                    before = len(X_work)
                    X_work = X_work[mask]
                    removed = before - len(X_work)
                    if removed > 0:
                        logger.info(
                            "[OutlierTransformer] Removed %d outlier rows (%.2f%%) via IQR",
                            removed,
                            100 * removed / before,
                        )

            elif (
                self.strategy == "zscore"
                and "mean" in self.outlier_params_
                and "std" in self.outlier_params_
            ):
                mean = self.outlier_params_["mean"]
                std = self.outlier_params_["std"]
                z_scores = np.abs((X_work[self.valid_cols_] - mean) / std)

                if self.method == "clip":
                    # Clip based on z-score threshold
                    outlier_mask = z_scores > self.zscore_threshold
                    for col in self.valid_cols_:
                        col_outliers = outlier_mask[col]
                        if col_outliers.any():
                            # Calculate clip bounds
                            lower_bound = mean[col] - self.zscore_threshold * std[col]
                            upper_bound = mean[col] + self.zscore_threshold * std[col]
                            X_work[col] = X_work[col].clip(
                                lower=lower_bound, upper=upper_bound
                            )

                    outliers_handled = outlier_mask.sum().sum()
                    if outliers_handled > 0:
                        logger.info(
                            "[OutlierTransformer] Clipped %d outliers via Z-score",
                            outliers_handled,
                        )
                else:  # remove
                    mask = ~(z_scores > self.zscore_threshold).any(axis=1)
                    before = len(X_work)
                    X_work = X_work[mask]
                    removed = before - len(X_work)
                    if removed > 0:
                        logger.info(
                            "[OutlierTransformer] Removed %d outlier rows (%.2f%%) via Z-score",
                            removed,
                            100 * removed / before,
                        )

        except Exception as e:
            logger.error(
                "[OutlierTransformer] Error applying outlier treatment for strategy '%s': %s",
                self.strategy,
                e,
            )
            return X

        # Log shape change
        if X_work.shape != original_shape:
            logger.info(
                "[OutlierTransformer] Shape changed: %s -> %s",
                original_shape,
                X_work.shape,
            )
        else:
            logger.info(
                "[OutlierTransformer] Applied outlier treatment: %s (shape unchanged)",
                original_shape,
            )

        return X_work
