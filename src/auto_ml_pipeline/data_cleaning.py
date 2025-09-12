import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Optional, Union, TypedDict
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
    """Clean data with pre-split operations."""
    _validate_config(cfg)
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


# datetime and time detection and conversion utilities
class DateTimeDetectionResult(TypedDict):
    is_datetime: bool
    confidence: float
    format: Optional[str]
    successful_parses: int


class TimeDetectionResult(TypedDict):
    is_time: bool
    confidence: float
    format: Optional[str]
    successful_parses: int


def _detect_datetime_patterns(
    series: pd.Series, sample_size: int = 1000
) -> DateTimeDetectionResult:
    """Detect common datetime patterns in a string series."""
    # Sample the series for efficiency
    sample = series.dropna().astype(str).str.strip()
    if len(sample) == 0:
        return {
            "is_datetime": False,
            "confidence": 0.0,
            "format": None,
            "successful_parses": 0,
        }

    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=42)

    # Check if this appears to be a time-only column (already processed by TimeConverter)
    # If 70% or more values match HH:MM:SS pattern, exclude from datetime detection
    time_pattern_matches = sample.str.match(r"^\d{1,2}:\d{2}:\d{2}$").sum()
    if len(sample) > 0 and time_pattern_matches >= 0.7 * len(sample):
        return {
            "is_datetime": False,
            "confidence": 0.0,
            "format": None,
            "successful_parses": 0,
        }

    # Common datetime patterns to test
    datetime_patterns = [
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        (r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$", ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"]),
        # MM-DD-YYYY, MM/DD/YYYY, MM.DD.YYYY
        (r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$", ["%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y"]),
        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        (r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]),
        # DD-MM-YY, DD/MM/YY, DD.MM.YY
        (r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2}$", ["%d-%m-%y", "%d/%m/%y", "%d.%m.%y"]),
        # MM-DD-YY, MM/DD/YY, MM.DD.YY
        (r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2}$", ["%m-%d-%y", "%m/%d/%y", "%m.%d.%y"]),
        # ISO format with time
        (
            r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",
            ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"],
        ),
        # Month names
        (
            r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
            ["%d %b %Y"],
        ),
        (
            r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}$",
            ["%b %d, %Y", "%b %d %Y"],
        ),
    ]

    best_result: DateTimeDetectionResult = {
        "is_datetime": False,
        "confidence": 0.0,
        "format": None,
        "successful_parses": 0,
    }

    for pattern_regex, formats in datetime_patterns:
        # Check if values match the pattern
        pattern_matches = sample.str.match(pattern_regex, case=False).sum()
        if pattern_matches == 0:
            continue

        # Try each format for this pattern
        for fmt in formats:
            try:
                successful_parses = 0
                for value in sample.head(min(100, len(sample))):
                    try:
                        pd.to_datetime(value, format=fmt, errors="raise")
                        successful_parses += 1
                    except (ValueError, TypeError):
                        continue

                if successful_parses > 0:
                    confidence = successful_parses / len(
                        sample.head(min(100, len(sample)))
                    )
                    if confidence > best_result["confidence"]:
                        best_result = {
                            "is_datetime": confidence
                            >= 0.7,  # At least 70% success rate
                            "confidence": confidence,
                            "format": fmt,
                            "successful_parses": successful_parses,
                        }
            except Exception:
                continue

    # Also try pandas' flexible parsing as fallback
    # But skip if this looks like time-only data that should be handled by TimeConverter
    if not best_result["is_datetime"]:
        # Double-check for time patterns before pandas fallback
        time_pattern_matches = sample.str.match(r"^\d{1,2}:\d{2}:\d{2}$").sum()
        if len(sample) > 0 and time_pattern_matches >= 0.7 * len(sample):
            # Skip pandas fallback for time-only data
            pass
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually",
                        category=UserWarning,
                    )
                    parsed = pd.to_datetime(
                        sample.head(min(50, len(sample))), errors="coerce"
                    )
                valid_parses = parsed.notna().sum()
                confidence = valid_parses / len(parsed)
                if confidence >= 0.7:
                    best_result = {
                        "is_datetime": True,
                        "confidence": confidence,
                        "format": "infer",  # Let pandas infer
                        "successful_parses": valid_parses,
                    }
            except Exception:
                pass

    return best_result


class DateTimeConverter(BaseEstimator, TransformerMixin):
    """Convert string columns that contain datetime values to actual datetime types."""

    def __init__(self, confidence_threshold: float = 0.7, sample_size: int = 1000):
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size
        self.datetime_cols_: Dict[str, DateTimeDetectionResult] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            self.datetime_cols_ = {}
            logger.warning("DateTimeConverter received non-DataFrame; skipping")
            return self

        # Only check object/string columns
        string_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in string_cols:
            try:
                detection_result = _detect_datetime_patterns(X[col], self.sample_size)
                if (
                    detection_result["is_datetime"]
                    and detection_result["confidence"] >= self.confidence_threshold
                ):
                    self.datetime_cols_[col] = detection_result
                    logger.info(
                        "[DateTimeConverter] Detected datetime column '%s' with %.1f%% confidence (format: %s)",
                        col,
                        detection_result["confidence"] * 100,
                        detection_result["format"],
                    )
            except Exception as e:
                logger.warning(
                    "Error analyzing column %s for datetime conversion: %s", col, e
                )
                continue

        if not self.datetime_cols_:
            logger.info("[DateTimeConverter] No datetime columns detected")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame) or not self.datetime_cols_:
            return X

        X_out = X.copy()

        for col, detection_info in self.datetime_cols_.items():
            if col not in X_out.columns:
                continue

            try:
                fmt = detection_info["format"]
                if fmt == "infer":
                    # Let pandas infer the format (suppress noisy UserWarning)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Could not infer format, so each element will be parsed individually",
                            category=UserWarning,
                        )
                        X_out[col] = pd.to_datetime(X_out[col], errors="coerce")
                else:
                    # Use the specific format we detected
                    X_out[col] = pd.to_datetime(X_out[col], format=fmt, errors="coerce")

            except Exception as e:
                logger.warning("Error converting column %s to datetime: %s", col, e)
                continue

        return X_out


def _detect_time_patterns(
    series: pd.Series, sample_size: int = 1000
) -> TimeDetectionResult:
    """Detect common time-only patterns in a string series."""
    # Sample the series for efficiency
    sample = series.dropna().astype(str).str.strip()
    if len(sample) == 0:
        return {
            "is_time": False,
            "confidence": 0.0,
            "format": None,
            "successful_parses": 0,
        }

    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=42)

    # Time-only patterns to test
    time_patterns = [
        # HH:MM:SS
        (r"^\d{1,2}:\d{2}:\d{2}$", ["%H:%M:%S"]),
        # HH:MM
        (r"^\d{1,2}:\d{2}$", ["%H:%M"]),
        # HH:MM AM/PM
        (r"^\d{1,2}:\d{2}\s*(AM|PM|am|pm)$", ["%I:%M %p"]),
        # HH:MM:SS AM/PM
        (r"^\d{1,2}:\d{2}:\d{2}\s*(AM|PM|am|pm)$", ["%I:%M:%S %p"]),
    ]

    best_result: TimeDetectionResult = {
        "is_time": False,
        "confidence": 0.0,
        "format": None,
        "successful_parses": 0,
    }

    for pattern_regex, formats in time_patterns:
        # Check if values match the pattern
        pattern_matches = sample.str.match(pattern_regex, case=False).sum()
        if pattern_matches == 0:
            continue

        # Try each format for this pattern
        for fmt in formats:
            try:
                successful_parses = 0
                for value in sample.head(min(100, len(sample))):
                    try:
                        pd.to_datetime(value, format=fmt, errors="raise")
                        successful_parses += 1
                    except (ValueError, TypeError):
                        continue

                if successful_parses > 0:
                    confidence = successful_parses / len(
                        sample.head(min(100, len(sample)))
                    )
                    if confidence > best_result["confidence"]:
                        best_result = {
                            "is_time": confidence >= 0.7,  # At least 70% success rate
                            "confidence": confidence,
                            "format": fmt,
                            "successful_parses": successful_parses,
                        }
            except Exception:
                continue

    return best_result


# [Example time conversion]
#  arr_time: raw=['21:05', '08:40', '06:35'] -> clean=['21:05:00', '08:40:00', '06:35:00']
class TimeConverter(BaseEstimator, TransformerMixin):
    """Convert string columns that contain time-only values to normalized time strings."""

    def __init__(self, confidence_threshold: float = 0.7, sample_size: int = 1000):
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size
        self.time_cols_: Dict[str, TimeDetectionResult] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            self.time_cols_ = {}
            logger.warning("TimeConverter received non-DataFrame; skipping")
            return self

        # Only check object/string columns
        string_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in string_cols:
            try:
                detection_result = _detect_time_patterns(X[col], self.sample_size)
                if (
                    detection_result["is_time"]
                    and detection_result["confidence"] >= self.confidence_threshold
                ):
                    self.time_cols_[col] = detection_result
                    logger.info(
                        "[TimeConverter] Detected time column '%s' with %.1f%% confidence (format: %s)",
                        col,
                        detection_result["confidence"] * 100,
                        detection_result["format"],
                    )
            except Exception as e:
                logger.warning(
                    "Error analyzing column %s for time conversion: %s", col, e
                )
                continue

        if not self.time_cols_:
            logger.info("[TimeConverter] No time columns detected")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame) or not self.time_cols_:
            return X

        X_out = X.copy()

        for col, detection_info in self.time_cols_.items():
            if col not in X_out.columns:
                continue

            try:
                fmt = detection_info["format"]

                # Parse the time values and convert to normalized time strings
                parsed_times: List[Optional[str]] = []
                for value in X_out[col]:
                    if pd.isna(value):
                        parsed_times.append(None)
                        continue

                    try:
                        # Parse as time and extract time components
                        time_obj = pd.to_datetime(
                            str(value), format=fmt, errors="raise"
                        ).time()
                        # Convert to HH:MM:SS format
                        normalized_time = time_obj.strftime("%H:%M:%S")
                        parsed_times.append(normalized_time)
                    except (ValueError, TypeError):
                        parsed_times.append(None)

                X_out[col] = parsed_times

            except Exception as e:
                logger.warning("Error converting column %s to time: %s", col, e)
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
