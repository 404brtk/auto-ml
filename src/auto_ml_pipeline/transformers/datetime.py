import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Optional, Union, TypedDict
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin


logger = get_logger(__name__)


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

                # Time parsing and normalization
                try:
                    # Convert to datetime
                    datetime_series = pd.to_datetime(
                        X_out[col], format=fmt, errors="coerce"
                    )

                    # Extract time component and format as HH:MM:SS
                    # Use dt.time to get time objects, then format them
                    time_strings = datetime_series.dt.strftime("%H:%M:%S")

                    # Handle NaT values (which become NaN after strftime)
                    time_strings = time_strings.where(datetime_series.notna(), None)

                    X_out[col] = time_strings

                except Exception as e:
                    logger.warning(
                        "Error in vectorized time conversion for column %s, falling back to row-by-row: %s",
                        col,
                        e,
                    )
                    # Fallback to row-by-row conversion
                    parsed_times: List[Optional[str]] = []
                    for value in X_out[col]:
                        if pd.isna(value):
                            parsed_times.append(None)
                            continue

                        try:
                            time_obj = pd.to_datetime(
                                str(value), format=fmt, errors="raise"
                            ).time()
                            normalized_time = time_obj.strftime("%H:%M:%S")
                            parsed_times.append(normalized_time)
                        except (ValueError, TypeError):
                            parsed_times.append(None)
                    X_out[col] = parsed_times

            except Exception as e:
                logger.warning("Error converting column %s to time: %s", col, e)
                continue

        return X_out


class SimpleDateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract basic datetime features (year, month, day, dayofweek, quarter, is_weekend)."""

    def __init__(self):
        self.datetime_cols_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit stores datetime columns and generates output feature names."""
        self.datetime_cols_ = X.columns.tolist()
        self.feature_names_out_ = []
        for col in self.datetime_cols_:
            self.feature_names_out_.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_quarter",
                    f"{col}_is_weekend",
                ]
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract datetime features as numeric array."""
        features = []
        for col in self.datetime_cols_:
            dt_series = pd.to_datetime(X[col], errors="coerce")
            features.extend(
                [
                    dt_series.dt.year.values,
                    dt_series.dt.month.values,
                    dt_series.dt.day.values,
                    dt_series.dt.dayofweek.values,
                    dt_series.dt.quarter.values,
                    (dt_series.dt.dayofweek >= 5).astype(int).values,
                ]
            )
        return np.column_stack(features) if features else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformed data."""
        return np.array(self.feature_names_out_)


class SimpleTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract basic time features (hour, minute, second, is_business_hours, time_of_day_category)."""

    def __init__(self):
        self.time_cols_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit stores time columns and generates output feature names."""
        self.time_cols_ = X.columns.tolist()
        self.feature_names_out_ = []
        for col in self.time_cols_:
            self.feature_names_out_.extend(
                [
                    f"{col}_hour",
                    f"{col}_minute",
                    f"{col}_second",
                    f"{col}_is_business_hours",
                    f"{col}_time_category",  # 0=night, 1=morning, 2=afternoon, 3=evening
                ]
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract time features as numeric array."""
        features = []
        for col in self.time_cols_:
            # Parse time strings (HH:MM or HH:MM:SS format)
            time_parts = X[col].astype(str).str.split(":", expand=True)

            # Convert to numeric, fill NaN with 0
            hours = (
                pd.to_numeric(time_parts.iloc[:, 0], errors="coerce")
                .fillna(0)
                .astype(int)
            )

            minutes = (
                (
                    pd.to_numeric(time_parts.iloc[:, 1], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                if time_parts.shape[1] > 1
                else pd.Series([0] * len(X), dtype=int)
            )

            seconds = (
                (
                    pd.to_numeric(time_parts.iloc[:, 2], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                if time_parts.shape[1] > 2
                else pd.Series([0] * len(X), dtype=int)
            )

            # Business hours (9 AM to 5 PM)
            is_business_hours = ((hours >= 9) & (hours < 17)).astype(int)

            # Time categories: 0=night (0-6), 1=morning (6-12), 2=afternoon (12-18), 3=evening (18-24)
            time_category = np.where(
                hours < 6, 0, np.where(hours < 12, 1, np.where(hours < 18, 2, 3))
            )

            features.extend(
                [
                    hours.values,
                    minutes.values,
                    seconds.values,
                    is_business_hours.values,
                    time_category,
                ]
            )

        return np.column_stack(features) if features else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformed data."""
        return np.array(self.feature_names_out_)
