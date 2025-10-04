import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin
import re

logger = get_logger(__name__)


class NumericLikeCoercer(BaseEstimator, TransformerMixin):
    """Convert string columns to numeric with intelligent format detection."""

    def __init__(
        self,
        threshold: float = 0.95,
        sample_size: int = 10000,
        target_col: Optional[str] = None,
        detect_integers: bool = True,
    ):
        if not (0 < threshold <= 1):
            raise ValueError(f"threshold must be in (0,1], got {threshold}")

        self.threshold = float(threshold)
        self.sample_size = max(1000, sample_size)
        self.target_col = target_col
        self.detect_integers = detect_integers

        # Fitted attributes
        self.convert_cols_: List[str] = []
        self.conversion_stats_: Dict[str, Dict[str, Any]] = {}
        self.detected_dtypes_: Dict[str, np.dtype] = {}

    def _clean_numeric_string(self, s: str) -> str:
        """Remove common non-numeric characters but preserve potential number structure."""
        if pd.isna(s) or not isinstance(s, str):
            return ""

        # Remove currency symbols, whitespace, and common formatting
        s = s.strip()
        s = re.sub(r"[\$€£¥₹\s\'_]", "", s)  # Remove currency and whitespace
        return s

    def _detect_format(self, series: pd.Series) -> Dict[str, Any]:
        """Detect number format."""
        sample = series.dropna().astype(str).apply(self._clean_numeric_string)
        sample = sample[sample.str.len() > 0]

        if len(sample) == 0:
            return {"decimal_sep": ".", "thousands_sep": None, "confidence": 0.0}

        # Take a reasonable sample for analysis
        sample = sample.head(min(1000, len(sample)))

        # Count patterns based on separator positions and structure
        us_votes = 0  # Evidence for US format (comma thousands, dot decimal)
        eu_votes = 0  # Evidence for EU format (dot thousands, comma decimal)

        for val in sample:
            # Skip if no separators
            if "," not in val and "." not in val:
                continue

            # Both separators present - strongest signal
            if "," in val and "." in val:
                comma_pos = val.rfind(",")
                dot_pos = val.rfind(".")

                if dot_pos > comma_pos:
                    # Dot is last: 1,234.56 (US format)
                    us_votes += 2  # Strong signal
                else:
                    # Comma is last: 1.234,56 (EU format)
                    eu_votes += 2  # Strong signal

            # Only comma present
            elif "," in val:
                parts = val.split(",")
                last_part_len = len(parts[-1])

                # Comma with exactly 2 digits after: likely EU decimal (12,34)
                if last_part_len == 2 and parts[-1].isdigit():
                    eu_votes += 1
                # Comma with exactly 3 digits after: likely US thousands (1,234)
                elif last_part_len == 3 and parts[-1].isdigit():
                    us_votes += 1
                # Multiple commas every 3 digits: likely US thousands (1,234,567)
                elif len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
                    us_votes += 1

            # Only dot present
            elif "." in val:
                parts = val.split(".")
                last_part_len = len(parts[-1])

                # Dot with exactly 2 digits after: likely US decimal (12.34)
                if last_part_len == 2 and parts[-1].isdigit():
                    us_votes += 1
                # Dot with exactly 3 digits after: likely EU thousands (1.234)
                elif last_part_len == 3 and parts[-1].isdigit():
                    eu_votes += 1
                # Multiple dots every 3 digits: likely EU thousands (1.234.567)
                elif len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
                    eu_votes += 1

        total_votes = us_votes + eu_votes

        if total_votes == 0:
            # No clear pattern, assume US format (default)
            return {"decimal_sep": ".", "thousands_sep": None, "confidence": 0.5}

        # Determine format
        if us_votes >= eu_votes:
            confidence = us_votes / total_votes
            return {"decimal_sep": ".", "thousands_sep": ",", "confidence": confidence}
        else:
            confidence = eu_votes / total_votes
            return {"decimal_sep": ",", "thousands_sep": ".", "confidence": confidence}

    def _normalize_to_numeric_string(
        self, s: str, decimal_sep: str, thousands_sep: Optional[str]
    ) -> str:
        """Convert to standard numeric format (dot as decimal, no thousands separator)."""
        if not isinstance(s, str) or not s:
            return ""

        s = self._clean_numeric_string(s)
        if not s:
            return ""

        # Remove thousands separator
        if thousands_sep:
            s = s.replace(thousands_sep, "")

        # Convert decimal separator to dot
        if decimal_sep == ",":
            s = s.replace(",", ".")

        return s

    def _determine_dtype(self, numeric_series: pd.Series) -> np.dtype:
        """Determine optimal numeric dtype for sklearn compatibility.

        When detect_integers=True:
        - Converts to int32 or int64 only (with int8/int16 or unsigned types it gets problematic and e.g. TargetEncoder fails)
        - Uses float64 if NaN values present or if decimals present

        When detect_integers=False:
        - Always uses float64
        """
        if not self.detect_integers:
            return np.dtype(np.float64)

        # Check if all non-null values are integers
        non_null = numeric_series.dropna()
        if len(non_null) == 0:
            return np.dtype(np.float64)

        # If NaN present, must use float64
        has_nulls = numeric_series.isna().any()
        if has_nulls:
            return np.dtype(np.float64)

        # Check if all values are whole numbers
        if (non_null % 1 == 0).all():
            # Determine integer size needed
            min_val, max_val = non_null.min(), non_null.max()

            # Only use int32 or int64 for sklearn compatibility
            # Avoid int8, int16, uint types which can cause issues
            if np.iinfo(np.int32).min <= min_val and max_val <= np.iinfo(np.int32).max:
                return np.dtype(np.int32)
            else:
                return np.dtype(np.int64)

        # Fallback - has decimals
        return np.dtype(np.float64)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Detect numeric-like columns and learn conversion parameters."""
        if not isinstance(X, pd.DataFrame):
            self.convert_cols_ = []
            self.conversion_stats_ = {}
            self.detected_dtypes_ = {}
            logger.warning("NumericLikeCoercer requires DataFrame input; skipping")
            return self

        # Only consider object and category columns
        candidate_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not candidate_cols:
            logger.info("[Coercer] No object/category columns to analyze")
            self.convert_cols_ = []
            self.conversion_stats_ = {}
            self.detected_dtypes_ = {}
            return self

        convert_cols = []

        for col in candidate_cols:
            try:
                series = X[col]

                # Sample for efficiency
                if len(series) > self.sample_size:
                    series_sample = series.sample(n=self.sample_size, random_state=42)
                else:
                    series_sample = series

                # Skip if all null
                if series_sample.isna().all():
                    continue

                # Detect format
                format_info = self._detect_format(series_sample)

                # Attempt conversion on sample
                cleaned = series_sample.astype(str).apply(
                    lambda x: self._normalize_to_numeric_string(
                        x, format_info["decimal_sep"], format_info["thousands_sep"]
                    )
                )

                # Convert to numeric
                numeric = pd.to_numeric(cleaned, errors="coerce")

                # Calculate conversion rate (excluding original NaN values)
                original_valid = series_sample.notna()
                successfully_converted = numeric.notna() & original_valid
                conversion_rate = (
                    successfully_converted.sum() / original_valid.sum()
                    if original_valid.sum() > 0
                    else 0.0
                )

                if conversion_rate >= self.threshold:
                    # Determine optimal dtype
                    optimal_dtype = self._determine_dtype(numeric)

                    convert_cols.append(col)
                    self.conversion_stats_[col] = {
                        "conversion_rate": float(conversion_rate),
                        "format_info": format_info,
                        "sample_size": len(series_sample),
                        "original_nulls": series_sample.isna().sum(),
                    }
                    self.detected_dtypes_[col] = optimal_dtype

                    # Warnings for imperfect conversion
                    if conversion_rate < 1.0:
                        failed_pct = 100 * (1 - conversion_rate)
                        if self.target_col and col == self.target_col:
                            logger.warning(
                                "Target column '%s' will be converted to numeric, but %.1f%% of non-null values "
                                "cannot be parsed and will become NaN. These rows will be dropped during cleaning. "
                                "Review and clean these values before training.",
                                col,
                                failed_pct,
                            )
                        else:
                            logger.info(
                                "Column '%s' will be converted to numeric with %.1f%% conversion rate (%.1f%% unparseable)",
                                col,
                                100 * conversion_rate,
                                failed_pct,
                            )

                elif conversion_rate > 0.3:
                    # Warn about ambiguous columns
                    logger.warning(
                        "Column '%s' is %.1f%% numeric-convertible but below threshold (%.1f%%). "
                        "This mixed-type column may affect downstream processing. "
                        "Consider: (1) manually cleaning invalid values, or (2) adjusting threshold to %.1f%%",
                        col,
                        100 * conversion_rate,
                        100 * self.threshold,
                        100 * conversion_rate,
                    )

            except Exception as e:
                logger.warning(
                    "Error analyzing column '%s' for numeric conversion: %s",
                    col,
                    str(e),
                )
                continue

        self.convert_cols_ = convert_cols

        if convert_cols:
            logger.info(
                "[Coercer] Will convert %d column(s) to numeric: %s",
                len(convert_cols),
                ", ".join(convert_cols[:5])
                + (
                    f" ... (+{len(convert_cols) - 5} more)"
                    if len(convert_cols) > 5
                    else ""
                ),
            )
        else:
            logger.info("[Coercer] No columns meet numeric conversion threshold")

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply numeric conversion to identified columns."""
        if not isinstance(X, pd.DataFrame) or not self.convert_cols_:
            return X

        X_out = X.copy()

        for col in self.convert_cols_:
            if col not in X_out.columns:
                logger.warning("Column '%s' not found in transform data; skipping", col)
                continue

            try:
                stats = self.conversion_stats_[col]
                format_info = stats["format_info"]

                # Normalize numeric strings
                cleaned = (
                    X_out[col]
                    .astype(str)
                    .apply(
                        lambda x: self._normalize_to_numeric_string(
                            x, format_info["decimal_sep"], format_info["thousands_sep"]
                        )
                    )
                )

                # Convert to numeric
                numeric = pd.to_numeric(cleaned, errors="coerce")

                # Cast to optimal dtype (handles nullable integers correctly)
                target_dtype = self.detected_dtypes_.get(col, np.float64)
                X_out[col] = numeric.astype(target_dtype)

            except Exception as e:
                logger.error(
                    "Error converting column '%s': %s. Keeping original.", col, str(e)
                )
                continue

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility."""
        if input_features is None:
            return np.array([])
        return np.asarray(input_features, dtype=object)
