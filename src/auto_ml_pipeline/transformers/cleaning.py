import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin

logger = get_logger(__name__)


class NumericLikeCoercer(BaseEstimator, TransformerMixin):
    """Numeric coercion with number format detection."""

    def __init__(
        self,
        threshold: float = 0.95,
        thousand_sep: Optional[str] = None,
        sample_size: int = 10000,
        target_col: Optional[str] = None,
    ):
        if not (0 < threshold <= 1):
            raise ValueError(f"threshold must be in (0,1], got {threshold}")

        self.threshold = float(threshold)
        self.thousand_sep = thousand_sep
        self.sample_size = max(1000, sample_size)  # Minimum sample size
        self.target_col = target_col
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
                    # Warn if conversion will create NaN values
                    if conversion_rate < 1.0:
                        failed_pct = 100 * (1 - conversion_rate)
                        if self.target_col and col == self.target_col:
                            logger.warning(
                                "Target column '%s' will be converted to numeric, but %.1f%% of values "
                                "will become NaN (non-numeric strings). These rows WILL BE DROPPED during cleaning. "
                                "Consider pre-cleaning these values before training.",
                                col,
                                failed_pct,
                            )
                        else:
                            logger.warning(
                                "Column '%s' will be converted to numeric, but %.1f%% of values "
                                "will become NaN (non-numeric strings).",
                                col,
                                failed_pct,
                            )
                elif conversion_rate > 0.3:  # Warn for ambiguous columns
                    logger.warning(
                        "Column '%s' is %.1f%% numeric-coercible (threshold: %.1f%%). "
                        "This mixed-type column will NOT be converted and may affect task inference. "
                        "Consider: (1) cleaning invalid values manually before training, "
                        "or (2) lowering 'numeric_coercion_threshold' in config if %.1f%% is acceptable.",
                        col,
                        100 * conversion_rate,
                        100 * self.threshold,
                        100 * conversion_rate,
                    )

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
                + (f" ... (+{len(convert) - 10} more)" if len(convert) > 10 else ""),
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
