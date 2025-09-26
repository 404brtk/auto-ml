import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin

logger = get_logger(__name__)


# TODO: add IsolationForest
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
