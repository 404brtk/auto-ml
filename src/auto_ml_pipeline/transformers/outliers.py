import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


logger = get_logger(__name__)


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Detect and handle outliers in numeric features using multiple strategies."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        method: str = "clip",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: Union[float, str] = "auto",
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        random_state: Optional[int] = None,
    ):
        self.strategy = strategy.lower() if strategy else None
        self.method = (method or "clip").lower()
        self.iqr_multiplier = float(iqr_multiplier)
        self.zscore_threshold = float(zscore_threshold)
        self.contamination = contamination
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.random_state = random_state

        # Fitted parameters
        self.num_cols_: List[str] = []
        self.valid_cols_: List[str] = []
        self.outlier_params_: Dict[str, Any] = {}
        self.isolation_forest_: Optional[IsolationForest] = None

        # Validation
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate initialization parameters."""
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
        if self.strategy not in [None, "iqr", "zscore", "isolation_forest"]:
            raise ValueError(
                f"strategy must be None, 'iqr', 'zscore', or 'isolation_forest', got {self.strategy}"
            )

        # Validate IsolationForest only supports 'remove'
        if self.strategy == "isolation_forest" and self.method == "clip":
            raise ValueError(
                "IsolationForest does not support method='clip'. "
                "IsolationForest is a classification-based method that identifies outliers "
                "but does not produce natural numerical bounds for clipping. "
                "Please use method='remove' with strategy='isolation_forest', "
                "or use strategy='iqr' or 'zscore' if you need clipping functionality."
            )

        # Validate contamination
        if self.contamination != "auto":
            try:
                contam_val = float(self.contamination)
                if contam_val <= 0 or contam_val > 0.5:
                    raise ValueError(
                        f"contamination must be 'auto' or in (0, 0.5], got {self.contamination}"
                    )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"contamination must be 'auto' or float in (0, 0.5], got {self.contamination}"
                ) from e

        # Validate n_estimators
        if self.n_estimators < 10 or self.n_estimators > 1000:
            raise ValueError(
                f"n_estimators should be in [10, 1000], got {self.n_estimators}"
            )

        # Validate max_samples
        if self.max_samples != "auto":
            if isinstance(self.max_samples, int):
                if self.max_samples <= 0:
                    raise ValueError(
                        f"max_samples as int must be positive, got {self.max_samples}"
                    )
            elif isinstance(self.max_samples, float):
                if self.max_samples <= 0 or self.max_samples > 1:
                    raise ValueError(
                        f"max_samples as float must be in (0, 1], got {self.max_samples}"
                    )
            else:
                raise ValueError(
                    f"max_samples must be 'auto', positive int, or float in (0, 1], got {type(self.max_samples).__name__}"
                )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit outlier detection parameters."""
        if not isinstance(X, pd.DataFrame):
            logger.warning(
                "[OutlierTransformer] Received non-DataFrame input; no outlier detection applied"
            )
            return self

        if self.strategy is None:
            logger.info(
                "[OutlierTransformer] No outlier detection requested (strategy=None)"
            )
            return self

        # Check for empty DataFrame
        if X.empty:
            logger.warning(
                "[OutlierTransformer] Empty DataFrame provided; skipping outlier detection"
            )
            return self

        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(self.num_cols_) == 0:
            logger.info(
                "[OutlierTransformer] No numeric columns found for outlier detection"
            )
            return self

        try:
            if self.strategy == "iqr":
                self._fit_iqr(X)
            elif self.strategy == "zscore":
                self._fit_zscore(X)
            elif self.strategy == "isolation_forest":
                self._fit_isolation_forest(X)

        except Exception as e:
            logger.error(
                "[OutlierTransformer] Error fitting outlier parameters for strategy '%s': %s",
                self.strategy,
                e,
                exc_info=True,
            )
            self.strategy = None
            self.valid_cols_ = []
            return self

        if self.valid_cols_:
            logger.info(
                "[OutlierTransformer] Fitted outlier detection: strategy=%s, method=%s, valid_cols=%d/%d",
                self.strategy,
                self.method,
                len(self.valid_cols_),
                len(self.num_cols_),
            )
        else:
            logger.info("[OutlierTransformer] No valid columns for outlier detection")

        return self

    def _fit_iqr(self, X: pd.DataFrame):
        Q1 = X[self.num_cols_].quantile(0.25)
        Q3 = X[self.num_cols_].quantile(0.75)
        IQR = Q3 - Q1

        # Handle zero IQR (constant columns)
        self.valid_cols_ = IQR[IQR > 1e-10].index.tolist()
        if not self.valid_cols_:
            logger.warning(
                "[OutlierTransformer] All numeric columns have zero IQR; skipping outlier detection"
            )
            return

        lower = Q1[self.valid_cols_] - self.iqr_multiplier * IQR[self.valid_cols_]
        upper = Q3[self.valid_cols_] + self.iqr_multiplier * IQR[self.valid_cols_]
        self.outlier_params_ = {"lower": lower, "upper": upper}

    def _fit_zscore(self, X: pd.DataFrame):
        mean = X[self.num_cols_].mean()
        std = X[self.num_cols_].std(ddof=1)  # Sample std

        # Filter out columns with zero/near-zero std
        self.valid_cols_ = std[std > 1e-10].index.tolist()
        if not self.valid_cols_:
            logger.warning(
                "[OutlierTransformer] All numeric columns have zero std; skipping outlier detection"
            )
            return

        self.outlier_params_ = {
            "mean": mean[self.valid_cols_],
            "std": std[self.valid_cols_],
        }

    def _fit_isolation_forest(self, X: pd.DataFrame):
        # Check for sufficient data
        if len(X) < 10:
            logger.warning(
                "[OutlierTransformer] Insufficient samples (%d) for stable IsolationForest; "
                "recommend at least 10 samples. Skipping outlier detection.",
                len(X),
            )
            return

        # Filter columns with variance
        X_numeric = X[self.num_cols_]

        # Handle NaN/inf values
        if X_numeric.isnull().any().any():
            logger.warning(
                "[OutlierTransformer] NaN values detected in numeric columns. "
                "IsolationForest requires clean data. Skipping outlier detection."
            )
            return

        if np.isinf(X_numeric.values).any():
            logger.warning(
                "[OutlierTransformer] Inf values detected in numeric columns. "
                "IsolationForest requires clean data. Skipping outlier detection."
            )
            return

        variances = X_numeric.var()
        self.valid_cols_ = variances[variances > 1e-10].index.tolist()

        if not self.valid_cols_:
            logger.warning(
                "[OutlierTransformer] All numeric columns have zero variance; skipping outlier detection"
            )
            return

        # Fit IsolationForest
        self.isolation_forest_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
            warm_start=False,
        )
        self.isolation_forest_.fit(X[self.valid_cols_])

        # Log effective max_samples value for clarity
        effective_max_samples = self.max_samples
        if self.max_samples == "auto":
            effective_max_samples = f"min(256, {len(X)}) = {min(256, len(X))}"
        elif isinstance(self.max_samples, float):
            effective_max_samples = (
                f"{self.max_samples} * {len(X)} = {int(self.max_samples * len(X))}"
            )

        logger.info(
            "[OutlierTransformer] IsolationForest fitted: contamination=%s, n_estimators=%d, "
            "max_samples=%s (effective: %s), method='remove'",
            self.contamination,
            self.n_estimators,
            self.max_samples,
            effective_max_samples,
        )

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply outlier treatment to data."""
        if not isinstance(X, pd.DataFrame):
            return X

        if self.strategy is None or not self.valid_cols_:
            return X

        if X.empty:
            return X

        X_work = X.copy()
        original_shape = X_work.shape

        try:
            if self.strategy == "iqr":
                self._transform_iqr(X, X_work)
            elif self.strategy == "zscore":
                self._transform_zscore(X, X_work)
            elif self.strategy == "isolation_forest":
                self._transform_isolation_forest(X, X_work)

        except Exception as e:
            logger.error(
                "[OutlierTransformer] Error applying outlier treatment for strategy '%s': %s",
                self.strategy,
                e,
                exc_info=True,
            )
            return X

        # Log shape changes
        if X_work.shape != original_shape:
            logger.info(
                "[OutlierTransformer] Shape changed: %s -> %s (%d rows removed, %.2f%%)",
                original_shape,
                X_work.shape,
                original_shape[0] - X_work.shape[0],
                100 * (original_shape[0] - X_work.shape[0]) / original_shape[0],
            )
        else:
            logger.debug(
                "[OutlierTransformer] Transform complete: shape unchanged %s",
                original_shape,
            )

        return X_work

    def _transform_iqr(self, X_original: pd.DataFrame, X_work: pd.DataFrame):
        """Apply IQR-based outlier treatment."""
        if "lower" not in self.outlier_params_ or "upper" not in self.outlier_params_:
            return

        lower = self.outlier_params_["lower"]
        upper = self.outlier_params_["upper"]

        if self.method == "clip":
            X_work[self.valid_cols_] = X_work[self.valid_cols_].clip(
                lower=lower, upper=upper, axis=1
            )
            outliers_handled = (
                (
                    (X_original[self.valid_cols_] < lower)
                    | (X_original[self.valid_cols_] > upper)
                )
                .sum()
                .sum()
            )
            if outliers_handled > 0:
                logger.info(
                    "[OutlierTransformer] Clipped %d outlier values via IQR",
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
            X_work.drop(X_work.index[~mask], inplace=True)
            removed = before - len(X_work)
            if removed > 0:
                logger.info(
                    "[OutlierTransformer] Removed %d outlier rows (%.2f%%) via IQR",
                    removed,
                    100 * removed / before,
                )

    def _transform_zscore(self, X_original: pd.DataFrame, X_work: pd.DataFrame):
        """Apply Z-score based outlier treatment."""
        if "mean" not in self.outlier_params_ or "std" not in self.outlier_params_:
            return

        mean = self.outlier_params_["mean"]
        std = self.outlier_params_["std"]
        z_scores = np.abs((X_work[self.valid_cols_] - mean) / std)

        if self.method == "clip":
            outlier_mask = z_scores > self.zscore_threshold
            for col in self.valid_cols_:
                col_outliers = outlier_mask[col]
                if col_outliers.any():
                    lower_bound = mean[col] - self.zscore_threshold * std[col]
                    upper_bound = mean[col] + self.zscore_threshold * std[col]
                    X_work.loc[col_outliers, col] = X_work.loc[col_outliers, col].clip(
                        lower=lower_bound, upper=upper_bound
                    )

            outliers_handled = outlier_mask.sum().sum()
            if outliers_handled > 0:
                logger.info(
                    "[OutlierTransformer] Clipped %d outlier values via Z-score",
                    outliers_handled,
                )
        else:  # remove
            mask = ~(z_scores > self.zscore_threshold).any(axis=1)
            before = len(X_work)
            X_work.drop(X_work.index[~mask], inplace=True)
            removed = before - len(X_work)
            if removed > 0:
                logger.info(
                    "[OutlierTransformer] Removed %d outlier rows (%.2f%%) via Z-score",
                    removed,
                    100 * removed / before,
                )

    def _transform_isolation_forest(
        self, X_original: pd.DataFrame, X_work: pd.DataFrame
    ):
        """
        Apply IsolationForest-based outlier treatment (remove only).

        Note: IsolationForest only supports 'remove' method because it produces
        binary classifications without natural numerical bounds for clipping.
        """
        if self.isolation_forest_ is None:
            return

        # Predict outliers (-1 for outliers, 1 for inliers)
        predictions = self.isolation_forest_.predict(X_work[self.valid_cols_])
        outlier_mask = predictions == -1

        before = len(X_work)
        X_work.drop(X_work.index[outlier_mask], inplace=True)
        removed = before - len(X_work)

        if removed > 0:
            logger.info(
                "[OutlierTransformer] Removed %d outlier rows (%.2f%%) via IsolationForest",
                removed,
                100 * removed / before,
            )
        else:
            logger.info(
                "[OutlierTransformer] IsolationForest detected no outliers (all samples classified as inliers)"
            )

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation (sklearn compatibility)."""
        if input_features is None:
            return self.num_cols_ if hasattr(self, "num_cols_") else None
        return input_features
