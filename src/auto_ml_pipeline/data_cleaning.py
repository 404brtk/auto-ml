import numpy as np
import pandas as pd
from auto_ml_pipeline.config import CleaningConfig
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin

logger = get_logger(__name__)


def _ensure_target_exists(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")


def remove_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    before = len(df)
    df2 = df[~df[target].isna()].copy()
    logger.info("Removed %d rows with missing target", before - len(df2))
    return df2


def _compute_high_missing_cols(X: pd.DataFrame, threshold: float) -> list[str]:
    if not (0 <= threshold <= 1):
        logger.warning(
            "feature_missing_threshold %.3f is out of [0,1]; skipping", threshold
        )
        return []
    miss_ratio = X.isna().mean()
    return miss_ratio[miss_ratio > threshold].index.tolist()


def _compute_constant_cols(X: pd.DataFrame) -> list[str]:
    nunique = X.nunique(dropna=True)
    return nunique[nunique <= 1].index.tolist()


def drop_high_missing_features(
    df: pd.DataFrame, threshold: float, target: str
) -> pd.DataFrame:
    if threshold is None or threshold <= 0:
        return df
    X = df.drop(columns=[target])
    to_drop = _compute_high_missing_cols(X, threshold)
    if to_drop:
        show = ", ".join(to_drop[:20]) + (" ..." if len(to_drop) > 20 else "")
        logger.info(
            "Dropping %d features with missingness > %.2f: %s",
            len(to_drop),
            threshold,
            show,
        )
        X = X.drop(columns=to_drop)
    kept_cols_in_order = [c for c in df.columns if c != target and c in X.columns]
    return pd.concat([X[kept_cols_in_order], df[target]], axis=1)


def remove_constant_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = df.drop(columns=[target])
    to_drop = _compute_constant_cols(X)
    if to_drop:
        show = ", ".join(to_drop[:20]) + (" ..." if len(to_drop) > 20 else "")
        logger.info("Removed %d constant features: %s", len(to_drop), show)
        X = X.drop(columns=to_drop)
    kept_cols_in_order = [c for c in df.columns if c != target and c in X.columns]
    return pd.concat([X[kept_cols_in_order], df[target]], axis=1)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df2))
    return df2


def clean_data(df: pd.DataFrame, target: str, cfg: CleaningConfig) -> pd.DataFrame:
    _ensure_target_exists(df, target)
    # IMPORTANT: To avoid leakage, we only perform row-wise target cleaning pre-split.
    # Duplicate removal happens train-only after the split in trainer.
    # Feature-based decisions (missingness, constant features) are handled by
    # sklearn transformers inside the Pipeline (train-only):
    #   - FeatureMissingnessDropper
    #   - ConstantFeatureDropper
    if cfg.drop_missing_target:
        df = remove_missing_target(df, target)
    # outliers are handled post-split in trainer to avoid leakage
    return df.reset_index(drop=True)


# ---------- Sklearn transformers (train-only feature droppers) ----------
class FeatureMissingnessDropper(BaseEstimator, TransformerMixin):
    """
    Drop columns whose missing ratio (computed on training data during fit) exceeds a threshold.
    Works with pandas DataFrames. If given ndarrays at transform time, acts as passthrough.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.drop_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            self.drop_cols_ = []
            logger.warning(
                "FeatureMissingnessDropper received non-DataFrame input in fit; skipping column selection"
            )
            return self
        self.drop_cols_ = _compute_high_missing_cols(X, self.threshold)
        if self.drop_cols_:
            show = ", ".join(self.drop_cols_[:20]) + (
                " ..." if len(self.drop_cols_) > 20 else ""
            )
            logger.info(
                "[Dropper] High-missing features selected to drop (> %.2f): %d -> %s",
                self.threshold,
                len(self.drop_cols_),
                show,
            )
        else:
            logger.info("[Dropper] No high-missing features selected to drop")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.drop_cols_:
            return X.drop(columns=self.drop_cols_, errors="ignore")
        return X


class ConstantFeatureDropper(BaseEstimator, TransformerMixin):
    """
    Drop columns that are constant (nunique <= 1) on training data.
    Works with pandas DataFrames. If given ndarrays at transform time, acts as passthrough.
    """

    def __init__(self):
        self.drop_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            self.drop_cols_ = []
            logger.warning(
                "ConstantFeatureDropper received non-DataFrame input in fit; skipping column selection"
            )
            return self
        self.drop_cols_ = _compute_constant_cols(X)
        if self.drop_cols_:
            show = ", ".join(self.drop_cols_[:20]) + (
                " ..." if len(self.drop_cols_) > 20 else ""
            )
            logger.info(
                "[Dropper] Constant features selected to drop: %d -> %s",
                len(self.drop_cols_),
                show,
            )
        else:
            logger.info("[Dropper] No constant features selected to drop")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.drop_cols_:
            return X.drop(columns=self.drop_cols_, errors="ignore")
        return X


class NumericLikeCoercer(BaseEstimator, TransformerMixin):
    """
    Coerce columns to numeric values.
    Works with pandas DataFrames. If given ndarrays at transform time, acts as passthrough.
    """

    def __init__(self, threshold: float = 0.95, thousand_sep: str | None = None):
        self.threshold = float(threshold)
        # thousand_sep kept for backward compatibility; if None, auto-detect
        self.thousand_sep = thousand_sep
        self.convert_cols_: list[str] = []

    @staticmethod
    def _normalize_number_string(s: str) -> str:
        # Trim and remove spaces and apostrophes commonly used as group separators
        s = s.strip().replace(" ", "").replace("'", "")
        if s == "":
            return s
        has_comma = "," in s
        has_dot = "." in s
        # If explicit thousand_sep provided (legacy path)
        # We still run through general cleanups above first
        # and then apply simple removal or swap if sep is comma and dot used as decimal.
        # Prefer general heuristic when thousand_sep is None.
        if has_comma and has_dot:
            # Decide which is decimal: pick the last occurring symbol; the other is grouping
            last_comma = s.rfind(",")
            last_dot = s.rfind(".")
            if last_comma > last_dot:
                # Comma likely decimal, dot grouping (EU): remove dots, replace last comma with dot
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                # Dot likely decimal, comma grouping (US): remove commas
                s = s.replace(",", "")
            return s
        if has_comma and not has_dot:
            # Single symbol: decide if decimal or grouping based on count/position
            if s.count(",") == 1:
                # If exactly one comma near end with 1-2 digits after, treat as decimal.
                # Else treat as grouping.
                after = len(s) - s.rfind(",") - 1
                if 1 <= after <= 2:
                    return s.replace(",", ".")
            # Otherwise remove grouping commas
            return s.replace(",", "")
        if has_dot and not has_comma:
            # If multiple dots, likely grouping -> remove all dots
            if s.count(".") > 1:
                return s.replace(".", "")
            # One dot: assume decimal
            return s
        # No comma/dot left; already clean
        return s

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            self.convert_cols_ = []
            logger.warning(
                "NumericLikeCoercer received non-DataFrame input in fit; skipping column selection"
            )
            return self
        obj_cols = X.select_dtypes(include=["object", "category"]).columns
        convert: list[str] = []
        for c in obj_cols:
            s_raw = X[c].astype(str)
            if self.thousand_sep:
                s = s_raw.str.replace(" ", "", regex=False).str.replace(
                    "'", "", regex=False
                )
                s = s.str.replace(self.thousand_sep, "", regex=False)
            else:
                s = s_raw.apply(self._normalize_number_string)
            num = pd.to_numeric(s, errors="coerce")
            if len(num) == 0:
                continue
            ratio = float(num.notna().mean())
            if ratio >= self.threshold:
                convert.append(c)
        self.convert_cols_ = convert
        if convert:
            logger.info(
                "[Coercer] Converting %d object cols to numeric (thr=%.2f): %s",
                len(convert),
                self.threshold,
                ", ".join(convert[:20]) + (" ..." if len(convert) > 20 else ""),
            )
        else:
            logger.info("[Coercer] No numeric-like object columns detected")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or not self.convert_cols_:
            return X
        Xo = X.copy()
        for c in self.convert_cols_:
            s_raw = Xo[c].astype(str)
            if self.thousand_sep:
                s = s_raw.str.replace(" ", "", regex=False).str.replace(
                    "'", "", regex=False
                )
                s = s.str.replace(self.thousand_sep, "", regex=False)
            else:
                s = s_raw.apply(self._normalize_number_string)
            Xo[c] = pd.to_numeric(s, errors="coerce")
        return Xo


# ---------- Outlier utilities (fit on train, apply on train/test) ----------
def fit_outlier_params(df: pd.DataFrame, target: str, cfg: CleaningConfig) -> dict:
    """Compute outlier detection parameters on numeric features using train data.

    Returns a dict with necessary parameters depending on strategy.
    """
    strategy = (cfg.outlier_strategy or "").lower()
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns
    params: dict = {"strategy": strategy, "num_cols": list(num_cols)}
    if len(num_cols) == 0 or strategy in {None, "", "none"}:
        return params

    if strategy == "iqr":
        Q1 = X[num_cols].quantile(0.25)
        Q3 = X[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        k = cfg.outlier_iqr_multiplier
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        params.update({"lower": lower, "upper": upper})
    elif strategy == "zscore":
        mean = X[num_cols].mean()
        std = X[num_cols].std(ddof=0).replace(0, np.nan)
        params.update({"mean": mean, "std": std, "thr": cfg.outlier_zscore_threshold})
    elif strategy == "isoforest":
        try:
            iso = IsolationForest(
                contamination=cfg.outlier_contamination, random_state=42
            )
            iso.fit(X[num_cols].fillna(X[num_cols].median()))
            params.update({"iso": iso})
        except Exception as e:
            logger.warning("IsolationForest unavailable (%s); skipping outliers", e)
    else:
        logger.warning("Unknown outlier_strategy '%s'; skipping", strategy)
    return params


def apply_outliers(
    df: pd.DataFrame,
    target: str,
    cfg: CleaningConfig,
    params: dict,
    scope: str,
) -> pd.DataFrame:
    """Apply outlier treatment according to params.

    scope: 'train' or 'test' context for logging. Behavior is controlled by cfg.outlier_method
    and cfg.outlier_apply_scope (handled by caller for whether to apply on test).
    """
    strategy = params.get("strategy")
    num_cols = params.get("num_cols", [])
    if not strategy or not num_cols:
        return df

    X = df.drop(columns=[target]).copy()
    y = df[target]

    method = (cfg.outlier_method or "clip").lower()
    if strategy == "iqr" and "lower" in params and "upper" in params:
        lower = params["lower"]
        upper = params["upper"]
        if method == "clip":
            X[num_cols] = X[num_cols].clip(lower=lower, upper=upper, axis=1)
            logger.info("Clipped outliers via IQR on %s data", scope)
            return pd.concat([X, y], axis=1)
        else:  # remove
            mask = ~(((X[num_cols] < lower) | (X[num_cols] > upper)).any(axis=1))
            before = len(X)
            X = X[mask]
            y = y.loc[X.index]
            logger.info(
                "Removed %d outliers via IQR on %s data", before - len(X), scope
            )
            return pd.concat([X, y], axis=1)
    elif strategy == "zscore" and "mean" in params and "std" in params:
        mean = params["mean"]
        std = params["std"].replace(0, np.nan)
        thr = params.get("thr", 3.0)
        # Operate only on columns with valid std
        valid = [c for c in num_cols if pd.notna(std.get(c, np.nan))]
        if not valid:
            logger.info(
                "Z-score outlier handling skipped on %s data: no valid numeric columns",
                scope,
            )
            return pd.concat([X, y], axis=1)
        z = (X[valid] - mean[valid]) / std[valid]
        if method == "clip":
            X[valid] = X[valid].where(
                z.abs() <= thr, mean[valid] + thr * np.sign(z) * std[valid]
            )
            logger.info(
                "Clipped outliers via Z-score on %s data (thr=%.2f)", scope, thr
            )
            return pd.concat([X, y], axis=1)
        else:
            mask = (z.abs() <= thr).all(axis=1)
            before = len(X)
            X = X[mask]
            y = y.loc[X.index]
            logger.info(
                "Removed %d outliers via Z-score on %s data (thr=%.2f)",
                before - len(X),
                scope,
                thr,
            )
            return pd.concat([X, y], axis=1)
    elif strategy == "isoforest" and "iso" in params:
        iso = params["iso"]
        preds = iso.predict(
            X[num_cols].fillna(X[num_cols].median())
        )  # 1 normal, -1 outlier
        if method == "clip":
            # IsolationForest doesn't provide values to clip to; leave as-is for clip mode
            logger.info(
                "IsolationForest used with method=clip: no changes applied on %s data",
                scope,
            )
            return pd.concat([X, y], axis=1)
        else:
            mask = preds == 1
            before = len(X)
            X = X[mask]
            y = y.loc[X.index]
            logger.info(
                "Removed %d outliers via IsolationForest on %s data",
                before - len(X),
                scope,
            )
            return pd.concat([X, y], axis=1)

    return df
