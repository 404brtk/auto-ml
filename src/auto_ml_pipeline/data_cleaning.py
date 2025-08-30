import numpy as np
import pandas as pd
from auto_ml_pipeline.config import CleaningConfig
from auto_ml_pipeline.logging_utils import get_logger
from sklearn.ensemble import IsolationForest

logger = get_logger(__name__)


def _ensure_target_exists(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")


def remove_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    before = len(df)
    df2 = df[~df[target].isna()].copy()
    logger.info("Removed %d rows with missing target", before - len(df2))
    return df2


def drop_high_missing_features(
    df: pd.DataFrame, threshold: float, target: str
) -> pd.DataFrame:
    if threshold <= 0:
        return df
    if not 0 <= threshold <= 1:
        logger.warning(
            "feature_missing_threshold %.3f is out of [0,1]; skipping", threshold
        )
        return df
    missing_ratio = df.drop(columns=[target]).isna().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if to_drop:
        show = ", ".join(to_drop[:20]) + (" ..." if len(to_drop) > 20 else "")
        logger.info(
            "Dropping %d features with missingness > %.2f: %s",
            len(to_drop),
            threshold,
            show,
        )
        return df.drop(columns=to_drop)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df2))
    return df2


def remove_constant_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = df.drop(columns=[target])
    nunique = X.nunique(dropna=True)
    to_drop_const = nunique[nunique <= 1].index.tolist()
    if to_drop_const:
        show = ", ".join(to_drop_const[:20]) + (
            " ..." if len(to_drop_const) > 20 else ""
        )
        logger.info("Removed %d constant features: %s", len(to_drop_const), show)
        X = X.drop(columns=to_drop_const)

    # Preserve original column order
    kept_cols_in_order = [c for c in df.columns if c != target and c in X.columns]
    return pd.concat([X[kept_cols_in_order], df[target]], axis=1)


def clean_data(df: pd.DataFrame, target: str, cfg: CleaningConfig) -> pd.DataFrame:
    _ensure_target_exists(df, target)
    if cfg.drop_missing_target:
        df = remove_missing_target(df, target)
    if cfg.remove_duplicates:
        df = drop_duplicates(df)
    if cfg.feature_missing_threshold is not None:
        df = drop_high_missing_features(df, cfg.feature_missing_threshold, target)
    if cfg.remove_constant:
        df = remove_constant_features(df, target)
    # outliers are handled post-split in trainer to avoid leakage
    return df.reset_index(drop=True)


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
