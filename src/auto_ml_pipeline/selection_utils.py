from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def correlation_threshold_columns(
    X: pd.DataFrame,
    threshold: float = 0.95,
    prefer: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Return a list of columns to DROP based on absolute correlation threshold among numeric features.
    Deterministic rule: keep the first column in order (or by `prefer` order if provided), drop later ones.
    """
    if threshold <= 0 or threshold >= 1:
        return []
    X_num = X.select_dtypes(include=[np.number])
    cols = list(X_num.columns)
    if not cols:
        return []
    order = (
        list(prefer) + [c for c in cols if c not in (prefer or [])] if prefer else cols
    )
    # map priority index
    prio = {c: i for i, c in enumerate(order)}

    corr = X_num.corr().abs().fillna(0.0)
    to_drop: set[str] = set()
    kept: List[str] = []
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        kept.append(c1)
        for c2 in cols[i + 1 :]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] > threshold:
                # drop the one with lower priority (higher prio index)
                drop = c2 if prio.get(c1, i) <= prio.get(c2, i + 1) else c1
                if drop == c1:
                    to_drop.add(c1)
                    break
                else:
                    to_drop.add(c2)
    logger.info(
        "Correlation thresholding: dropping %d features (> %.2f)",
        len(to_drop),
        threshold,
    )
    return list(to_drop)


def permutation_importance_report(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance and return a DataFrame sorted by importance.
    Works for fitted estimators compatible with sklearn's permutation_importance.
    """
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
    else:
        names = [f"f{i}" for i in range(result.importances_mean.shape[0])]
    df = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df.reset_index(drop=True)
