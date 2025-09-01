from typing import Any, Dict, Tuple

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Optional deps
_HAS_XGB = True
_HAS_LGBM = True
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    _HAS_LGBM = False


def available_models_classification(
    random_state: int = 42, n_jobs: int = 1
) -> Dict[str, Any]:
    models = {
        "random_forest": RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state
        ),
        "gbrt": GradientBoostingClassifier(random_state=random_state),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )
    if _HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )
    return models


def available_models_regression(
    random_state: int = 42, n_jobs: int = 1
) -> Dict[str, Any]:
    models = {
        "random_forest": RandomForestRegressor(
            n_jobs=n_jobs, random_state=random_state
        ),
        "gbrt": GradientBoostingRegressor(random_state=random_state),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
        )
    if _HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )
    return models


# spec: ("int" | "uniform" | "loguniform", (low, high))
def model_search_space(name: str) -> Dict[str, Tuple[str, Tuple[float, float]]]:
    space: Dict[str, Tuple[str, Tuple[float, float]]] = {}
    if name == "random_forest":
        space["model__n_estimators"] = ("int", (200, 1000))
        space["model__max_depth"] = ("int", (3, 20))
        space["model__min_samples_split"] = ("int", (2, 20))
        space["model__min_samples_leaf"] = ("int", (1, 10))
    elif name == "gbrt":
        space["model__n_estimators"] = ("int", (100, 800))
        space["model__learning_rate"] = ("loguniform", (1e-3, 0.3))
        space["model__max_depth"] = ("int", (2, 6))
    elif name == "xgboost":
        space["model__n_estimators"] = ("int", (200, 800))
        space["model__max_depth"] = ("int", (3, 10))
        space["model__learning_rate"] = ("loguniform", (1e-3, 0.3))
        space["model__subsample"] = ("uniform", (0.5, 1.0))
        space["model__colsample_bytree"] = ("uniform", (0.5, 1.0))
        space["model__min_child_weight"] = ("int", (1, 20))
        space["model__reg_lambda"] = ("loguniform", (1e-3, 10.0))
    elif name == "lightgbm":
        space["model__n_estimators"] = ("int", (200, 1200))
        space["model__num_leaves"] = ("int", (16, 256))
        space["model__learning_rate"] = ("loguniform", (1e-3, 0.3))
        space["model__feature_fraction"] = ("uniform", (0.5, 1.0))
        space["model__bagging_fraction"] = ("uniform", (0.5, 1.0))
        space["model__bagging_freq"] = ("int", (0, 7))
        space["model__min_child_samples"] = ("int", (5, 50))
    return space
