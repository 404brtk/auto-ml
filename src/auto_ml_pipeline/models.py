from typing import Any, Dict, Tuple, cast

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# Optional dependencies
_HAS_XGB = True
_HAS_LGBM = True

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    _HAS_LGBM = False


def available_models_classification(
    random_state: int = 42, n_jobs: int = 1
) -> Dict[str, Any]:
    """Get available classification models."""
    models = {
        "random_forest": RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
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
    """Get available regression models."""
    models = {
        "random_forest": RandomForestRegressor(
            n_jobs=n_jobs, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
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


ParamDist = Tuple[str, Tuple[float, float]]
HyperparamSpace = Dict[str, ParamDist]


def model_search_space(model_name: str) -> HyperparamSpace:
    """Get hyperparameter search space for a model."""
    search_spaces: Dict[str, HyperparamSpace] = {
        "random_forest": {
            "n_estimators": ("int", (100, 500)),
            "max_depth": ("int", (3, 15)),
            "min_samples_split": ("int", (2, 10)),
            "min_samples_leaf": ("int", (1, 5)),
        },
        "gradient_boosting": {
            "n_estimators": ("int", (100, 300)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "max_depth": ("int", (3, 8)),
        },
        "xgboost": {
            "n_estimators": ("int", (100, 500)),
            "max_depth": ("int", (3, 10)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "subsample": ("uniform", (0.7, 1.0)),
            "colsample_bytree": ("uniform", (0.7, 1.0)),
        },
        "lightgbm": {
            "n_estimators": ("int", (100, 500)),
            "num_leaves": ("int", (20, 100)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "feature_fraction": ("uniform", (0.7, 1.0)),
            "bagging_fraction": ("uniform", (0.7, 1.0)),
        },
    }

    return search_spaces.get(model_name, cast(HyperparamSpace, {}))
