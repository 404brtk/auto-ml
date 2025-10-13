from typing import Any, Dict, Tuple, Union, cast

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    RidgeClassifier,
    Lasso,
    ElasticNet,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Optional dependencies
_HAS_XGB = True
_HAS_LGBM = True
_HAS_CATBOOST = True

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    _HAS_CATBOOST = False


def available_models_classification(
    random_state: int = 42, n_jobs: int = 1
) -> Dict[str, Any]:
    models = {
        "logistic": LogisticRegression(random_state=random_state, n_jobs=n_jobs),
        "ridge": RidgeClassifier(random_state=random_state),
        "sgd": SGDClassifier(random_state=random_state, n_jobs=n_jobs, tol=1e-3),
        "naive_bayes": GaussianNB(),
        "svm": SVC(random_state=random_state),
        "knn": KNeighborsClassifier(n_jobs=n_jobs),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state
        ),
        "extra_trees": ExtraTreesClassifier(n_jobs=n_jobs, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state
        ),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        "mlp": MLPClassifier(random_state=random_state, early_stopping=True),
    }

    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )

    if _HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )

    if _HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )

    return models


def available_models_regression(
    random_state: int = 42, n_jobs: int = 1
) -> Dict[str, Any]:
    models = {
        "linear": LinearRegression(n_jobs=n_jobs),
        "ridge": Ridge(random_state=random_state),
        "lasso": Lasso(random_state=random_state),
        "elastic_net": ElasticNet(random_state=random_state),
        "sgd": SGDRegressor(random_state=random_state, tol=1e-3),
        "svm": SVR(),
        "knn": KNeighborsRegressor(n_jobs=n_jobs),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_jobs=n_jobs, random_state=random_state
        ),
        "extra_trees": ExtraTreesRegressor(n_jobs=n_jobs, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=random_state
        ),
        "adaboost": AdaBoostRegressor(random_state=random_state),
        "mlp": MLPRegressor(random_state=random_state, early_stopping=True),
    }

    if _HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )

    if _HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )

    if _HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
        )

    return models


ParamBounds = Union[Tuple[float, float], Tuple[int, int], Tuple[Any, ...]]
ParamDist = Tuple[str, ParamBounds]
HyperparamSpace = Dict[str, ParamDist]


def model_search_space(model_name: str) -> HyperparamSpace:
    search_spaces: Dict[str, HyperparamSpace] = {
        "logistic": {  # Classification only
            "C": ("loguniform", (1e-4, 100.0)),
            # incompatible penalty-solver combinations possible
            # caution required when specifying search space
            "penalty": ("categorical", ("l2",)),
            "solver": ("categorical", ("lbfgs", "liblinear", "saga")),
            "max_iter": ("int", (100, 5000)),
        },
        "linear": {},  # Regression only, no hyperparameters to tune
        "lasso": {  # Regression only
            "alpha": ("loguniform", (1e-4, 10.0)),
            "max_iter": ("int", (2000, 10000)),
        },
        "elastic_net": {  # Regression only
            "alpha": ("loguniform", (1e-4, 10.0)),
            "l1_ratio": ("uniform", (0.1, 1.0)),
            "max_iter": ("int", (2000, 10000)),
        },
        "naive_bayes": {  # Classification only
            "var_smoothing": ("loguniform", (1e-12, 1e-5)),
        },
        # Both classification and regression
        "ridge": {
            "alpha": ("loguniform", (1e-4, 10.0)),
            "solver": ("categorical", ("auto", "svd", "cholesky", "lsqr", "saga")),
        },
        "sgd": {
            "alpha": ("loguniform", (1e-5, 1e-1)),
            "penalty": ("categorical", ("l2", "elasticnet")),
            "l1_ratio": ("uniform", (0.0, 1.0)),
            "learning_rate": ("categorical", ("constant", "optimal", "adaptive")),
            "eta0": ("loguniform", (1e-4, 1.0)),
            "max_iter": ("int", (500, 3000)),
        },
        "decision_tree": {
            "max_depth": ("int", (3, 20)),
            "min_samples_split": ("int", (2, 20)),
            "min_samples_leaf": ("int", (1, 10)),
            "max_features": ("uniform", (0.3, 1.0)),
        },
        "random_forest": {
            "n_estimators": ("int", (100, 500)),
            "max_depth": ("int", (4, 20)),
            "min_samples_split": ("int", (2, 10)),
            "min_samples_leaf": ("int", (1, 5)),
            "max_features": ("uniform", (0.3, 1.0)),
            "bootstrap": ("categorical", (True, False)),
        },
        "extra_trees": {
            "n_estimators": ("int", (100, 500)),
            "max_depth": ("int", (4, 20)),
            "min_samples_split": ("int", (2, 10)),
            "min_samples_leaf": ("int", (1, 5)),
            "max_features": ("uniform", (0.3, 1.0)),
            "bootstrap": ("categorical", (True, False)),
        },
        "gradient_boosting": {
            "n_estimators": ("int", (100, 500)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "max_depth": ("int", (3, 8)),
            "min_samples_split": ("int", (2, 10)),
            "min_samples_leaf": ("int", (1, 5)),
            "subsample": ("uniform", (0.7, 1.0)),
        },
        "hist_gradient_boosting": {
            "max_iter": ("int", (100, 500)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "max_depth": ("int", (3, 10)),
            "min_samples_leaf": ("int", (5, 50)),
            "l2_regularization": ("loguniform", (1e-10, 1.0)),
        },
        "adaboost": {
            "n_estimators": ("int", (50, 500)),
            "learning_rate": ("loguniform", (0.01, 2.0)),
        },
        "xgboost": {
            "n_estimators": ("int", (100, 500)),
            "max_depth": ("int", (3, 10)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "subsample": ("uniform", (0.7, 1.0)),
            "colsample_bytree": ("uniform", (0.7, 1.0)),
            "min_child_weight": ("int", (1, 7)),
            "gamma": ("loguniform", (1e-8, 1.0)),
            "reg_alpha": ("loguniform", (1e-8, 1.0)),
            "reg_lambda": ("loguniform", (1e-8, 10.0)),
            "tree_method": ("categorical", ("auto", "hist")),
        },
        "lightgbm": {
            "n_estimators": ("int", (100, 500)),
            "num_leaves": ("int", (20, 150)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "feature_fraction": ("uniform", (0.7, 1.0)),
            "bagging_fraction": ("uniform", (0.7, 1.0)),
            "bagging_freq": ("int", (1, 7)),
            "min_child_samples": ("int", (5, 50)),
            "reg_alpha": ("loguniform", (1e-8, 10.0)),
            "reg_lambda": ("loguniform", (1e-8, 10.0)),
        },
        "catboost": {
            "iterations": ("int", (100, 500)),
            "depth": ("int", (4, 10)),
            "learning_rate": ("loguniform", (0.01, 0.3)),
            "l2_leaf_reg": ("loguniform", (1.0, 10.0)),
            "random_strength": ("loguniform", (1e-9, 10.0)),
            "bagging_temperature": ("uniform", (0.0, 1.0)),
            "border_count": ("int", (32, 255)),
        },
        "mlp": {
            "hidden_layer_sizes": (
                "categorical",
                (
                    "64",
                    "128",
                    "256",
                    "128,64",
                    "256,128",
                    "200,100",
                    "128,128",
                    "100,100",
                    "128,64,32",
                    "256,128,64",
                    "100,100,100",
                ),
            ),
            "activation": ("categorical", ("relu", "tanh")),
            "solver": ("categorical", ("adam",)),
            "alpha": ("loguniform", (1e-6, 1e-2)),
            "learning_rate": ("categorical", ("constant", "adaptive")),
            "learning_rate_init": ("loguniform", (0.01, 0.3)),
            "batch_size": ("categorical", ("auto", 64, 128, 256)),
            "max_iter": ("int", (3000, 5000)),
            "validation_fraction": ("uniform", (0.1, 0.2)),
            "n_iter_no_change": ("int", (10, 25)),
            "tol": ("loguniform", (1e-4, 1e-2)),
        },
        "svm": {
            "C": ("loguniform", (0.1, 100.0)),
            "kernel": ("categorical", ("linear", "rbf", "poly")),
            "gamma": ("categorical", ("scale", "auto")),
            "degree": ("int", (2, 5)),
            "max_iter": ("int", (20000, 50000)),
        },
        "knn": {
            "n_neighbors": ("int", (3, 30)),
            "weights": ("categorical", ("uniform", "distance")),
            "p": ("int", (1, 3)),
            "leaf_size": ("int", (10, 50)),
        },
    }

    return search_spaces.get(model_name, cast(HyperparamSpace, {}))
