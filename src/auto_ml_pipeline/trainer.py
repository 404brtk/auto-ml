import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline

from auto_ml_pipeline.config import PipelineConfig, TaskType
from auto_ml_pipeline.data_cleaning import (
    clean_data,
    fit_outlier_params,
    apply_outliers,
)
from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.feature_selection import build_selector
from auto_ml_pipeline.models import (
    available_models_classification,
    available_models_regression,
    model_search_space,
)
from auto_ml_pipeline.evaluation import evaluate_predictions
from auto_ml_pipeline.io_utils import make_run_dir, save_json, save_model
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    run_dir: Path
    best_model_name: str
    best_score: float
    best_estimator: Any
    metrics: Dict[str, float]
    best_params: Dict[str, Any]


def _choose_cv(task: TaskType, n_splits: int, stratify: bool, random_state: int):
    if task == TaskType.classification and stratify:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _default_scorer(task: TaskType) -> str:
    return (
        "f1_macro" if task == TaskType.classification else "neg_root_mean_squared_error"
    )


def _suggest(trial: optuna.Trial, key: str, spec: Tuple[str, Any]):
    kind, params = spec
    if kind == "int":
        low, high = params
        return trial.suggest_int(key, int(low), int(high))
    if kind == "loguniform":
        low, high = params
        return trial.suggest_float(key, float(low), float(high), log=True)
    if kind == "uniform":
        low, high = params
        return trial.suggest_float(key, float(low), float(high))
    return None


def _get_models(task: TaskType, random_state: int) -> Dict[str, Any]:
    # n_jobs=1 inside estimators to avoid nested parallelism
    if task == TaskType.classification:
        return available_models_classification(random_state=random_state, n_jobs=1)
    return available_models_regression(random_state=random_state, n_jobs=1)


def _build_pipeline(
    df: pd.DataFrame, target: str, cfg: PipelineConfig, model: Any
) -> SkPipeline:
    preproc, _ = build_preprocessor(df, target, cfg.features)
    selector = build_selector(cfg.selection, cfg.task or TaskType.regression)
    steps = [("pre", preproc)]
    if selector is not None:
        steps.append(("sel", selector))
    steps.append(("model", model))
    return SkPipeline(steps)


def train(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> TrainResult:
    # Seed everything for reproducibility
    seed = cfg.split.random_state
    np.random.seed(seed)
    random.seed(seed)

    # Task
    if cfg.task is None:
        from .task_inference import infer_task

        cfg.task = infer_task(df, target)
    task = cfg.task

    # Clean (pre-split only leakage-safe ops)
    df = clean_data(df, target, cfg.cleaning)

    # Split holdout test
    if task == TaskType.classification and cfg.split.stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[target]),
            df[target],
            test_size=cfg.split.test_size,
            random_state=seed,
            stratify=df[target],
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[target]),
            df[target],
            test_size=cfg.split.test_size,
            random_state=seed,
        )
    df_train = pd.concat([X_train, y_train], axis=1)

    # Outliers: fit on train, apply as configured (default: clip, train_only)
    strategy = (cfg.cleaning.outlier_strategy or "").lower()
    if strategy not in {None, "", "none"}:
        params = fit_outlier_params(df_train, target, cfg.cleaning)
        df_train = apply_outliers(df_train, target, cfg.cleaning, params, scope="train")
        X_train, y_train = df_train.drop(columns=[target]), df_train[target]
        if (cfg.cleaning.outlier_apply_scope or "train_only").lower() == "both":
            df_test = pd.concat([X_test, y_test], axis=1)
            df_test = apply_outliers(
                df_test, target, cfg.cleaning, params, scope="test"
            )
            X_test, y_test = df_test.drop(columns=[target]), df_test[target]

    # CV + scoring
    cv = _choose_cv(task, cfg.split.n_splits, cfg.split.stratify, seed)
    scorer_name = _default_scorer(task)

    models = _get_models(task, seed)

    best_score = -np.inf
    best_name = ""
    best_estimator: Any = None
    best_params: Dict[str, Any] = {}

    # Parallelize across folds; estimators use n_jobs=1
    cv_n_jobs = -1

    for name, base_model in models.items():
        pipe = _build_pipeline(df_train, target, cfg, base_model)

        mean_score: float
        model_best_params: Dict[str, Any] = {}

        if cfg.optimization.enabled:
            space = model_search_space(name)
            if not space:
                # No hyperparameters to tune for this model
                try:
                    scores = cross_val_score(
                        pipe,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scorer_name,
                        n_jobs=cv_n_jobs,
                        error_score="raise",
                    )
                    mean_score = float(np.nanmean(scores))
                except Exception as e:
                    logger.exception("CV failed for model %s: %s", name, e)
                    mean_score = -np.inf
            else:

                def objective(trial: optuna.Trial) -> float:
                    params = {}
                    for k, spec in space.items():
                        params[k] = _suggest(trial, k, spec)
                    pipe.set_params(
                        **{k: v for k, v in params.items() if v is not None}
                    )
                    # Fail fast on errors; better than silently returning NaNs
                    scores = cross_val_score(
                        pipe,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scorer_name,
                        n_jobs=cv_n_jobs,
                        error_score="raise",
                    )
                    return float(np.mean(scores))

                sampler = optuna.samplers.TPESampler(seed=seed)
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=min(5, max(1, cfg.optimization.n_trials // 5))
                )
                study = optuna.create_study(
                    direction="maximize", sampler=sampler, pruner=pruner
                )
                study.optimize(
                    objective,
                    n_trials=cfg.optimization.n_trials,
                    timeout=cfg.optimization.timeout,
                    show_progress_bar=False,
                )
                model_best_params = study.best_trial.params
                pipe.set_params(**model_best_params)
                mean_score = float(study.best_value)
        else:
            try:
                scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scorer_name,
                    n_jobs=cv_n_jobs,
                    error_score="raise",
                )
                mean_score = float(np.mean(scores))
            except Exception as e:
                logger.exception("CV failed for model %s: %s", name, e)
                mean_score = -np.inf

        logger.info("Model %s CV %s: %.5f", name, scorer_name, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_estimator = pipe
            best_params = model_best_params

    # Fit best on train and evaluate on test
    assert best_estimator is not None
    best_estimator.fit(X_train, y_train)

    y_pred = best_estimator.predict(X_test)
    y_proba = None
    if task == TaskType.classification and hasattr(best_estimator, "predict_proba"):
        try:
            y_proba = best_estimator.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = evaluate_predictions(task, y_test, y_pred, y_proba)

    run_dir = make_run_dir(cfg.io.output_dir)
    save_model(best_estimator, run_dir / "model.joblib")
    save_json(
        {
            "task": task.value,
            "best_model": best_name,
            "best_cv_score": best_score,
            "metrics": metrics,
            "best_params": best_params,
            "seed": seed,
            "cv_n_splits": cfg.split.n_splits,
            "scorer": scorer_name,
        },
        run_dir / "report.json",
    )

    return TrainResult(
        run_dir=run_dir,
        best_model_name=best_name,
        best_score=best_score,
        best_estimator=best_estimator,
        metrics=metrics,
        best_params=best_params,
    )
