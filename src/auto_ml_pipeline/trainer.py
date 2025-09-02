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
from auto_ml_pipeline.data_cleaning import (
    FeatureMissingnessDropper,
    ConstantFeatureDropper,
    NumericLikeCoercer,
)
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


def _choose_scorer_from_cfg(task: TaskType, metrics_cfg: Any | None) -> str:
    """Map cfg.eval.metrics[0] to a sklearn scorer name, fallback to default if unknown.
    Accepts common aliases like 'rmse' -> 'neg_root_mean_squared_error', 'mae' -> 'neg_mean_absolute_error'.
    """
    if not metrics_cfg:
        return _default_scorer(task)
    # take the first metric for scoring
    m = str(metrics_cfg[0]).lower()
    if task == TaskType.classification:
        if m in {"accuracy", "acc"}:
            return "accuracy"
        if m in {"f1", "f1_macro"}:
            return "f1_macro"
        if m in {"roc_auc", "auc"}:
            return "roc_auc"
    else:
        if m in {"rmse", "neg_rmse", "neg_root_mean_squared_error"}:
            return "neg_root_mean_squared_error"
        if m in {"mse", "neg_mse", "neg_mean_squared_error"}:
            return "neg_mean_squared_error"
        if m in {"mae", "neg_mae", "neg_mean_absolute_error"}:
            return "neg_mean_absolute_error"
        if m in {"r2"}:
            return "r2"
    return _default_scorer(task)


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
    steps = []
    # Robust numeric-like coercion for object columns before any feature-based droppers
    steps.append(("coerce_numeric_like", NumericLikeCoercer(threshold=0.95)))
    # Train-only droppers working on raw DataFrame before columnwise transforms
    if cfg.cleaning.feature_missing_threshold is not None:
        steps.append(
            (
                "drop_high_missing",
                FeatureMissingnessDropper(
                    threshold=cfg.cleaning.feature_missing_threshold
                ),
            )
        )
    if cfg.cleaning.remove_constant:
        steps.append(("drop_constant", ConstantFeatureDropper()))
    # Column-wise preprocessor next
    steps.append(("pre", preproc))
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
    logger.info("Task inferred/selected: %s", task.value)

    # Clean (pre-split only leakage-safe ops)
    df = clean_data(df, target, cfg.cleaning)
    logger.info(
        "Initial data after target cleaning: %s rows, %s columns",
        df.shape[0],
        df.shape[1],
    )

    # If regression but target is stored as string with separators, coerce robustly to numeric
    if task == TaskType.regression:
        y = df[target]
        if y.dtype.kind in {"O"}:
            # Reuse the same normalization heuristic as NumericLikeCoercer
            try:
                from auto_ml_pipeline.data_cleaning import (
                    NumericLikeCoercer,
                )  # local import to avoid cycles

                norm = np.vectorize(NumericLikeCoercer._normalize_number_string)
                y_norm = pd.Series(norm(y.astype(str)), index=y.index)
            except Exception:
                # Fallback: basic cleanup
                y_norm = (
                    y.astype(str)
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.replace("'", "", regex=False)
                )
                # Heuristic handling of comma/dot
                y_norm = y_norm.apply(
                    lambda s: s.replace(".", "") if s.count(".") > 1 else s
                )
                y_norm = y_norm.apply(
                    lambda s: (
                        s.replace(",", ".")
                        if ("," in s and (len(s) - s.rfind(",") - 1) in (1, 2))
                        else s.replace(",", "")
                    )
                )

            y_coerced = pd.to_numeric(y_norm, errors="coerce")
            non_na_ratio = float(y_coerced.notna().mean()) if len(y_coerced) else 0.0
            if non_na_ratio >= 0.95:
                df[target] = y_coerced
                before = len(df)
                df = df[~df[target].isna()].copy()
                dropped = before - len(df)
                if dropped:
                    logger.info(
                        "Dropped %d rows with non-numeric target after coercion",
                        dropped,
                    )

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
    logger.info(
        "Performed train/test split -> X_train: %s, X_test: %s",
        X_train.shape,
        X_test.shape,
    )

    # Train-only deduplication (row-wise) to avoid leakage
    if cfg.cleaning.remove_duplicates:
        before = len(X_train)
        df_train_tmp = pd.concat([X_train, y_train], axis=1)
        df_train_tmp = df_train_tmp.drop_duplicates()
        removed = before - len(df_train_tmp)
        if removed:
            logger.info("Train-only duplicates dropped: %d rows", removed)
        X_train, y_train = df_train_tmp.drop(columns=[target]), df_train_tmp[target]
    # Recombine train frame for downstream pipeline column inference
    df_train = pd.concat([X_train, y_train], axis=1)
    logger.info(
        "Train-only column droppers configured in pipeline: high_missing=%s, constant=%s",
        cfg.cleaning.feature_missing_threshold,
        cfg.cleaning.remove_constant,
    )

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
        logger.info(
            "After outlier handling -> X_train: %s, X_test: %s",
            X_train.shape,
            X_test.shape,
        )

    # CV + scoring
    cv = _choose_cv(task, cfg.split.n_splits, cfg.split.stratify, seed)
    scorer_name = _choose_scorer_from_cfg(task, cfg.eval.metrics)
    if cfg.eval.metrics:
        logger.info(
            "Requested metrics in config: %s; using scorer: %s",
            cfg.eval.metrics,
            scorer_name,
        )
    else:
        logger.info("No metrics configured; using default scorer: %s", scorer_name)

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
                try:
                    logger.info(
                        "Starting Optuna tuning for model %s with %d trials",
                        name,
                        cfg.optimization.n_trials,
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
                except RuntimeError as e:
                    # Handle nested Optuna invocation (e.g., when train() is called from within another study.optimize)
                    if "nested invocation" in str(e).lower():
                        logger.warning(
                            "Optuna nested optimize detected; skipping internal tuning for model %s.",
                            name,
                        )
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
                            model_best_params = {}
                        except Exception as e2:
                            logger.exception(
                                "CV failed for model %s after skipping tuning: %s",
                                name,
                                e2,
                            )
                            mean_score = -np.inf
                            model_best_params = {}
                    else:
                        raise
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
    logger.info("Fitting best model %s on full training data", best_name)
    best_estimator.fit(X_train, y_train)

    y_pred = best_estimator.predict(X_test)
    y_proba = None
    if task == TaskType.classification and hasattr(best_estimator, "predict_proba"):
        try:
            y_proba = best_estimator.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = evaluate_predictions(task, y_test, y_pred, y_proba)
    # Optionally filter metrics to those requested in cfg
    if cfg.eval.metrics:
        wanted = [str(m).lower() for m in cfg.eval.metrics]
        filtered = {}
        for k, v in metrics.items():
            if k.lower() in wanted:
                filtered[k] = v
            # allow rmse alias mapping if requested but not present
            if "rmse" in wanted and k.lower() == "rmse":
                filtered[k] = v
        # if filtering removed everything, keep original
        if filtered:
            metrics = filtered
    logger.info("Test metrics: %s", metrics)

    run_dir = make_run_dir(cfg.io.output_dir)
    logger.info("Saving artifacts to %s", run_dir)
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
