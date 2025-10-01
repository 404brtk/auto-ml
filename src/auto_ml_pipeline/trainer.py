import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import difflib

import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    train_test_split,
    BaseCrossValidator,
)
from sklearn.pipeline import Pipeline as SkPipeline


from auto_ml_pipeline.config import PipelineConfig, TaskType
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.transformers import (
    OutlierTransformer,
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
from auto_ml_pipeline.task_inference import infer_task

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Results from training process."""

    run_dir: Path
    best_model_name: str
    best_score: float
    best_estimator: Any
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    production_estimator: Optional[Any] = None  # Model retrained on full data


def get_cv_splitter(task: TaskType, n_splits: int, stratify: bool, random_state: int):
    """Get appropriate cross-validation splitter."""
    if task == TaskType.classification and stratify:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_scorer_name(task: TaskType, requested_metrics: Optional[list] = None) -> str:
    """Get sklearn scorer name based on task and requested metrics."""
    # Default scorers
    defaults = {
        TaskType.classification: "f1_macro",
        TaskType.regression: "neg_root_mean_squared_error",
    }

    if not requested_metrics:
        return defaults[task]

    # Map common metric names to sklearn scorers
    metric = str(requested_metrics[0]).lower()

    metric_mapping = {
        # Classification
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1_macro",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "auc": "roc_auc",
        # Regression
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    return metric_mapping.get(metric, defaults[task])


def suggest_hyperparameter(
    trial: optuna.Trial, param_name: str, param_spec: tuple
) -> Any:
    """Suggest hyperparameter value from Optuna trial."""
    param_type, bounds = param_spec

    if param_type == "int":
        return trial.suggest_int(param_name, int(bounds[0]), int(bounds[1]))
    elif param_type == "loguniform":
        return trial.suggest_float(
            param_name, float(bounds[0]), float(bounds[1]), log=True
        )
    elif param_type == "uniform":
        return trial.suggest_float(param_name, float(bounds[0]), float(bounds[1]))
    else:
        logger.warning("Unknown parameter type: %s", param_type)
        return None


def get_available_models(task: TaskType, random_state: int) -> Dict[str, Any]:
    """Get available models for the task."""
    if task == TaskType.classification:
        return available_models_classification(random_state=random_state, n_jobs=1)
    return available_models_regression(random_state=random_state, n_jobs=1)


def build_ml_pipeline(X: pd.DataFrame, cfg: PipelineConfig, model: Any) -> SkPipeline:
    """Build complete ML pipeline with all preprocessing steps."""
    preprocessor, _ = build_preprocessor(X, cfg.features)
    selector = build_selector(cfg.selection, cfg.task or TaskType.regression)

    steps = [
        # Main preprocessing (imputation, scaling, encoding)
        ("preprocessor", preprocessor),
    ]

    if selector is not None:
        steps.append(("feature_selection", selector))

    steps.append(("model", model))

    pipeline = SkPipeline(steps)

    return pipeline


def optimize_model(
    model_name: str,
    pipeline: SkPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Union[int, BaseCrossValidator],
    scorer_name: str,
    n_trials: int,
    timeout: Optional[int],
    random_state: int,
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    pruner_startup_trials: int = 5,
) -> tuple[float, Dict[str, Any]]:
    """Optimize model hyperparameters using Optuna."""
    search_space = model_search_space(model_name)

    if not search_space:
        # No hyperparameters to tune
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
        )
        return float(np.mean(scores)), {}

    # Clone pipeline to avoid mutation
    pipeline_copy = clone(pipeline)

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {}
        for param_name, param_spec in search_space.items():
            suggested_value = suggest_hyperparameter(trial, param_name, param_spec)
            if suggested_value is not None:
                params[f"model__{param_name}"] = suggested_value

        # Set parameters and evaluate
        pipeline_copy.set_params(**params)
        scores = cross_val_score(
            pipeline_copy, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
        )
        return float(np.mean(scores))

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Configure sampler
    sampler: optuna.samplers.BaseSampler
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=random_state)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler_name == "cmaes":
        # CMA-ES works best with continuous parameters
        # warn_independent_sampling=False to handle mixed search spaces better
        sampler = optuna.samplers.CmaEsSampler(
            seed=random_state, warn_independent_sampling=False
        )
    else:
        logger.warning("Unknown sampler %s, using TPE", sampler_name)
        sampler = optuna.samplers.TPESampler(seed=random_state)

    # Configure pruner
    pruner: optuna.pruners.BasePruner
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials)
    elif pruner_name == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource="auto", reduction_factor=4, min_early_stopping_rate=0
        )
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3
        )
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        logger.warning("Unknown pruner %s, using Median", pruner_name)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials)

    # Create study and optimize
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    try:
        logger.info("Optimizing %s with %d trials", model_name, n_trials)
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False
        )

        best_params = {
            k.replace("model__", ""): v for k, v in study.best_trial.params.items()
        }
        return float(study.best_value), best_params

    except (optuna.exceptions.OptunaError, ValueError, RuntimeError) as e:
        logger.warning(
            "Optimization failed for %s: %s. Using default parameters.", model_name, e
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
        )
        return float(np.mean(scores)), {}


def train(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> TrainResult:
    """Train AutoML pipeline and return best model."""

    # Fuzzy matching for target column
    if target not in df.columns:
        # Try case-insensitive match
        case_matches = [col for col in df.columns if col.lower() == target.lower()]
        if case_matches:
            if len(case_matches) == 1:
                logger.warning(
                    f"Target column '{target}' not found. Using case-insensitive match: '{case_matches[0]}'"
                )
                target = case_matches[0]
            else:
                logger.warning(
                    f"Multiple case-insensitive matches for '{target}': {case_matches}. Using the first: '{case_matches[0]}'"
                )
                target = case_matches[0]
        else:
            # Closest matches
            close = difflib.get_close_matches(target, df.columns, n=3, cutoff=0.6)
            if close:
                logger.warning(
                    f"Target column '{target}' not found. Closest matches: {close}"
                )
            raise ValueError(
                f"Target column '{target}' not found. Available: {list(df.columns)}"
            )

    # Set random seeds for reproducibility
    random_state = cfg.split.random_state
    np.random.seed(random_state)
    random.seed(random_state)

    # Infer task if not specified
    if cfg.task is None:
        cfg.task = infer_task(
            df,
            target,
            numeric_coercion_threshold=cfg.cleaning.numeric_coercion_threshold,
            classification_cardinality_threshold=cfg.cleaning.classification_cardinality_threshold,
        )

    task = cfg.task
    logger.info("Task: %s", task.value)

    # Clean data (pre-split operations only)
    df = clean_data(df, target, cfg.cleaning)
    logger.info("After cleaning: %d rows, %d columns", df.shape[0], df.shape[1])

    # Train/test split
    stratify_y = (
        df[target] if (task == TaskType.classification and cfg.split.stratify) else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=cfg.split.test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    logger.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    # Handle outliers
    if cfg.cleaning.outlier.strategy not in [None, "none"]:
        try:
            outlier_transformer = OutlierTransformer(
                strategy=cfg.cleaning.outlier.strategy,
                method=cfg.cleaning.outlier.method,
                iqr_multiplier=cfg.cleaning.outlier.iqr_multiplier,
                zscore_threshold=cfg.cleaning.outlier.zscore_threshold,
            )
            # Fit on training data to learn outlier boundaries
            outlier_transformer.fit(X_train)
            X_train_cleaned = outlier_transformer.transform(X_train)

            # If rows were removed (method="remove"), update y_train accordingly
            if len(X_train_cleaned) != len(X_train):
                # Use the preserved index to align y_train
                if hasattr(X_train_cleaned, "index"):
                    # Align y_train with the kept indices from X_train_cleaned
                    y_train_aligned = y_train.loc[X_train_cleaned.index]
                    y_train = (
                        y_train_aligned.reset_index(drop=True)
                        if hasattr(y_train_aligned, "reset_index")
                        else y_train_aligned
                    )
                    X_train = (
                        X_train_cleaned.reset_index(drop=True)
                        if hasattr(X_train_cleaned, "reset_index")
                        else X_train_cleaned
                    )
                else:
                    # Fallback: assume first N rows were kept
                    y_train_sliced = y_train.iloc[: len(X_train_cleaned)]
                    y_train = (
                        y_train_sliced.reset_index(drop=True)
                        if hasattr(y_train_sliced, "reset_index")
                        else y_train_sliced
                    )
                    X_train = X_train_cleaned
            else:
                X_train = X_train_cleaned

            # For clip method, apply same transformation to test set
            # For remove method, we don't transform test (can't remove test rows)
            if cfg.cleaning.outlier.method == "clip":
                X_test = outlier_transformer.transform(X_test)
                logger.info(
                    "Applied outlier clipping to test set using training boundaries"
                )
        except Exception as e:
            logger.warning("Skipping OutlierTransformer due to: %s", e)

    # Apply high-missing and constant feature droppers based on training data
    # These operate on raw DataFrames prior to column-wise preprocessing

    # Setup cross-validation and scoring
    cv = get_cv_splitter(task, cfg.split.n_splits, cfg.split.stratify, random_state)
    scorer_name = get_scorer_name(task, getattr(cfg.eval, "metrics", None))
    logger.info("Using scorer: %s", scorer_name)

    # Get available models and filter per config
    models = get_available_models(task, random_state)
    include = getattr(cfg.models, "include", None)
    exclude = getattr(cfg.models, "exclude", None)

    if include:
        models = {k: v for k, v in models.items() if k in set(include)}
    if exclude:
        models = {k: v for k, v in models.items() if k not in set(exclude)}

    if not models:
        raise ValueError(
            "No models selected to run after applying models.include/exclude"
        )
    logger.info("Models to evaluate: %s", list(models.keys()))

    best_score = -np.inf
    best_model_name = ""
    best_pipeline = None
    best_params = {}

    for model_name, base_model in models.items():
        logger.info("Evaluating model: %s", model_name)

        # Build pipeline
        pipeline = build_ml_pipeline(X_train, cfg, base_model)

        try:
            if cfg.optimization.enabled:
                # Hyperparameter optimization
                score, params = optimize_model(
                    model_name,
                    pipeline,
                    X_train,
                    y_train,
                    cv,
                    scorer_name,
                    cfg.optimization.n_trials,
                    cfg.optimization.timeout,
                    random_state,
                    cfg.optimization.sampler,
                    cfg.optimization.pruner,
                    cfg.optimization.pruner_startup_trials,
                )
            else:
                # No optimization - just cross-validate
                scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
                )
                score = float(np.mean(scores))
                params = {}

            logger.info("Model %s CV score: %.4f", model_name, score)

            # Track best model
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline
                best_params = params

        except Exception as e:
            logger.exception("Failed to evaluate model %s: %s", model_name, e)
            continue

    if best_pipeline is None:
        # Fallback: try a simple baseline model to avoid hard failure on edge cases
        logger.warning(
            "No models succeeded during CV; falling back to a baseline model"
        )
        try:
            baseline_models = get_available_models(task, random_state)
            # Prefer random_forest if available, else take any
            baseline_key = (
                "random_forest"
                if "random_forest" in baseline_models
                else next(iter(baseline_models.keys()))
            )
            baseline_model = baseline_models[baseline_key]
            best_model_name = baseline_key
            best_pipeline = build_ml_pipeline(X_train, cfg, baseline_model)
            best_params = {}
        except Exception as e:
            logger.exception("Baseline model construction failed: %s", e)
            raise RuntimeError("No models could be successfully trained")

    # Fit best model on full training data
    logger.info("Training best model: %s", best_model_name)
    if best_params:
        param_dict = {f"model__{k}": v for k, v in best_params.items()}
        best_pipeline.set_params(**param_dict)

    best_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    y_proba = None

    if task == TaskType.classification and hasattr(best_pipeline, "predict_proba"):
        try:
            y_proba = best_pipeline.predict_proba(X_test)
        except Exception:
            pass

    # Calculate metrics
    metrics = evaluate_predictions(task, y_test, y_pred, y_proba)
    logger.info("Test metrics: %s", metrics)

    # Optionally retrain on full dataset
    production_estimator = None
    if cfg.optimization.retrain_on_full_data:
        logger.info("Retraining best model on full dataset for production use")
        try:
            # Combine train and test data
            X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
            y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

            # Create new pipeline instance with same parameters
            production_pipeline = build_ml_pipeline(
                X_full, cfg, models[best_model_name]
            )

            # Apply best parameters
            if best_params:
                param_dict = {f"model__{k}": v for k, v in best_params.items()}
                production_pipeline.set_params(**param_dict)

            # Fit on full data
            production_pipeline.fit(X_full, y_full)
            production_estimator = production_pipeline
            logger.info(
                "Production model trained on full dataset (%d samples)", len(X_full)
            )
        except Exception as e:
            logger.warning(
                "Failed to retrain on full data: %s. Using evaluation model.", e
            )

    # Save results
    run_dir = make_run_dir(cfg.io.output_dir)
    logger.info("Saving results to: %s", run_dir)

    save_model(best_pipeline, run_dir / "eval_model.joblib")
    if production_estimator is not None:
        save_model(production_estimator, run_dir / "production_model.joblib")
        logger.info("Saved both evaluation and production models")
    save_json(
        {
            "task": task.value,
            "best_model": best_model_name,
            "best_cv_score": best_score,
            "test_metrics": metrics,
            "best_params": best_params,
            "random_state": random_state,
            "cv_folds": cfg.split.n_splits,
            "scorer": scorer_name,
            "retrained_on_full_data": cfg.optimization.retrain_on_full_data,
            "production_model_available": production_estimator is not None,
            "eval_model_file": "eval_model.joblib",
            "production_model_file": (
                "production_model.joblib" if production_estimator is not None else None
            ),
        },
        run_dir / "results.json",
    )

    return TrainResult(
        run_dir=run_dir,
        best_model_name=best_model_name,
        best_score=best_score,
        best_estimator=best_pipeline,
        metrics=metrics,
        best_params=best_params,
        production_estimator=production_estimator,
    )
