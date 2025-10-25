import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import difflib
import time

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
from sklearn.preprocessing import LabelEncoder

from auto_ml_pipeline.config import PipelineConfig, TaskType
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.transformers import OutlierTransformer
from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.feature_selection import build_selector
from auto_ml_pipeline.models import (
    available_models_classification,
    available_models_regression,
    model_search_space,
)
from auto_ml_pipeline.metrics import (
    compute_metrics,
    get_optimization_metric,
    get_optimization_metric_from_config,
    get_evaluation_metrics_from_config,
)
from auto_ml_pipeline.io_utils import make_run_dir, save_json, save_model
from auto_ml_pipeline.logging_utils import get_logger
from auto_ml_pipeline.task_inference import infer_task
from auto_ml_pipeline.reporting import ReportGenerator

# suppress lightgbm feature names mismatch warning
# pipeline transforms df to numpy arrays, triggering this warning
# safe to ignore
import warnings

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

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


def parse_hidden_layer_sizes(value: Any) -> Any:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return value


def get_cv_splitter(task: TaskType, n_splits: int, stratify: bool, random_state: int):
    """Get appropriate cross-validation splitter."""
    if task == TaskType.classification and stratify:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


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
    elif param_type == "categorical":
        choices = list(bounds) if isinstance(bounds, tuple) else bounds
        return trial.suggest_categorical(param_name, choices)
    else:
        logger.warning("Unknown parameter type: %s", param_type)
        return None


def get_available_models(task: TaskType, random_state: int) -> Dict[str, Any]:
    """Get available models for the task."""
    if task == TaskType.classification:
        return available_models_classification(random_state=random_state, n_jobs=1)
    return available_models_regression(random_state=random_state, n_jobs=1)


def get_models_from_cfg(
    cfg: PipelineConfig, task: TaskType, random_state: int
) -> Dict[str, Any]:
    """Get filtered models based on configuration."""
    all_models = get_available_models(task, random_state)

    models_list = getattr(cfg.models, "models", None)

    if not models_list:
        logger.info("No specific models configured. Using all available models.")
        return all_models

    models = {k: v for k, v in all_models.items() if k in set(models_list)}

    requested = set(models_list)
    available = set(all_models.keys())
    invalid = requested - available

    if invalid:
        if models:
            logger.warning(
                f"Some requested models are not available for {task.value}: {sorted(invalid)}. "
                f"Proceeding with valid models: {sorted(models.keys())}"
            )
        else:
            raise ValueError(
                f"No valid models found for {task.value}. "
                f"Invalid models: {sorted(invalid)}. "
                f"Available models: {sorted(available)}"
            )

    return models


def build_ml_pipeline(X: pd.DataFrame, cfg: PipelineConfig, model: Any) -> SkPipeline:
    """Build complete ML pipeline with all preprocessing steps."""
    task = cfg.task or TaskType.regression
    preprocessor, _ = build_preprocessor(X, cfg.features, task)
    selector = build_selector(cfg.selection, task)

    # Main preprocessing (imputation, scaling, encoding, etc.)
    steps = [("preprocessor", preprocessor)]

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
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    pruner_startup_trials: int = 5,
    optuna_random_seed: Optional[int] = None,
    fixed_hyperparams: Optional[Dict[str, Any]] = None,
) -> tuple[float, Dict[str, Any]]:
    """Optimize model hyperparameters using Optuna."""
    search_space = model_search_space(model_name)

    if fixed_hyperparams:
        for param in fixed_hyperparams.keys():
            search_space.pop(param, None)
        logger.info(
            f"Using fixed hyperparameters for {model_name}: {fixed_hyperparams}"
        )
    if not search_space:
        # No hyperparameters to tune
        try:
            logger.warning(
                f"No search space defined for {model_name}. "
                "Will evaluate with default hyperparameters only."
            )
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
            )
            return float(np.mean(scores)), {}
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate model {model_name} with default parameters: {e}"
            ) from e

    base_pipeline = clone(pipeline)

    if fixed_hyperparams:
        fixed_pipeline_params = {f"model__{k}": v for k, v in fixed_hyperparams.items()}
        base_pipeline.set_params(**fixed_pipeline_params)

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {}
        for param_name, param_spec in search_space.items():
            suggested_value = suggest_hyperparameter(trial, param_name, param_spec)
            if suggested_value is not None:
                if param_name == "hidden_layer_sizes":
                    suggested_value = parse_hidden_layer_sizes(suggested_value)
                params[f"model__{param_name}"] = suggested_value

        base_pipeline.set_params(**params)

        try:
            scores = cross_val_score(
                base_pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
            )
            return float(np.mean(scores))
        except Exception as e:
            # Log trial failure but let Optuna continue
            logger.debug(f"Trial failed with params {params}: {e}")
            # Return very poor score to mark this trial as bad
            return -np.inf

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Configure sampler
    sampler: optuna.samplers.BaseSampler
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=optuna_random_seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=optuna_random_seed)
    elif sampler_name == "cmaes":
        # CMA-ES works best with continuous parameters
        # warn_independent_sampling=False to handle mixed search spaces better
        sampler = optuna.samplers.CmaEsSampler(
            seed=optuna_random_seed, warn_independent_sampling=False
        )
    else:
        logger.warning("Unknown sampler %s, using TPE", sampler_name)
        sampler = optuna.samplers.TPESampler(seed=optuna_random_seed)

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

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    try:
        logger.info(
            "Optimizing %s with %d trials (timeout=%s)", model_name, n_trials, timeout
        )
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False
        )

        # Check if any trials succeeded
        if study.best_trial.value == -np.inf:
            raise ValueError(f"All optimization trials failed for {model_name}")

        best_params = {
            k.replace("model__", ""): v for k, v in study.best_trial.params.items()
        }

        if "hidden_layer_sizes" in best_params:
            best_params["hidden_layer_sizes"] = parse_hidden_layer_sizes(
                best_params["hidden_layer_sizes"]
            )

        if fixed_hyperparams:
            best_params.update(fixed_hyperparams)

        logger.info(
            "Optimization complete: best score = %.4f after %d trials",
            study.best_value,
            len(study.trials),
        )

        return float(study.best_value), best_params

    except optuna.exceptions.OptunaError as e:
        logger.error("Optuna optimization failed for %s: %s", model_name, e)
        raise RuntimeError(
            f"Hyperparameter optimization failed for {model_name}"
        ) from e

    except ValueError as e:
        logger.error("All trials failed for %s: %s", model_name, e)
        raise

    except Exception as e:
        logger.exception("Unexpected error during optimization of %s", model_name)
        raise RuntimeError(
            f"Unexpected error during hyperparameter optimization: {e}"
        ) from e


def train(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> TrainResult:
    """Train auto-ml pipeline and return best model."""
    training_start = time.time()

    # Validate and match target column
    if target not in df.columns:
        case_matches = [col for col in df.columns if col.lower() == target.lower()]
        if case_matches:
            logger.warning(
                "Target column '%s' not found. Using case-insensitive match: '%s'",
                target,
                case_matches[0],
            )
            target = case_matches[0]
        else:
            close = difflib.get_close_matches(target, df.columns, n=3, cutoff=0.6)
            error_msg = f"Target column '{target}' not found in DataFrame."
            if close:
                error_msg += f" Did you mean one of these: {close}?"
            error_msg += f" Available columns: {list(df.columns)}"
            raise ValueError(error_msg)

    # Set random seeds for reproducibility
    random_state = cfg.split.random_state
    np.random.seed(random_state)
    random.seed(random_state)

    # Clean data (pre-split operations only)
    df, target, name_mapping_full = clean_data(df, target, cfg.cleaning)
    logger.info("After cleaning: %d rows, %d columns", df.shape[0], df.shape[1])

    feature_columns = [col for col in df.columns if col != target]
    name_mapping = {
        original: std
        for original, std in name_mapping_full.items()
        if std in feature_columns
    }
    logger.info(f"Feature mapping (excluding target): {list(name_mapping.keys())}")

    # Infer task if not specified
    if cfg.task is None:
        cfg.task = infer_task(
            df,
            target,
            uniqueness_ratio_threshold=cfg.cleaning.uniqueness_ratio_threshold,
            max_categories_absolute=cfg.cleaning.max_categories_absolute,
        )

    task = cfg.task
    logger.info("Task: %s", task.value)

    # Train/test split
    stratify_y = (
        df[target] if (task == TaskType.classification and cfg.split.stratify) else None
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[target]),
            df[target],
            test_size=cfg.split.test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
    except ValueError as e:
        raise ValueError(
            f"Train/test split failed. This may be due to insufficient samples "
            f"or class imbalance. Details: {e}"
        ) from e

    logger.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    # Encode target variable if non-numeric for classification
    if task == TaskType.classification and not pd.api.types.is_numeric_dtype(y_train):
        logger.info("Encoding non-numeric target: %s", sorted(y_train.unique()))

        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)

        logger.info(
            "Encoded to: %s", {cls: i for i, cls in enumerate(label_encoder.classes_)}
        )
    else:
        label_encoder = None

    # Handle outliers
    if cfg.cleaning.outlier.strategy not in [None, "none"]:
        try:
            outlier_transformer = OutlierTransformer(
                strategy=cfg.cleaning.outlier.strategy,
                method=cfg.cleaning.outlier.method,
                iqr_multiplier=cfg.cleaning.outlier.iqr_multiplier,
                zscore_threshold=cfg.cleaning.outlier.zscore_threshold,
                contamination=cfg.cleaning.outlier.contamination,
                n_estimators=cfg.cleaning.outlier.n_estimators,
                max_samples=cfg.cleaning.outlier.max_samples,
                random_state=cfg.cleaning.outlier.random_state,
            )
            # Fit on training data to learn outlier boundaries
            outlier_transformer.fit(X_train)
            X_train_cleaned = outlier_transformer.transform(X_train)

            # If rows were removed (method="remove"), update y_train accordingly
            if len(X_train_cleaned) != len(X_train):
                if hasattr(X_train_cleaned, "index") and isinstance(
                    X_train_cleaned, pd.DataFrame
                ):
                    # X_train_cleaned is a DataFrame - align y_train with kept indices
                    y_train = y_train.loc[X_train_cleaned.index].reset_index(drop=True)
                    X_train = X_train_cleaned.reset_index(drop=True)
                else:
                    # X_train_cleaned is a numpy array - slice y_train
                    y_train_sliced = y_train.iloc[: len(X_train_cleaned)]

                    # Ensure it's a Series before calling reset_index
                    if isinstance(y_train_sliced, pd.Series):
                        y_train = y_train_sliced.reset_index(drop=True)
                    else:
                        # Fallback: convert to Series
                        y_train = pd.Series(y_train_sliced).reset_index(drop=True)

                    # Convert X_train_cleaned to DataFrame if original was DataFrame
                    if isinstance(X_train, pd.DataFrame):
                        X_train = pd.DataFrame(
                            X_train_cleaned, columns=X_train.columns
                        ).reset_index(drop=True)
                    else:
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

    # Setup cross-validation and scoring
    cv = get_cv_splitter(task, cfg.split.n_splits, cfg.split.stratify, random_state)
    scorer_name = get_optimization_metric(
        task, get_optimization_metric_from_config(task, cfg.metrics)
    )
    logger.info("Using scorer: %s", scorer_name)

    # Get models
    models = get_models_from_cfg(cfg, task, random_state)

    logger.info("Models to evaluate: %s", list(models.keys()))

    if not cfg.optimization.enabled:
        logger.info(
            "Optimization disabled - evaluating all models with default hyperparameters"
        )

    # Model evaluation loop
    best_score = -np.inf
    best_model_name = ""
    best_pipeline = None
    best_params = {}
    failed_models = []
    all_model_scores: List[Dict[str, Any]] = []

    for model_name, base_model in models.items():
        logger.info("Evaluating model: %s", model_name)

        pipeline = build_ml_pipeline(X_train, cfg, base_model)

        fixed_hyperparams = None
        if (
            cfg.models.fixed_hyperparameters
            and model_name in cfg.models.fixed_hyperparameters
        ):
            fixed_hyperparams = cfg.models.fixed_hyperparameters[model_name]
            # set fixed params on base model before pipeline creation
            param_dict = {f"model__{k}": v for k, v in fixed_hyperparams.items()}
            pipeline.set_params(**param_dict)

        try:
            if cfg.optimization.enabled:
                score, params = optimize_model(
                    model_name=model_name,
                    pipeline=pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    cv=cv,
                    scorer_name=scorer_name,
                    n_trials=cfg.optimization.n_trials,
                    timeout=cfg.optimization.timeout,
                    sampler_name=cfg.optimization.sampler,
                    pruner_name=cfg.optimization.pruner,
                    pruner_startup_trials=cfg.optimization.pruner_startup_trials,
                    optuna_random_seed=cfg.optimization.optuna_random_seed,
                    fixed_hyperparams=fixed_hyperparams,
                )
            else:
                # No optimization - just cross-validate with default hyperparameters
                # or with fixed hyperparameters if provided
                if fixed_hyperparams:
                    logger.info(
                        f"Using fixed hyperparameters for {model_name}: {fixed_hyperparams}"
                    )
                    param_dict = {
                        f"model__{k}": v for k, v in fixed_hyperparams.items()
                    }
                    pipeline.set_params(**param_dict)

                scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv, scoring=scorer_name, n_jobs=-1
                )
                score = float(np.mean(scores))
                params = fixed_hyperparams if fixed_hyperparams else {}

                logger.info("Model %s CV score: %.4f", model_name, score)

            all_model_scores.append({"model_name": model_name, "cv_score": score})

            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline
                best_params = params

        except Exception as e:
            logger.exception("Failed to evaluate model %s: %s", model_name, e)
            failed_models.append(model_name)
            continue

    # Complete failure scenario
    if best_pipeline is None:
        error_details = (
            f"All {len(models)} models failed during cross-validation. "
            f"Failed models: {failed_models}. "
            "This indicates a fundamental issue. Please check: "
            "(1) Data quality and sufficient samples, "
            "(2) Feature types and missing values, "
            "(3) Target variable distribution, "
            "(4) Pipeline configuration compatibility."
        )
        logger.error(error_details)
        raise RuntimeError(error_details)

    # Fit best model on training data
    logger.info("Training best model: %s with score: %.4f", best_model_name, best_score)

    if best_params:
        param_dict = {f"model__{k}": v for k, v in best_params.items()}
        best_pipeline.set_params(**param_dict)

    try:
        best_pipeline.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fit best model '{best_model_name}' on training data: {e}"
        ) from e

    # Evaluate on test set
    try:
        y_pred = best_pipeline.predict(X_test)
        y_proba = None

        if task == TaskType.classification and hasattr(best_pipeline, "predict_proba"):
            try:
                y_proba = best_pipeline.predict_proba(X_test)
            except Exception as proba_error:
                logger.warning("Could not get probability predictions: %s", proba_error)

        metrics = compute_metrics(
            task,
            y_test,
            y_pred,
            y_proba,
            metric_names=get_evaluation_metrics_from_config(task, cfg.metrics),
        )
        logger.info("Test metrics: %s", metrics)

    except Exception as e:
        raise RuntimeError(f"Failed to evaluate model on test set: {e}") from e

    # Optionally retrain on full dataset for production
    production_estimator = None

    if cfg.optimization.retrain_on_full_data:
        logger.info("Retraining best model on full dataset for production use")
        try:
            X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
            y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

            # Create new pipeline instance with same parameters
            production_pipeline = build_ml_pipeline(
                X_full, cfg, models[best_model_name]
            )

            if best_params:
                param_dict = {f"model__{k}": v for k, v in best_params.items()}
                production_pipeline.set_params(**param_dict)

            production_pipeline.fit(X_full, y_full)
            production_estimator = production_pipeline
            logger.info(
                "Production model trained on full dataset (%d samples)", len(X_full)
            )

        except Exception as e:
            logger.warning(
                "Failed to retrain on full data: %s. "
                "Using evaluation model as production model.",
                e,
            )
            production_estimator = best_pipeline

    # Save results
    run_dir = make_run_dir(cfg.io.output_dir)
    logger.info("Saving results to: %s", run_dir)

    result = TrainResult(
        run_dir=run_dir,
        best_model_name=best_model_name,
        best_score=best_score,
        best_estimator=best_pipeline,
        metrics=metrics,
        best_params=best_params,
        production_estimator=production_estimator,
    )

    try:
        save_model(best_pipeline, run_dir / "eval_model.joblib")
        save_json(name_mapping, run_dir / "name_mapping.json")

        if label_encoder is not None:
            save_model(label_encoder, run_dir / "label_encoder.joblib")
            logger.info(
                "Saved label encoder with classes: %s", label_encoder.classes_.tolist()
            )

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
                    "production_model.joblib"
                    if production_estimator is not None
                    else None
                ),
                "label_encoder_file": (
                    "label_encoder.joblib" if label_encoder else None
                ),
                "name_mapping_file": "name_mapping.json",
                "class_labels": (
                    label_encoder.classes_.tolist() if label_encoder else None
                ),
                "failed_models": failed_models if failed_models else None,
            },
            run_dir / "results.json",
        )

        if cfg.reporting.enabled:
            try:
                logger.info("Generating training report...")
                training_time = time.time() - training_start

                report_generator = ReportGenerator(run_dir)
                report_generator.generate_report(
                    best_estimator=best_pipeline,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    task=task,
                    best_model_name=best_model_name,
                    best_params=best_params,
                    cv_score=best_score,
                    test_metrics=metrics,
                    training_time=training_time,
                    all_model_scores=all_model_scores,
                    report_config=cfg.reporting,
                    scorer_name=scorer_name,
                    n_splits=cfg.split.n_splits,
                    random_state=cfg.split.random_state,
                )

            except Exception as e:
                logger.error(f"Failed to generate training report: {e}")

    except Exception as e:
        logger.error("Failed to save results: %s", e)
        raise

    return result
