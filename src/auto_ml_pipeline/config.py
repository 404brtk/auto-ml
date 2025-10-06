from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator, field_validator
import json
import tomllib
import yaml  # type: ignore


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


class SplitConfig(BaseModel):
    """Configuration for train/test splitting and cross-validation."""

    test_size: float = Field(
        default=0.2, gt=0, lt=1, description="Proportion of data for testing"
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (None for non-reproducible)",
    )
    stratify: bool = Field(
        default=True, description="Whether to stratify the split based on target"
    )
    n_splits: int = Field(
        default=5, ge=2, le=20, description="Number of cross-validation folds"
    )


class OutlierConfig(BaseModel):
    """Configuration for outlier detection and handling."""

    strategy: Optional[Literal["iqr", "zscore", "isolation_forest", "none"]] = Field(
        default="none", description="Outlier detection strategy"
    )
    method: Literal["clip", "remove"] = Field(
        default="remove",
        description="How to handle detected outliers. "
        "NOTE: IsolationForest only supports 'remove' method. ",
    )
    # IQR
    iqr_multiplier: float = Field(
        default=1.5,
        gt=0,
        description="IQR multiplier for outlier detection (1.5 = standard, 3.0 = conservative)",
    )
    # Z-score
    zscore_threshold: float = Field(
        default=3.0, gt=0, description="Z-score threshold for outlier detection"
    )
    # IsolationForest
    contamination: Union[float, Literal["auto"]] = Field(
        default="auto",
        description="Expected proportion of outliers for IsolationForest (0, 0.5] or 'auto'",
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of isolation trees (more trees = more stable but slower)",
    )
    max_samples: Union[int, float, Literal["auto"]] = Field(
        default="auto",
        description="Number of samples to train each tree. 'auto' uses min(256, n_samples), "
        "int for absolute count, float (0, 1] for fraction of samples",
    )
    random_state: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for IsolationForest reproducibility (None for non-reproducible)",
    )

    @field_validator("contamination")
    @classmethod
    def validate_contamination(cls, v):
        if v == "auto":
            return v
        if isinstance(v, (int, float)):
            if v <= 0 or v > 0.5:
                raise ValueError(
                    f"contamination must be 'auto' or in range (0, 0.5], got {v}"
                )
            return float(v)
        raise ValueError(
            f"contamination must be 'auto' or float, got {type(v).__name__}"
        )

    @field_validator("max_samples")
    @classmethod
    def validate_max_samples(cls, v):
        if v == "auto":
            return v

        if isinstance(v, int):
            if v <= 0:
                raise ValueError(f"max_samples as int must be positive, got {v}")
            return v

        if isinstance(v, float):
            if v <= 0 or v > 1:
                raise ValueError(
                    f"max_samples as float must be in range (0, 1], got {v}"
                )
            return v

        raise ValueError(
            f"max_samples must be 'auto', positive int, or float in (0, 1], got {type(v).__name__}"
        )

    @model_validator(mode="after")
    def validate_strategy_method_combination(self):
        """
        Validate that strategy and method combinations are valid.

        IsolationForest only supports 'remove' method because it produces
        binary classifications (outlier/inlier) without natural clipping bounds.
        """
        if self.strategy == "isolation_forest" and self.method == "clip":
            raise ValueError(
                "IsolationForest does not support method='clip'. "
                "IsolationForest is a classification-based method that identifies outliers "
                "but does not produce natural numerical bounds for clipping. "
                "Please use method='remove' with strategy='isolation_forest', "
                "or use strategy='iqr' or 'zscore' if you need clipping functionality."
            )
        return self


class CleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""

    # Pre-split cleaning (row-wise operations, no feature learning)
    drop_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    max_missing_row_ratio: Optional[float] = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Max missing ratio per row before dropping (0-1)",
    )
    remove_constant_features: bool = Field(
        default=True, description="Remove constant/quasi-constant features (PRE-SPLIT)"
    )
    constant_tolerance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Threshold for constant features. 1.0 = only 100% constant, 0.95 = 95%+ same value",
    )

    remove_id_columns: bool = Field(
        default=True,
        description="Automatically remove columns with >95% unique values (likely IDs)",
    )
    id_column_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Uniqueness ratio threshold for ID column detection",
    )
    handle_mixed_types: Literal["coerce", "drop"] = Field(
        default="coerce",
        description="How to handle mixed type columns: warn, coerce to string, or drop column",
    )
    numeric_coercion_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of values that must be numeric-coercible for object columns (0-1)",
    )
    uniqueness_ratio_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Uniqueness ratio threshold for integer targets with high cardinality. "
            "If unique_values/sample_size < threshold, infer classification; "
            "otherwise infer regression. Example: 0.05 means if <5% of values are unique, "
            "treat as classification"
        ),
    )
    max_categories_absolute: int = Field(
        default=20,
        ge=2,
        description=(
            "Maximum unique values for automatic classification inference. "
            "Integer targets with <= this many unique values are automatically "
            "treated as classification, regardless of dataset size. "
            "Does not prevent classification tasks with higher cardinality"
        ),
    )
    min_rows_after_cleaning: int = Field(
        default=1,
        ge=1,
        description="Minimum rows required after cleaning (raises error if below)",
    )
    min_cols_after_cleaning: int = Field(
        default=1,
        ge=1,
        description="Minimum columns required after cleaning (raises error if below)",
    )
    special_null_values: List[str] = Field(
        default_factory=lambda: [
            "?",
            "N/A",
            "n/a",
            "NA",
            "null",
            "NULL",
            "None",
            "none",
            "nan",
            "NaN",
            "NAN",
            "undefined",
            "missing",
            "MISSING",
            "-",
            "--",
            "---",
            "",
            " ",
        ],
        description="List of string values to treat as null/missing values",
    )

    # Post-split cleaning (fit on train, transform both train+test for clip, train only for remove)
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)


class ImputationConfig(BaseModel):
    """Configuration for missing value imputation."""

    strategy_cat: Literal["most_frequent", "random_sample"] = Field(
        default="most_frequent",
        description="Imputation strategy for missing values in categorical features",
    )
    strategy_num: Literal["mean", "median", "knn", "random_sample"] = Field(
        default="median",
        description="Imputation strategy for missing values in numeric features",
    )
    knn_neighbors: int = Field(
        default=5, ge=1, le=20, description="Number of neighbors for KNN imputation"
    )
    random_sample_seed: Optional[int] = Field(
        default=42,
        description="Random seed for random_sample imputation (None for non-reproducible)",
    )


class ScalingConfig(BaseModel):
    """Configuration for feature scaling."""

    strategy: Literal["standard", "minmax", "robust", "none"] = Field(
        default="standard", description="Feature scaling strategy"
    )


class EncodingConfig(BaseModel):
    """Configuration for categorical encoding."""

    high_cardinality_number_threshold: int = Field(
        default=100,
        ge=2,
        le=1000,
        description="Number of unique values to consider a feature as high cardinality",
    )
    high_cardinality_pct_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Percentage of unique values to consider a feature as high cardinality",
    )
    high_cardinality_encoder: Literal["target", "frequency"] = Field(
        default="target", description="Encoder to use for high cardinality features"
    )
    scale_high_card: bool = Field(
        default=False, description="Apply scaling to high cardinality encoded features"
    )


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering operations."""

    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)

    # Feature extraction
    extract_datetime: bool = Field(
        default=True, description="Extract features from datetime columns"
    )
    extract_time: bool = Field(
        default=True, description="Extract features from time-only columns"
    )
    handle_text: bool = Field(
        default=False, description="Enable text feature extraction"
    )
    max_features_text: int = Field(
        default=2000, ge=100, le=10000, description="Maximum text features to extract"
    )
    text_length_threshold: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Minimum text length to consider for feature extraction",
    )


class FeatureSelectionConfig(BaseModel):
    """Configuration for feature selection methods."""

    # Variance-based selection
    variance_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        description="Remove features with variance below this threshold",
    )

    # Correlation-based selection
    correlation_threshold: Optional[float] = Field(
        default=None,
        gt=0,
        lt=1,
        description="Remove features with correlation above this threshold",
    )
    correlation_method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation method for correlation-based selection",
    )

    # Univariate selection
    mutual_info_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of top features to select using mutual information",
    )

    # Dimensionality reduction
    pca_components: Optional[Union[int, float]] = Field(
        default=None,
        description="Number of PCA components (int) or variance ratio (float)",
    )

    @field_validator("pca_components")
    @classmethod
    def validate_pca_components(
        cls, v: Optional[Union[int, float]]
    ) -> Optional[Union[int, float]]:
        if v is not None:
            if isinstance(v, int) and v <= 0:
                raise ValueError(
                    "PCA components must be positive when specified as integer"
                )
            elif isinstance(v, float) and not (0 < v <= 1):
                raise ValueError("PCA variance ratio must be between 0 and 1")
        return v


class OptimizationConfig(BaseModel):
    """Configuration for hyperparameter optimization."""

    enabled: bool = Field(
        default=True, description="Enable hyperparameter optimization"
    )
    n_trials: int = Field(
        default=1, ge=1, le=1000, description="Number of optimization trials"
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=60,
        description="Timeout in seconds for optimization (None = no timeout)",
    )
    retrain_on_full_data: bool = Field(
        default=True,
        description="Retrain best model on full dataset (train + test) for production use",
    )
    sampler: Literal["tpe", "random", "cmaes"] = Field(
        default="tpe",
        description="Optuna sampler algorithm: tpe (Tree-structured Parzen Estimator), random, or cmaes",
    )
    pruner: Literal["median", "successive_halving", "hyperband", "none"] = Field(
        default="median",
        description="Optuna pruner for early stopping: median, successive_halving, hyperband, or none",
    )
    pruner_startup_trials: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Number of trials before pruning starts (for median pruner)",
    )


class EvalConfig(BaseModel):
    """Configuration for model evaluation."""

    metrics: Optional[List[str]] = Field(
        default=None,
        description="Custom metrics to evaluate (None = use default metrics)",
    )


class IOConfig(BaseModel):
    """Configuration for input/output operations."""

    dataset_path: Optional[Path] = Field(
        default=None, description="Path to the input dataset"
    )
    target: Optional[str] = Field(default=None, description="Name of the target column")
    output_dir: Path = Field(
        default=Path("outputs"), description="Directory for output files"
    )


class ModelsConfig(BaseModel):
    """Configuration for model selection and training.

    - include: only run these model keys (e.g., ["random_forest", "lightgbm"]).
    - exclude: run all available models except these.
    If both are provided, include takes precedence and exclude is applied after include.
    """

    include: Optional[List[str]] = Field(
        default=["xgboost"],
        description="Specific models to include (None = all available models)",
    )
    exclude: Optional[List[str]] = Field(
        default=None, description="Models to exclude from training"
    )

    @field_validator("include", "exclude")
    @classmethod
    def validate_model_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            # Import here to avoid circular imports
            from auto_ml_pipeline.models import (
                available_models_classification,
                available_models_regression,
            )

            # Get all available model names from both classification and regression
            valid_models: set[str] = set()
            valid_models.update(available_models_classification().keys())
            valid_models.update(available_models_regression().keys())

            invalid_models = set(v) - valid_models
            if invalid_models:
                raise ValueError(
                    f"Invalid model names: {invalid_models}. Valid models: {sorted(valid_models)}"
                )
        return v


class PipelineConfig(BaseModel):
    """Main configuration class for the AutoML pipeline."""

    task: Optional[TaskType] = Field(
        default=None, description="ML task type (auto-detected if None)"
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "PipelineConfig":
        # Ensure paths are Path objects
        if self.io.dataset_path is not None:
            self.io.dataset_path = Path(self.io.dataset_path)
        self.io.output_dir = Path(self.io.output_dir)
        return self


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Load configuration from TOML, YAML, or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        PipelineConfig: Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is unsupported or invalid
        ValidationError: If config values are invalid
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    if not p.is_file():
        raise ValueError(f"Configuration path is not a file: {p}")

    try:
        with p.open(
            "rb" if p.suffix.lower() == ".toml" else "r",
            encoding="utf-8" if p.suffix.lower() != ".toml" else None,
        ) as f:
            if p.suffix.lower() == ".toml":
                data = tomllib.load(f)
            elif p.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif p.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {p.suffix}. Supported formats: .toml, .yaml, .yml, .json"
                )
    except Exception as e:
        raise ValueError(f"Failed to parse configuration file {p}: {e}") from e

    try:
        return PipelineConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid configuration in {p}: {e}") from e


def default_config() -> PipelineConfig:
    """Get default pipeline configuration.

    Returns:
        PipelineConfig: Default configuration with sensible defaults
    """
    return PipelineConfig()
