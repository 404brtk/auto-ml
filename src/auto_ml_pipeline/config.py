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
        default=42, ge=0, description="Random seed for reproducibility"
    )
    stratify: bool = Field(
        default=True, description="Whether to stratify the split based on target"
    )
    n_splits: int = Field(
        default=5, ge=2, le=20, description="Number of cross-validation folds"
    )


class CleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""

    # Pre-split cleaning (row-wise operations, no feature learning)
    drop_missing_target: bool = Field(
        default=True, description="Drop rows with missing target values"
    )
    drop_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    max_missing_row_ratio: Optional[float] = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Max missing ratio per row before dropping (0-1)",
    )

    # Post-split cleaning (applied after split, fit on train)
    feature_missing_threshold: Optional[float] = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Drop features with missing ratio above this threshold",
    )
    remove_constant: bool = Field(
        default=True, description="Remove constant/zero-variance features"
    )

    # Outlier detection (fit on train, apply to train only)
    outlier_strategy: Optional[Literal["iqr", "zscore", "none"]] = Field(
        default=None, description="Outlier detection strategy"
    )
    outlier_method: Literal["clip", "remove"] = Field(
        default="clip", description="How to handle detected outliers"
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5,
        gt=0,
        description="IQR multiplier for outlier detection (1.5 = standard, 3.0 = conservative)",
    )
    outlier_zscore_threshold: float = Field(
        default=3.0, gt=0, description="Z-score threshold for outlier detection"
    )


class ImputationConfig(BaseModel):
    """Configuration for missing value imputation."""

    strategy: Literal["mean", "median", "knn"] = Field(
        default="median", description="Imputation strategy for missing values"
    )
    knn_neighbors: int = Field(
        default=5, ge=1, le=20, description="Number of neighbors for KNN imputation"
    )


class ScalingConfig(BaseModel):
    """Configuration for feature scaling."""

    strategy: Literal["standard", "minmax", "robust", "none"] = Field(
        default="standard", description="Feature scaling strategy"
    )


class EncodingConfig(BaseModel):
    """Configuration for categorical encoding."""

    high_cardinality_threshold: int = Field(
        default=50,
        ge=2,
        le=1000,
        description="Threshold for high cardinality categorical features",
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
