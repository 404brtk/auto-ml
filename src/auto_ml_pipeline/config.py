from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, model_validator
import json
import tomllib
import yaml  # type: ignore


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


class SplitConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    n_splits: int = 5


class CleaningConfig(BaseModel):
    # Whether to drop rows with missing target (applied pre-split; leakage-safe)
    drop_missing_target: bool = True

    # Whether to remove duplicate rows (train-only, applied after split on X_train)
    remove_duplicates: bool = True

    # Threshold for missingness in features (train-only). If set, a
    # FeatureMissingnessDropper transformer is added into the sklearn Pipeline
    # and learns columns to drop based on missing ratio on the training folds only.
    # None = disabled
    feature_missing_threshold: Optional[float] = 0.5

    # Whether to remove features that are constant across training rows.
    # Implemented via a ConstantFeatureDropper transformer inside the Pipeline.
    remove_constant: bool = True

    # Outlier detection strategy
    outlier_strategy: Optional[str] = Field(
        default=None, description="iqr|zscore or None"
    )
    # How to treat detected outliers: clip keeps row counts consistent for CV; remove drops rows (train-only by default)
    outlier_method: str = Field(default="clip", description="clip|remove")
    # Scope of applying outlier logic
    outlier_apply_scope: str = Field(
        default="train_only", description="train_only|both"
    )
    # IQR/Z-score parameters
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0


class ImputationConfig(BaseModel):
    strategy: str = Field(
        default="auto",
        description="auto|mean|median|most_frequent|knn|iterative",
    )


class ScalingConfig(BaseModel):
    strategy: str = Field(
        default="auto", description="auto|standard|minmax|robust|none"
    )


class EncodingConfig(BaseModel):
    # Threshold for high-cardinality categoricals
    high_cardinality_threshold: int = 20
    # Strategy for high-cardinality categoricals
    # auto|frequency|target|none (auto/none fall back to frequency)
    strategy: str = Field(default="frequency", description="auto|frequency|target|none")
    # Optionally scale numeric encodings produced for high-cardinality categoricals (target/frequency)
    scale_high_card: bool = False


class FeatureEngineeringConfig(BaseModel):
    imputation: ImputationConfig = ImputationConfig()
    scaling: ScalingConfig = ScalingConfig()
    encoding: EncodingConfig = EncodingConfig()
    extract_datetime: bool = True
    handle_text: bool = False  # lightweight by default
    max_features_text: int = 2000


class FeatureSelectionConfig(BaseModel):
    # PCA configuration
    pca_components: Optional[Union[int, float]] = None

    # Correlation filtering
    correlation_threshold: Optional[float] = 0.95

    # Mutual information feature selection
    mutual_info_k: Optional[int] = None

    # Quasi-constant features removal threshold
    variance_threshold: Optional[float] = None


class ModelingConfig(BaseModel):
    models: List[str] = Field(
        default_factory=lambda: [
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
        ]
    )


class OptimizationConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 20
    timeout: Optional[int] = None


class EvalConfig(BaseModel):
    # Optional list of metric names. The first one determines the CV scorer.
    # Supported aliases:
    #  - Classification: 'accuracy', 'f1_macro', 'roc_auc'
    #  - Regression: 'rmse', 'mse', 'mae', 'r2'
    # At reporting time, if provided, metrics are filtered to the requested subset.
    metrics: Optional[List[str]] = None
    shap: bool = False


class IOConfig(BaseModel):
    dataset_path: Optional[Path] = None
    target: Optional[str] = None
    output_dir: Path = Path("outputs")


class PipelineConfig(BaseModel):
    task: Optional[TaskType] = None
    split: SplitConfig = SplitConfig()
    cleaning: CleaningConfig = CleaningConfig()
    features: FeatureEngineeringConfig = FeatureEngineeringConfig()
    selection: FeatureSelectionConfig = FeatureSelectionConfig()
    modeling: ModelingConfig = ModelingConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    eval: EvalConfig = EvalConfig()
    io: IOConfig = IOConfig()

    @model_validator(mode="after")
    def validate_io(self) -> "PipelineConfig":
        if self.io.dataset_path is not None:
            self.io.dataset_path = Path(self.io.dataset_path)
        self.io.output_dir = Path(self.io.output_dir)
        return self


def load_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    with p.open("rb") as f:
        if p.suffix.lower() in {".toml"}:
            data = tomllib.load(f)
        elif p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif p.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")
    return PipelineConfig(**data)


def default_config() -> PipelineConfig:
    return PipelineConfig()
