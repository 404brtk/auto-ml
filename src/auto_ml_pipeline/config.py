from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator
import json
import tomllib
import yaml  # type: ignore


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
    time_series = "time_series"


class SplitConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    time_column: Optional[str] = None
    n_splits: int = 5


class CleaningConfig(BaseModel):
    drop_missing_target: bool = True
    feature_missing_threshold: float = 0.5
    remove_constant: bool = True
    remove_duplicates: bool = True
    outlier_strategy: Optional[str] = Field(
        default=None, description="iqr|zscore|isoforest or None"
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
    # IsolationForest parameters
    outlier_contamination: float = 0.02


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
    high_cardinality_threshold: int = 20
    target_encoding: bool = True


class FeatureEngineeringConfig(BaseModel):
    imputation: ImputationConfig = ImputationConfig()
    scaling: ScalingConfig = ScalingConfig()
    encoding: EncodingConfig = EncodingConfig()
    extract_datetime: bool = True
    handle_text: bool = False  # lightweight by default
    max_features_text: int = 2000


class FeatureSelectionConfig(BaseModel):
    pca: bool = False
    pca_variance: float = 0.95
    correlation_threshold: Optional[float] = 0.95
    mutual_info_k: Optional[int] = None
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
