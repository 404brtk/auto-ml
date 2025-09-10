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
    # Pre-split cleaning (row-wise operations, no feature learning)
    drop_missing_target: bool = True
    drop_duplicates: bool = True
    max_missing_features_per_row: Optional[int] = Field(default=None)

    # Post-split cleaning (applied after split, fit on train)
    feature_missing_threshold: Optional[float] = 0.5
    remove_constant: bool = True

    # Outlier detection (fit on train, apply based on scope)
    outlier_strategy: Optional[str] = Field(default=None, description="iqr|zscore|none")
    outlier_method: str = Field(default="clip", description="clip|remove")
    outlier_apply_scope: str = Field(
        default="train_only", description="train_only|both"
    )
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0


class ImputationConfig(BaseModel):
    strategy: str = Field(default="median", description="mean|median|knn")


class ScalingConfig(BaseModel):
    strategy: str = Field(default="standard", description="standard|minmax|robust|none")


class EncodingConfig(BaseModel):
    high_cardinality_threshold: int = Field(default=50, ge=1)
    scale_high_card: bool = False


class FeatureEngineeringConfig(BaseModel):
    imputation: ImputationConfig = ImputationConfig()
    scaling: ScalingConfig = ScalingConfig()
    encoding: EncodingConfig = EncodingConfig()

    # Feature extraction
    extract_datetime: bool = True
    handle_text: bool = False
    max_features_text: int = Field(default=2000, ge=100, le=10000)
    text_length_threshold: int = Field(default=50, ge=10, le=200)


class FeatureSelectionConfig(BaseModel):
    variance_threshold: Optional[float] = Field(default=None, ge=0)
    correlation_threshold: Optional[float] = Field(default=None, gt=0, lt=1)
    mutual_info_k: Optional[int] = Field(default=None, gt=0)
    pca_components: Optional[Union[int, float]] = None


class OptimizationConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 1
    timeout: Optional[int] = None


class EvalConfig(BaseModel):
    metrics: Optional[List[str]] = None


class IOConfig(BaseModel):
    dataset_path: Optional[Path] = None
    target: Optional[str] = None
    output_dir: Path = Path("outputs")


class ModelsConfig(BaseModel):
    """Configure which models to run.

    - include: only run these model keys (e.g., ["random_forest", "lightgbm"]).
    - exclude: run all available models except these.
    If both are provided, include takes precedence and exclude is applied after include.
    """

    include: Optional[List[str]] = ["lightgbm"]
    exclude: Optional[List[str]] = Field(default=None)


class PipelineConfig(BaseModel):
    task: Optional[TaskType] = None
    split: SplitConfig = SplitConfig()
    cleaning: CleaningConfig = CleaningConfig()
    features: FeatureEngineeringConfig = FeatureEngineeringConfig()
    selection: FeatureSelectionConfig = FeatureSelectionConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    eval: EvalConfig = EvalConfig()
    io: IOConfig = IOConfig()
    models: ModelsConfig = ModelsConfig()

    @model_validator(mode="after")
    def validate_paths(self) -> "PipelineConfig":
        if self.io.dataset_path is not None:
            self.io.dataset_path = Path(self.io.dataset_path)
        self.io.output_dir = Path(self.io.output_dir)
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Load configuration from TOML, YAML, or JSON file."""
    p = Path(path)

    with p.open("rb" if p.suffix.lower() == ".toml" else "r") as f:
        if p.suffix.lower() == ".toml":
            data = tomllib.load(f)
        elif p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif p.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")

    return PipelineConfig(**data)


def default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()
