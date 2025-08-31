from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from auto_ml_pipeline.config import (
    FeatureEngineeringConfig,
    EncodingConfig,
    ScalingConfig,
    ImputationConfig,
)
from auto_ml_pipeline.logging_utils import get_logger


logger = get_logger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.columns:
            counts = X[col].value_counts(dropna=False)
            self.freq_[col] = (counts / counts.sum()).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X_enc = X.copy()
        for col in X.columns:
            mapping = self.freq_.get(col, {})
            X_enc[col] = X[col].map(mapping).fillna(0.0)
        return X_enc.values


@dataclass
class InferredColumns:
    numeric: List[str]
    categorical_low: List[str]
    categorical_high: List[str]
    datetime: List[str]
    text: List[str]


def infer_columns(df: pd.DataFrame, cfg: EncodingConfig) -> InferredColumns:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    text_cols: List[str] = []
    cat_low: List[str] = []
    cat_high: List[str] = []
    for c in obj_cols:
        nunique = df[c].nunique(dropna=True)
        # Simple heuristic: long strings => text
        if df[c].dropna().astype(str).str.len().mean() > 30:
            text_cols.append(c)
        elif nunique > cfg.high_cardinality_threshold:
            cat_high.append(c)
        else:
            cat_low.append(c)
    return InferredColumns(num_cols, cat_low, cat_high, dt_cols, text_cols)


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        Xo = pd.DataFrame(index=X.index)
        for col in self.cols_:
            s = pd.to_datetime(X[col], errors="coerce")
            Xo[f"{col}_year"] = s.dt.year
            Xo[f"{col}_month"] = s.dt.month
            Xo[f"{col}_day"] = s.dt.day
            Xo[f"{col}_dow"] = s.dt.dayofweek
        return Xo.values


def _choose_imputer(cfg: ImputationConfig, is_numeric: bool):
    if cfg.strategy == "auto":
        return SimpleImputer(strategy="median" if is_numeric else "most_frequent")
    if cfg.strategy in {"mean", "median", "most_frequent"}:
        return SimpleImputer(strategy=cfg.strategy)
    if cfg.strategy == "knn":
        return KNNImputer(n_neighbors=5)
    # fallback
    return SimpleImputer(strategy="median" if is_numeric else "most_frequent")


def _choose_scaler(cfg: ScalingConfig):
    if cfg.strategy in {"none"}:
        return "passthrough"
    if cfg.strategy in {"auto", "standard"}:
        return StandardScaler()
    if cfg.strategy == "minmax":
        return MinMaxScaler()
    if cfg.strategy == "robust":
        return RobustScaler()
    return StandardScaler()


def _choose_high_card_strategy(enc_cfg: EncodingConfig) -> str:
    s = (getattr(enc_cfg, "strategy", "frequency") or "frequency").lower()
    if s in {"target", "frequency"}:
        return s
    return "frequency"


def build_preprocessor(
    df: pd.DataFrame, target: str, cfg: FeatureEngineeringConfig
) -> Tuple[ColumnTransformer, InferredColumns]:
    X = df.drop(columns=[target])
    cols = infer_columns(X, cfg.encoding)

    num_pipe = Pipeline(
        steps=[
            ("imputer", _choose_imputer(cfg.imputation, True)),
            ("scaler", _choose_scaler(cfg.scaling)),
        ]
    )

    # TODO: Add ordinal or label encoding for low-cardinality categoricals
    cat_low_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # High-cardinality categorical encoder selection
    strategy = _choose_high_card_strategy(cfg.encoding)

    if strategy == "target":
        cat_high_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("targenc", TargetEncoder(smoothing="auto")),
        ]
    else:  # "frequency" and any fallback
        cat_high_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("freq", FrequencyEncoder()),
        ]
    if cfg.encoding.scale_high_card:
        cat_high_steps.append(("scale", StandardScaler()))
    cat_high_pipe = Pipeline(steps=cat_high_steps)

    transformers = []
    if cols.numeric:
        transformers.append(("num", num_pipe, cols.numeric))
    if cols.categorical_low:
        transformers.append(("cat_low", cat_low_pipe, cols.categorical_low))
    if cols.categorical_high:
        transformers.append(("cat_high", cat_high_pipe, cols.categorical_high))
    if cols.datetime and cfg.extract_datetime:
        transformers.append(
            (
                "dt",
                Pipeline(
                    [
                        ("dt", DateTimeFeatures()),
                        ("imp", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                cols.datetime,
            )
        )
    if cfg.handle_text and cols.text:
        # Vectorize each text column separately and concatenate
        for i, col in enumerate(cols.text):
            transformers.append(
                (
                    f"tfidf_{i}",
                    TfidfVectorizer(max_features=cfg.max_features_text),
                    [col],
                )
            )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, cols
