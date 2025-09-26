from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from auto_ml_pipeline.config import (
    FeatureEngineeringConfig,
)
from auto_ml_pipeline.transformers import (
    FrequencyEncoder,
    SimpleDateTimeFeatures,
    SimpleTimeFeatures,
)
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# TODO: fix: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
# need to investigate


def combine_text_columns(X):
    """Combine multiple text columns into a single string per row."""
    if hasattr(X, "to_numpy"):
        arr = X.astype(str).to_numpy()
    else:
        arr = np.asarray(X).astype(str)
    if arr.ndim == 1:
        return arr
    # Join columns with a space per row
    return np.array([" ".join(row) for row in arr])


@dataclass
class ColumnTypes:
    """Simple container for different column types."""

    numeric: List[str]
    categorical_low: List[str]
    categorical_high: List[str]
    datetime: List[str]
    time: List[str]
    text: List[str]


def categorize_columns(df: pd.DataFrame, cfg: FeatureEngineeringConfig) -> ColumnTypes:
    """Categorize columns by type using config-based heuristics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    categorical_low = []
    categorical_high = []
    time_cols = []
    text_cols = []

    for col in object_cols:
        if df[col].isna().all():
            continue

        nunique = df[col].nunique(dropna=True)
        sample_size = min(500, len(df))
        sample_values = df[col].dropna().head(sample_size).astype(str)

        if len(sample_values) == 0:
            continue

        avg_length = sample_values.str.len().mean()
        time_pattern_matches = sample_values.str.match(
            r"^\d{1,2}:\d{1,2}:\d{1,2}$"
        ).sum()
        time_pattern_pct = time_pattern_matches / len(sample_values)
        # additional data-driven check for high-cardinality categorical features
        is_short_and_unique = avg_length <= 20 and nunique > len(sample_values) * 0.8

        if time_pattern_pct >= 0.7:
            time_cols.append(col)
        elif avg_length > cfg.text_length_threshold:
            text_cols.append(col)
        elif nunique > cfg.encoding.high_cardinality_threshold or is_short_and_unique:
            categorical_high.append(col)
        else:
            categorical_low.append(col)

    return ColumnTypes(
        numeric_cols,
        categorical_low,
        categorical_high,
        datetime_cols,
        time_cols,
        text_cols,
    )


def get_imputer(strategy: str = "median", knn_neighbors: int = 5) -> BaseEstimator:
    """Get appropriate imputer based on strategy."""
    if strategy in ["mean", "median"]:
        return SimpleImputer(strategy=strategy)
    elif strategy == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    else:
        return SimpleImputer(strategy="median")


def get_scaler(strategy: str = "standard") -> Union[BaseEstimator, str]:
    """Get appropriate scaler based on strategy."""
    if strategy in ["none", None]:
        return "passthrough"
    elif strategy == "standard":
        return StandardScaler()
    elif strategy == "minmax":
        return MinMaxScaler()
    elif strategy == "robust":
        return RobustScaler()
    else:
        return StandardScaler()


def build_preprocessor(
    X: pd.DataFrame, cfg: FeatureEngineeringConfig
) -> Tuple[ColumnTransformer, ColumnTypes]:
    """Build preprocessing pipeline based on inferred column types."""

    col_types = categorize_columns(X, cfg)
    logger.info(
        "Column types: numeric=%d, cat_low=%d, cat_high=%d, datetime=%d, time=%d, text=%d",
        len(col_types.numeric),
        len(col_types.categorical_low),
        len(col_types.categorical_high),
        len(col_types.datetime),
        len(col_types.time),
        len(col_types.text),
    )

    transformers = []

    # Numeric pipeline
    if col_types.numeric:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    get_imputer(cfg.imputation.strategy, cfg.imputation.knn_neighbors),
                ),
                ("scaler", get_scaler(cfg.scaling.strategy)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, col_types.numeric))

    # Low cardinality categorical pipeline
    if col_types.categorical_low:
        cat_low_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat_low", cat_low_pipeline, col_types.categorical_low))

    # High cardinality categorical pipeline
    if col_types.categorical_high:
        steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

        # TEMP: Use FrequencyEncoder for all high-cardinality categorical features
        # TODO: add more encoders like HashingEncoder, etc.
        steps.append(("encoder", FrequencyEncoder()))

        # Optional scaling for encoded features
        if cfg.encoding.scale_high_card:
            steps.append(("scaler", get_scaler(cfg.scaling.strategy)))

        cat_high_pipeline = Pipeline(steps)
        transformers.append(("cat_high", cat_high_pipeline, col_types.categorical_high))

    # DateTime features
    if col_types.datetime and cfg.extract_datetime:
        datetime_pipeline = Pipeline(
            [
                ("datetime_features", SimpleDateTimeFeatures()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", get_scaler(cfg.scaling.strategy)),
            ]
        )
        transformers.append(("datetime", datetime_pipeline, col_types.datetime))

    # Time features
    if col_types.time and cfg.extract_time:
        time_pipeline = Pipeline(
            [
                ("time_features", SimpleTimeFeatures()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", get_scaler(cfg.scaling.strategy)),
            ]
        )
        transformers.append(("time", time_pipeline, col_types.time))

    # Text features (combine all text columns into one string per row)
    if col_types.text and cfg.handle_text:
        # Data-driven max_features: analyze vocabulary size
        sample_df = X[col_types.text].head(min(1000, len(X)))
        combined_sample = combine_text_columns(sample_df)
        vocab_size = len(set(" ".join(combined_sample).split()))

        # Adaptive max_features: 50% of vocabulary, bounded by config limits
        adaptive_max_features = min(
            max(int(vocab_size * 0.5), 100),  # at least 100, at most 50% of vocab
            cfg.max_features_text,  # don't exceed config limit
        )

        logger.info(
            "Text features: vocab_size=%d, adaptive_max_features=%d (config_limit=%d)",
            vocab_size,
            adaptive_max_features,
            cfg.max_features_text,
        )

        text_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("combine", FunctionTransformer(combine_text_columns, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=adaptive_max_features,
                        stop_words="english",
                        ngram_range=(1, 2),
                    ),
                ),
            ]
        )
        transformers.append(("text", text_pipeline, col_types.text))

    if not transformers:
        logger.warning(
            "No transformers created - all columns may have been filtered out"
        )
        # Create a dummy transformer to avoid errors
        transformers.append(("passthrough", "passthrough", []))

    logger.info("Built preprocessor with: %s", [name for name, _, _ in transformers])

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", n_jobs=1  # Avoid threading issues
    )

    return preprocessor, col_types
