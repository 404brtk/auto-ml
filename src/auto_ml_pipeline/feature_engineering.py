from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
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
    TargetEncoder,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from auto_ml_pipeline.config import (
    FeatureEngineeringConfig,
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


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Simple frequency encoder that replaces categorical values with their frequency ratios."""

    def __init__(self):
        self.frequency_maps_: Dict[str, Dict] = {}
        self.feature_names_in_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit frequency encoder on training data."""
        if isinstance(X, pd.DataFrame):
            X_df: pd.DataFrame = X.copy()
            self.feature_names_in_ = list(X_df.columns)
        else:
            # If array input, ensure 2D and create generic column names
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n_cols = arr.shape[1]
            self.feature_names_in_ = [f"col_{i}" for i in range(n_cols)]
            X_df = pd.DataFrame(arr, columns=self.feature_names_in_)

        for col in X_df.columns:
            value_counts = X_df[col].value_counts(dropna=False)
            self.frequency_maps_[col] = (value_counts / len(X_df)).to_dict()
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform categorical values to frequency ratios."""
        # Convert to DataFrame for consistent processing
        if isinstance(X, pd.DataFrame):
            X_df: pd.DataFrame = X.copy()
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_df = pd.DataFrame(arr, columns=self.feature_names_in_)

        for col in self.feature_names_in_:
            if col in X_df.columns:
                if col in self.frequency_maps_:
                    X_df[col] = X_df[col].map(self.frequency_maps_[col]).fillna(0.0)
                else:
                    X_df[col] = 0.0
            else:
                X_df[col] = 0.0

        # Return numpy array
        return X_df[self.feature_names_in_].values

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformed data."""
        return np.array(self.feature_names_in_)


@dataclass
class ColumnTypes:
    """Simple container for different column types."""

    numeric: List[str]
    categorical_low: List[str]
    categorical_high: List[str]
    datetime: List[str]
    text: List[str]


# TODO: fix datetime: probably should first try to convert to datetime,
# then check if it is actually datetime; should be done in data_cleaning probably
def categorize_columns(df: pd.DataFrame, cfg: FeatureEngineeringConfig) -> ColumnTypes:
    """Categorize columns by type using config-based heuristics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    categorical_low = []
    categorical_high = []
    text_cols = []

    for col in object_cols:
        nunique = df[col].nunique(dropna=True)
        sample_values = df[col].dropna().head(100).astype(str)
        avg_length = sample_values.str.len().mean() if len(sample_values) > 0 else 0

        if avg_length > cfg.text_length_threshold:
            text_cols.append(col)
        elif nunique > cfg.encoding.high_cardinality_threshold:
            categorical_high.append(col)
        else:
            categorical_low.append(col)

    return ColumnTypes(
        numeric_cols, categorical_low, categorical_high, datetime_cols, text_cols
    )


class SimpleDateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract basic datetime features (year, month, day, dayofweek, quarter, is_weekend)."""

    def __init__(self):
        self.datetime_cols_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit stores datetime columns and generates output feature names."""
        self.datetime_cols_ = X.columns.tolist()
        self.feature_names_out_ = []
        for col in self.datetime_cols_:
            self.feature_names_out_.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_quarter",
                    f"{col}_is_weekend",
                ]
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract datetime features as numeric array."""
        features = []
        for col in self.datetime_cols_:
            dt_series = pd.to_datetime(X[col], errors="coerce")
            features.extend(
                [
                    dt_series.dt.year.values,
                    dt_series.dt.month.values,
                    dt_series.dt.day.values,
                    dt_series.dt.dayofweek.values,
                    dt_series.dt.quarter.values,
                    (dt_series.dt.dayofweek >= 5).astype(int).values,
                ]
            )
        return np.column_stack(features) if features else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformed data."""
        return np.array(self.feature_names_out_)


# TODO: option: remove rows with missing values
def get_imputer(strategy: str = "auto") -> BaseEstimator:
    """Get appropriate imputer based on strategy."""
    if strategy in ["mean", "median"]:
        return SimpleImputer(strategy=strategy)
    elif strategy == "knn":
        return KNNImputer(n_neighbors=5)
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
    df: pd.DataFrame, target: str, cfg: FeatureEngineeringConfig
) -> Tuple[ColumnTransformer, ColumnTypes]:
    """Build preprocessing pipeline based on inferred column types."""

    X = df.drop(columns=[target])
    y = df[target]
    col_types = categorize_columns(X, cfg)
    logger.info(
        "Column types: numeric=%d, cat_low=%d, cat_high=%d, datetime=%d, text=%d",
        len(col_types.numeric),
        len(col_types.categorical_low),
        len(col_types.categorical_high),
        len(col_types.datetime),
        len(col_types.text),
    )

    transformers = []

    # Numeric pipeline
    if col_types.numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", get_imputer(cfg.imputation.strategy)),
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

        # Choose encoding strategy
        use_target_encoder = cfg.encoding.strategy != "frequency"

        # TODO: FIX: there is a problem with parsing target column
        # what is supposed to be float is parsed as int
        # as a result, TargetEncoder treats it as categorical
        # Safety: avoid TargetEncoder when too many target classes (can cause OOM)
        try:
            n_unique_target = y.nunique(dropna=True)
            is_float_target = getattr(y.dtype, "kind", None) == "f"
            if (
                use_target_encoder
                and not is_float_target
                and n_unique_target > cfg.encoding.target_encoder_max_classes
            ):
                logger.warning(
                    "Too many target classes (%d) for TargetEncoder; falling back to FrequencyEncoder",
                    n_unique_target,
                )
                use_target_encoder = False
        except Exception:
            # If anything goes wrong, default to safer frequency encoding
            use_target_encoder = False

        if not use_target_encoder:
            steps.append(("encoder", FrequencyEncoder()))
        else:
            steps.append(("encoder", TargetEncoder()))

        # Optional scaling for encoded features
        if cfg.encoding.scale_high_card:
            steps.append(("scaler", StandardScaler()))

        cat_high_pipeline = Pipeline(steps)
        transformers.append(("cat_high", cat_high_pipeline, col_types.categorical_high))

    # DateTime features
    if col_types.datetime and cfg.extract_datetime:
        datetime_pipeline = Pipeline(
            [
                ("datetime_features", SimpleDateTimeFeatures()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("datetime", datetime_pipeline, col_types.datetime))

    # Text features (combine all text columns into one string per row)
    if col_types.text and cfg.handle_text:
        text_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("combine", FunctionTransformer(combine_text_columns, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=cfg.max_features_text,
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
