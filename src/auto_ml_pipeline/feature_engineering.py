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


def combine_text_columns(X):
    """Combine multiple text columns into a single string per row.
    Accepts DataFrame or ndarray; returns 1D array of strings.
    """
    if hasattr(X, "to_numpy"):
        arr = X.astype(str).to_numpy()
    else:
        arr = np.asarray(X).astype(str)
    if arr.ndim == 1:
        return arr
    # Join columns with a space per row
    return np.array([" ".join(row) for row in arr])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Simple frequency encoder that works with DataFrames."""

    def __init__(self):
        self.frequency_maps_: Dict[str, Dict] = {}
        self.feature_names_in_: List[str] = []

    def _to_dataframe(self, X) -> pd.DataFrame:
        """Ensure input is a DataFrame with column names.
        - If X has `columns`, assume it's already a DataFrame-like and return as-is.
        - If X is a 1D array/Series, make it a single-column DataFrame.
        - If X is a 2D array, create generic column names as strings.
        """
        if hasattr(X, "columns"):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            return pd.DataFrame({"col_0": X})
        n_cols = X.shape[1]
        cols = [f"col_{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit frequency encoder on training data."""
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)

        for col in X_df.columns:
            value_counts = X_df[col].value_counts(dropna=False)
            # Store frequency ratios
            self.frequency_maps_[col] = (value_counts / len(X_df)).to_dict()
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform categorical values to frequencies."""
        X_df = self._to_dataframe(X).copy()

        # Process each fitted column
        result_data = {}
        for col in self.feature_names_in_:
            if col in X_df.columns:
                if col in self.frequency_maps_:
                    result_data[col] = (
                        X_df[col].map(self.frequency_maps_[col]).fillna(0.0)
                    )
                else:
                    result_data[col] = pd.Series([0.0] * len(X_df))
            else:
                # Add missing columns with default value
                result_data[col] = pd.Series([0.0] * len(X_df))

        # Create DataFrame with consistent column order
        return pd.DataFrame(result_data, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
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
    """Extract basic datetime features."""

    def __init__(self):
        self.datetime_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.datetime_cols_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract year, month, day, and day of week from datetime columns."""
        features = []

        for col in self.datetime_cols_:
            dt_series = pd.to_datetime(X[col], errors="coerce")

            # Extract basic features
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
        # Stack features horizontally
        return np.column_stack(features) if features else np.empty((len(X), 0))


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
        if cfg.encoding.strategy == "frequency":
            steps.append(("encoder", FrequencyEncoder()))
        else:  # Default to target
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
