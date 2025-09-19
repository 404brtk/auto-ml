import numpy as np
import pandas as pd
from typing import List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin


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
