"""Tests for skewness correction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from auto_ml_pipeline.config import FeatureEngineeringConfig, SkewConfig
from auto_ml_pipeline.feature_engineering import build_preprocessor


@pytest.fixture
def skew_config_enabled():
    """Return a FeatureEngineeringConfig with skewness correction enabled."""
    return FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))


@pytest.fixture
def skew_config_disabled():
    """Return a FeatureEngineeringConfig with skewness correction disabled."""
    return FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=False))


@pytest.fixture
def skewed_data():
    """Return a DataFrame with skewed data."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "highly_skewed": np.random.exponential(scale=2, size=100) ** 4,
            "moderately_skewed": np.random.exponential(scale=1, size=100) ** 2,
            "not_skewed": np.random.normal(loc=0, scale=1, size=100),
        }
    )
    return data


def test_skewness_correction_is_applied(skew_config_enabled, skewed_data):
    """Test that PowerTransformer is applied when skewness correction is enabled."""
    preprocessor, _ = build_preprocessor(skewed_data, skew_config_enabled)
    preprocessor.fit(skewed_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    assert any(isinstance(step[1], PowerTransformer) for step in numeric_pipeline.steps)


def test_skewness_correction_is_not_applied(skew_config_disabled, skewed_data):
    """Test that PowerTransformer is not applied when skewness correction is disabled."""
    preprocessor, _ = build_preprocessor(skewed_data, skew_config_disabled)
    preprocessor.fit(skewed_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    assert not any(
        isinstance(step[1], PowerTransformer) for step in numeric_pipeline.steps
    )


def test_skewness_correction_reduces_skew(skew_config_enabled, skewed_data):
    """Test that skewness correction actually reduces skewness."""
    preprocessor, _ = build_preprocessor(skewed_data, skew_config_enabled)
    transformed_data = preprocessor.fit_transform(skewed_data)
    transformed_df = pd.DataFrame(transformed_data, columns=skewed_data.columns)

    original_skew = skewed_data.skew().abs()
    transformed_skew = transformed_df.skew().abs()

    assert transformed_skew["highly_skewed"] < original_skew["highly_skewed"]
    assert transformed_skew["moderately_skewed"] < original_skew["moderately_skewed"]


def test_skewness_uses_yeo_johnson(skewed_data):
    """Test that Yeo-Johnson method is used."""
    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(skewed_data, config)
    preprocessor.fit(skewed_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    power_transformer = next(
        step[1]
        for step in numeric_pipeline.steps
        if isinstance(step[1], PowerTransformer)
    )

    assert power_transformer.method == "yeo-johnson"
    assert power_transformer.standardize is False


def test_skewness_with_negative_values():
    """Test that Yeo-Johnson handles negative values correctly."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "with_negatives": np.random.normal(loc=0, scale=10, size=100) ** 3,
        }
    )

    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(data, config)

    transformed_data = preprocessor.fit_transform(data)
    assert transformed_data.shape == data.shape
    assert not np.isnan(transformed_data).any()


def test_skewness_with_positive_data():
    """Test that Yeo-Johnson works with strictly positive data."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "positive_only": np.random.exponential(scale=2, size=100) + 1,
        }
    )

    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(data, config)

    transformed_data = preprocessor.fit_transform(data)
    assert transformed_data.shape == data.shape
    assert not np.isnan(transformed_data).any()


def test_skewness_with_zero_variance_column():
    """Test that skewness correction handles zero variance columns."""
    data = pd.DataFrame(
        {
            "normal_col": np.random.exponential(scale=2, size=100),
            "constant_col": np.ones(100),
        }
    )

    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(data, config)

    transformed_data = preprocessor.fit_transform(data)
    assert transformed_data.shape[0] == data.shape[0]


def test_skewness_preserves_data_shape(skewed_data):
    """Test that skewness correction preserves the shape of data."""
    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(skewed_data, config)

    transformed_data = preprocessor.fit_transform(skewed_data)
    assert transformed_data.shape == skewed_data.shape


def test_skewness_with_missing_values():
    """Test that skewness correction works with missing values."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "with_nan": np.random.exponential(scale=2, size=100),
        }
    )
    data.loc[data.sample(10, random_state=42).index, "with_nan"] = np.nan

    config = FeatureEngineeringConfig(skew=SkewConfig(handle_skewness=True))
    preprocessor, _ = build_preprocessor(data, config)

    transformed_data = preprocessor.fit_transform(data)
    assert not np.isnan(transformed_data).any()


def test_skewness_correction_order_in_pipeline(skew_config_enabled, skewed_data):
    """Test that PowerTransformer is placed after imputer but before scaler."""
    preprocessor, _ = build_preprocessor(skewed_data, skew_config_enabled)
    preprocessor.fit(skewed_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    step_names = [name for name, _ in numeric_pipeline.steps]

    imputer_idx = step_names.index("imputer")
    skew_idx = step_names.index("skew")
    scaler_idx = step_names.index("scaler")

    assert imputer_idx < skew_idx < scaler_idx


def test_skewness_step_name_is_skew(skew_config_enabled, skewed_data):
    """Test that the skewness step is named 'skew'."""
    preprocessor, _ = build_preprocessor(skewed_data, skew_config_enabled)
    preprocessor.fit(skewed_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    step_names = [name for name, _ in numeric_pipeline.steps]

    assert "skew" in step_names
