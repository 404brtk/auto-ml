"""Tests for build_preprocessor function."""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.config import (
    FeatureEngineeringConfig,
    EncodingConfig,
    TaskType,
    ImputationConfig,
    ScalingConfig,
)


def has_nan(X):
    """Check if array (dense or sparse) contains NaN values."""
    if issparse(X):
        return np.isnan(X.data).any() if X.data.size > 0 else False
    else:
        return np.isnan(X).any()


@pytest.fixture
def numeric_df():
    """DataFrame with only numeric columns."""
    return pd.DataFrame({"num1": [1, 2, 3, 4, 5], "num2": [1.1, 2.2, 3.3, 4.4, 5.5]})


@pytest.fixture
def categorical_low_df():
    """DataFrame with low-cardinality categorical columns."""
    return pd.DataFrame({"category": ["A", "B", "A", "B", "A"] * 20})


@pytest.fixture
def categorical_high_df():
    """DataFrame with high-cardinality categorical columns."""
    return pd.DataFrame({"high_card": [f"cat_{i}" for i in range(100)]})


@pytest.fixture
def text_df():
    """DataFrame with text columns."""
    return pd.DataFrame(
        {
            "description": [
                "This is a very long sample text for testing that should exceed the character threshold easily",
                "Another piece of longer text content that definitely qualifies as text processing material",
                "Machine learning and artificial intelligence text processing with natural language understanding",
                "Final text sample here with additional content to make it sufficiently long for categorization",
                "More text for the model with extended description that provides comprehensive information",
            ]
        }
    )


@pytest.fixture
def datetime_df():
    """DataFrame with datetime columns."""
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})


@pytest.fixture
def time_df():
    """DataFrame with time columns."""
    return pd.DataFrame(
        {"time": ["14:30:00", "09:15:30", "18:45:00", "12:00:00", "23:59:00"]}
    )


@pytest.fixture
def mixed_df():
    """DataFrame with mixed column types."""
    return pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5] * 20,
            "category": ["A", "B", "A", "B", "A"] * 20,
            "text": [
                "This is the first sample text that should be long enough to trigger text processing",
                "This is the second sample text with sufficient length to be properly classified",
                "This is the third sample text that contains enough characters to meet threshold",
                "This is the fourth sample text designed to be adequately long for extraction",
                "This is the fifth sample text that provides comprehensive content for testing",
            ]
            * 20,
            "date": pd.date_range("2023-01-01", periods=100),
        }
    )


@pytest.fixture
def numeric_with_missing():
    """DataFrame with numeric missing values."""
    return pd.DataFrame(
        {
            "numeric": [1, 2, np.nan, 4, 5] * 20,
            "category": ["A", "B", np.nan, "B", "A"] * 20,
        }
    )


@pytest.fixture
def default_config():
    """Default feature engineering configuration."""
    return FeatureEngineeringConfig()


@pytest.fixture
def regression_target():
    """Random regression target."""
    np.random.seed(42)
    return np.random.randn(200) * 100


@pytest.fixture
def classification_target():
    """Random classification target."""
    np.random.seed(42)
    return np.random.randint(0, 3, 200)


class TestBuildPreprocessorBasic:
    """Test basic build_preprocessor functionality with different column types."""

    def test_numeric_only(self, numeric_df, default_config):
        """Test preprocessor with only numeric columns."""
        preprocessor, col_types = build_preprocessor(numeric_df, default_config)

        assert isinstance(preprocessor, ColumnTransformer)
        assert len(col_types.numeric) == 2
        assert col_types.categorical_low == []
        assert col_types.categorical_high == []
        assert col_types.datetime == []
        assert col_types.time == []
        assert col_types.text == []

        X_transformed = preprocessor.fit_transform(numeric_df)
        assert X_transformed.shape == (5, 2)

    def test_categorical_low_cardinality(self, categorical_low_df, default_config):
        """Test preprocessor with low-cardinality categorical columns."""
        preprocessor, col_types = build_preprocessor(categorical_low_df, default_config)

        assert len(col_types.categorical_low) == 1

        X_transformed = preprocessor.fit_transform(categorical_low_df)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == 1

    def test_text_columns(self, text_df):
        """Test preprocessor with text columns."""
        cfg = FeatureEngineeringConfig(handle_text=True, max_features_text=100)
        preprocessor, col_types = build_preprocessor(text_df, cfg)

        assert len(col_types.text) == 1

        X_transformed = preprocessor.fit_transform(text_df)
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] <= 100

    def test_datetime_columns(self, datetime_df):
        """Test preprocessor with datetime columns."""
        cfg = FeatureEngineeringConfig(extract_datetime=True)
        preprocessor, col_types = build_preprocessor(datetime_df, cfg)

        assert len(col_types.datetime) == 1

        X_transformed = preprocessor.fit_transform(datetime_df)
        assert X_transformed.shape == (5, 6)

    def test_time_columns(self, time_df):
        """Test preprocessor with time columns."""
        cfg = FeatureEngineeringConfig(extract_time=True)
        preprocessor, col_types = build_preprocessor(time_df, cfg)

        assert len(col_types.time) == 1

        X_transformed = preprocessor.fit_transform(time_df)
        assert X_transformed.shape == (5, 5)

    def test_empty_dataframe(self, default_config):
        """Test preprocessor when no columns match any category."""
        df = pd.DataFrame()
        preprocessor, col_types = build_preprocessor(df, default_config)

        transformers = preprocessor.transformers
        assert len(transformers) == 1
        assert transformers[0][0] == "passthrough"

    def test_mixed_column_types(self, mixed_df):
        """Test preprocessor with mixed column types."""
        cfg = FeatureEngineeringConfig(
            handle_text=True, extract_datetime=True, max_features_text=100
        )
        preprocessor, col_types = build_preprocessor(mixed_df, cfg)

        assert len(col_types.numeric) == 1
        assert len(col_types.categorical_low) == 1
        assert len(col_types.text) == 1
        assert len(col_types.datetime) == 1

        X_transformed = preprocessor.fit_transform(mixed_df)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] > 5


class TestBuildPreprocessorAdvanced:
    """Test advanced preprocessing scenarios."""

    def test_handles_missing_values(self, numeric_with_missing, default_config):
        """Test preprocessor handles missing values correctly."""
        preprocessor, col_types = build_preprocessor(
            numeric_with_missing, default_config
        )

        X_transformed = preprocessor.fit_transform(numeric_with_missing)
        assert not has_nan(X_transformed)

    def test_high_cardinality_categorical(self, categorical_high_df):
        """Test preprocessor with high cardinality categorical."""
        y = np.random.randn(100)
        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(high_cardinality_number_threshold=50)
        )
        preprocessor, col_types = build_preprocessor(categorical_high_df, cfg)

        assert len(col_types.categorical_high) == 1
        X_transformed = preprocessor.fit_transform(categorical_high_df, y)
        assert X_transformed.shape[0] == 100

    def test_no_scaling_option(self):
        """Test preprocessor without scaling."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy="none"))
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.max() > 100

    def test_knn_imputation(self):
        """Test preprocessor with KNN imputation."""
        df = pd.DataFrame(
            {"feat1": [1, 2, np.nan, 4, 5], "feat2": [10, 20, 30, 40, 50]}
        )
        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_num="knn", knn_neighbors=3)
        )
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert not has_nan(X_transformed)

    def test_single_row(self):
        """Test preprocessor with constant values."""
        df = pd.DataFrame({"numeric": list(range(1, 101)), "category": ["A", "B"] * 50})
        cfg = FeatureEngineeringConfig(encoding=EncodingConfig(ohe_drop=None))
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 100

    def test_transform_consistency_on_new_data(self):
        """Test that transform produces consistent results on new data."""
        df_train = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5] * 20,
                "category": ["A", "B", "A", "B", "A"] * 20,
            }
        )
        df_test = pd.DataFrame({"numeric": [6, 7], "category": ["A", "B"]})

        cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(df_train, cfg)

        preprocessor.fit(df_train)
        X_test = preprocessor.transform(df_test)

        assert X_test.shape[0] == 2


class TestEncoders:
    """Test different encoding strategies."""

    def test_target_encoder_regression(self):
        """Test TargetEncoder with regression task."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "high_card": [f"cat_{i % 50}" for i in range(200)],
                "numeric": np.random.randn(200),
            }
        )
        y = np.random.randn(200) * 100

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=20,
            )
        )

        preprocessor, col_types = build_preprocessor(df, cfg, TaskType.regression)
        assert len(col_types.categorical_high) == 1

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)

        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
        assert not has_nan(X_train_transformed)
        assert not has_nan(X_test_transformed)

    def test_target_encoder_classification(self):
        """Test TargetEncoder with classification task."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "high_card": [f"cat_{i % 40}" for i in range(200)],
            }
        )
        y = np.random.randint(0, 3, 200)

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=15,
            )
        )

        preprocessor, _ = build_preprocessor(df, cfg, TaskType.classification)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)

        assert not has_nan(X_train_transformed)
        assert not has_nan(X_test_transformed)

    def test_frequency_encoder(self):
        """Test FrequencyEncoder for high cardinality."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "high_card": [f"cat_{i % 60}" for i in range(300)],
            }
        )
        y = np.random.randn(300) * 100

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="frequency",
                high_cardinality_number_threshold=30,
            )
        )

        preprocessor, col_types = build_preprocessor(df, cfg, TaskType.regression)
        assert len(col_types.categorical_high) == 1

        X_transformed = preprocessor.fit_transform(df, y)
        assert X_transformed.shape[0] == len(df)
        assert not has_nan(X_transformed)

    def test_onehot_encoder_low_cardinality_default(self):
        """Test OneHotEncoder for low cardinality with default settings."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_low) == 1
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 2
        assert not has_nan(X_transformed)

    def test_onehot_encoder_no_drop(self):
        """Test OneHotEncoder with drop=None."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop=None,
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 3

    def test_onehot_encoder_drop_first(self):
        """Test OneHotEncoder with drop='first' (explicit)."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop="first",
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 2

    def test_onehot_encoder_drop_if_binary(self):
        """Test OneHotEncoder with drop='if_binary'."""
        df = pd.DataFrame(
            {
                "binary": ["A", "B"] * 50,
                "ternary": ["X", "Y", "Z"] * 33 + ["X"],
            }
        )

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop="if_binary",
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 4

    def test_ordinal_encoder_low_cardinality(self):
        """Test OrdinalEncoder for low cardinality."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(low_cardinality_encoder="ordinal")
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 1
        assert not has_nan(X_transformed)

    def test_frequency_encoder_low_cardinality(self):
        """Test FrequencyEncoder for low cardinality."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )
        y = np.random.randn(100)

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(low_cardinality_encoder="frequency")
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed.shape[1] == 1
        assert not has_nan(X_transformed)

    def test_target_encoder_low_cardinality(self):
        """Test TargetEncoder for low cardinality."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )
        y = np.random.randn(100)

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(low_cardinality_encoder="target")
        )
        preprocessor, _ = build_preprocessor(df, cfg, TaskType.regression)
        X_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed.shape[1] == 1
        assert not has_nan(X_transformed)

    def test_target_encoder_with_rare_categories(self):
        """Test TargetEncoder handles rare categories."""
        np.random.seed(42)
        categories = (
            ["common_A"] * 80 + ["common_B"] * 80 + ["rare_1"] * 2 + ["rare_2"] * 1
        )
        np.random.shuffle(categories)

        df = pd.DataFrame(
            {
                "category": categories[:100],
            }
        )
        y = np.random.randn(100) * 100

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=3,
            )
        )

        preprocessor, _ = build_preprocessor(df, cfg, TaskType.regression)
        X_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed.shape[0] == len(df)
        assert not has_nan(X_transformed)

    def test_target_encoder_unseen_categories(self):
        """Test TargetEncoder with unseen categories in test set."""
        np.random.seed(42)
        train_cats = ["A", "B", "C"]
        test_cats = ["A", "B", "D"]

        df_train = pd.DataFrame(
            {
                "category": np.random.choice(train_cats, 100),
            }
        )
        df_test = pd.DataFrame(
            {
                "category": np.random.choice(test_cats, 20),
            }
        )
        y_train = np.random.randn(100) * 100

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=2,
            )
        )

        preprocessor, _ = build_preprocessor(df_train, cfg, TaskType.regression)
        preprocessor.fit(df_train, y_train)

        X_test_transformed = preprocessor.transform(df_test)
        assert X_test_transformed.shape[0] == len(df_test)
        assert not has_nan(X_test_transformed)


class TestScalers:
    """Test different scaling strategies."""

    @pytest.mark.parametrize(
        "strategy,expected_mean,expected_std",
        [
            ("standard", 0.0, 1.0),
        ],
    )
    def test_standard_scaler(self, strategy, expected_mean, expected_std):
        """Test StandardScaler."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy=strategy))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        assert abs(X_transformed.mean()) < 1e-10
        assert abs(X_transformed.std() - 1.0) < 1e-10

    def test_minmax_scaler(self):
        """Test MinMaxScaler."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy="minmax"))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        assert X_transformed.min() >= -1e-10
        assert X_transformed.max() <= 1 + 1e-10

    def test_robust_scaler(self):
        """Test RobustScaler."""
        df = pd.DataFrame({"numeric": [1, 2, 3, 4, 100]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy="robust"))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[0] == len(df)
        assert not has_nan(X_transformed)

    def test_no_scaling(self):
        """Test no scaling option."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy="none"))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        assert X_transformed.max() == 10000
        assert X_transformed.min() == 1


class TestImputers:
    """Test different imputation strategies."""

    def test_median_imputation(self):
        """Test median imputation."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5],
            }
        )
        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_num="median"),
            scaling=ScalingConfig(strategy="none"),
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)
        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()
        assert 3 in X_transformed

    def test_mean_imputation(self):
        """Test mean imputation."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5],
            }
        )
        cfg = FeatureEngineeringConfig(imputation=ImputationConfig(strategy_num="mean"))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)

    def test_knn_imputation(self):
        """Test KNN imputation."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5],
                "feat2": [10, 20, 30, 40, 50],
            }
        )
        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_num="knn", knn_neighbors=2)
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)

    def test_categorical_imputation(self):
        """Test categorical imputation with most_frequent."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", np.nan, "A", "A"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_cat="most_frequent")
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)

    def test_categorical_random_sample_imputation(self):
        """Test categorical imputation with random_sample."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", np.nan, "A", "A"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(
                strategy_cat="random_sample", random_sample_seed=42
            ),
            scaling=ScalingConfig(scale_low_card=True),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)
        assert X_transformed.shape[1] == 1

    def test_random_sample_imputation(self):
        """Test random sample imputation."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 10,
            }
        )
        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(
                strategy_num="random_sample", random_sample_seed=42
            ),
            scaling=ScalingConfig(strategy="none"),
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not has_nan(X_transformed)
        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()
        assert X_transformed.min() >= 1
        assert X_transformed.max() <= 10


class TestMixedPipeline:
    """Test pipeline with multiple component types."""

    def test_all_components_together(self):
        """Test pipeline with all components: encoders, scalers, imputers."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5] * 20,
                "low_card": ["A", "B", "C", "A", "B"] * 20,
                "high_card": [f"cat_{i % 30}" for i in range(100)],
            }
        )
        y = np.random.randn(100) * 100

        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_num="median"),
            scaling=ScalingConfig(strategy="standard"),
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=10,
            ),
        )

        preprocessor, col_types = build_preprocessor(df, cfg, TaskType.regression)

        assert len(col_types.numeric) == 1
        assert len(col_types.categorical_low) == 1
        assert len(col_types.categorical_high) == 1

        X_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed.shape[0] == len(df)
        assert not has_nan(X_transformed)

    def test_pipeline_with_train_test_split(self):
        """Test pipeline maintains consistency across train/test split."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "numeric": np.random.randn(200),
                "category": [f"cat_{i % 25}" for i in range(200)],
            }
        )
        y = np.random.randn(200) * 100

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=10,
            )
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        preprocessor, _ = build_preprocessor(X_train, cfg, TaskType.regression)

        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)

        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert not has_nan(X_train_transformed)
        assert not has_nan(X_test_transformed)


class TestOneHotEncoderConfigurations:
    """Test various OneHotEncoder configuration combinations."""

    @pytest.mark.parametrize(
        "drop,expected_cols",
        [
            ("first", 2),
            (None, 3),
            ("if_binary", 3),
        ],
    )
    def test_ohe_drop_variations(self, drop, expected_cols):
        """Test OneHotEncoder with different drop settings."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 30})

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop=drop,
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == expected_cols

    def test_ohe_drop_if_binary_with_binary_column(self):
        """Test drop='if_binary' actually drops for binary columns."""
        df = pd.DataFrame({"binary": ["A", "B"] * 50})

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop="if_binary",
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 1

    def test_ohe_with_ordinal_comparison(self):
        """Compare OHE with ordinal encoding."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 30})

        cfg_ohe = FeatureEngineeringConfig(
            encoding=EncodingConfig(low_cardinality_encoder="ohe", ohe_drop=None),
        )
        preprocessor_ohe, _ = build_preprocessor(df, cfg_ohe)
        X_ohe = preprocessor_ohe.fit_transform(df)

        cfg_ordinal = FeatureEngineeringConfig(
            encoding=EncodingConfig(low_cardinality_encoder="ordinal")
        )
        preprocessor_ordinal, _ = build_preprocessor(df, cfg_ordinal)
        X_ordinal = preprocessor_ordinal.fit_transform(df)

        assert X_ohe.shape[1] == 3
        assert X_ordinal.shape[1] == 1

    def test_ohe_with_multiple_columns(self):
        """Test OHE with multiple categorical columns."""
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * 30,
                "cat2": ["X", "Y"] * 45,
            }
        )

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop="first",
            ),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 3

    def test_ohe_with_scaling(self):
        """Test OHE with sparse-compatible scaling."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 30})

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(
                low_cardinality_encoder="ohe",
                ohe_drop=None,
                scale_low_card=True,
            ),
            scaling=ScalingConfig(strategy="standard"),
        )
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[1] == 3
        assert not has_nan(X_transformed)
