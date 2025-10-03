"""Tests for build_preprocessor function."""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.config import FeatureEngineeringConfig, EncodingConfig, TaskType
from sklearn.model_selection import train_test_split


class TestBuildPreprocessorBasic:
    """Test basic build_preprocessor functionality with different column types."""

    def test_numeric_only(self):
        """Test preprocessor with only numeric columns."""
        df = pd.DataFrame({"num1": [1, 2, 3, 4, 5], "num2": [1.1, 2.2, 3.3, 4.4, 5.5]})
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert isinstance(preprocessor, ColumnTransformer)
        assert len(col_types.numeric) == 2
        assert col_types.categorical_low == []
        assert col_types.categorical_high == []
        assert col_types.datetime == []
        assert col_types.time == []
        assert col_types.text == []

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] == 2

    def test_categorical_low_cardinality(self):
        """Test preprocessor with low-cardinality categorical columns."""
        df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"] * 20})
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_low) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == 2  # 2 unique categories → 2 features

    def test_text_columns(self):
        """Test preprocessor with text columns."""
        df = pd.DataFrame(
            {
                "description": [
                    "This is a very long sample text for testing that should exceed the character threshold easily",
                    "Another piece of longer text content that definitely qualifies as text processing material for the model",
                    "Machine learning and artificial intelligence text processing with natural language understanding capabilities",
                    "Final text sample here with additional content to make it sufficiently long for proper categorization",
                    "More text for the model with extended description that provides comprehensive information about the topic",
                ]
            }
        )
        cfg = FeatureEngineeringConfig(handle_text=True, max_features_text=100)
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.text) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] <= 100

    def test_datetime_columns(self):
        """Test preprocessor with datetime columns."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
        cfg = FeatureEngineeringConfig(extract_datetime=True)
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.datetime) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 5
        assert (
            X_transformed.shape[1] == 6
        )  # year, month, day, dayofweek, quarter, is_weekend

    def test_time_columns(self):
        """Test preprocessor with time columns."""
        df = pd.DataFrame(
            {"time": ["14:30:00", "09:15:30", "18:45:00", "12:00:00", "23:59:00"]}
        )
        cfg = FeatureEngineeringConfig(extract_time=True)
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.time) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 5
        assert (
            X_transformed.shape[1] == 5
        )  # hour, minute, second, is_business_hours, time_category

    def test_empty_dataframe(self):
        """Test preprocessor when no columns match any category."""
        df = pd.DataFrame()
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        transformers = preprocessor.transformers
        assert len(transformers) == 1
        assert transformers[0][0] == "passthrough"

    def test_mixed_column_types(self):
        """Test preprocessor with mixed column types."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5] * 20,
                "category": ["A", "B", "A", "B", "A"] * 20,
                "text": [
                    "This is the first sample text that should be long enough to trigger text processing in the categorization system",
                    "This is the second sample text with sufficient length to be properly classified as text content for processing",
                    "This is the third sample text that contains enough characters to meet the text length threshold requirement",
                    "This is the fourth sample text designed to be adequately long for proper text feature extraction and processing",
                    "This is the fifth sample text that provides comprehensive content for testing the text processing capabilities",
                ]
                * 20,
                "date": pd.date_range("2023-01-01", periods=100),
            }
        )
        cfg = FeatureEngineeringConfig(
            handle_text=True, extract_datetime=True, max_features_text=100
        )
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.numeric) == 1
        assert len(col_types.categorical_low) == 1
        assert len(col_types.text) == 1
        assert len(col_types.datetime) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] > 5


class TestBuildPreprocessorAdvanced:
    """Test advanced preprocessing scenarios."""

    def test_handles_missing_values(self):
        """Test preprocessor handles missing values correctly."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5] * 20,
                "category": ["A", "B", np.nan, "B", "A"] * 20,
            }
        )
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert not np.isnan(X_transformed).any()

    def test_high_cardinality_categorical(self):
        """Test preprocessor with high cardinality categorical."""
        df = pd.DataFrame({"high_card": [f"cat_{i}" for i in range(100)]})
        y = np.random.randn(100)  # Add target for TargetEncoder
        cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 50})
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_high) == 1
        X_transformed = preprocessor.fit_transform(df, y)
        assert X_transformed.shape[0] == 100

    def test_no_scaling_option(self):
        """Test preprocessor without scaling."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling={"strategy": "none"})
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.max() > 100

    def test_knn_imputation(self):
        """Test preprocessor with KNN imputation."""
        df = pd.DataFrame(
            {"feat1": [1, 2, np.nan, 4, 5], "feat2": [10, 20, 30, 40, 50]}
        )
        cfg = FeatureEngineeringConfig(
            imputation={"strategy": "knn", "knn_neighbors": 3}
        )
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert not np.isnan(X_transformed).any()

    def test_single_row(self):
        """Test preprocessor with minimal samples."""
        df = pd.DataFrame({"numeric": [1] * 100, "category": ["A"] * 100})
        cfg = FeatureEngineeringConfig()
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
        assert not np.isnan(X_train_transformed).any()
        assert not np.isnan(X_test_transformed).any()

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

        assert not np.isnan(X_train_transformed).any()
        assert not np.isnan(X_test_transformed).any()

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
        assert not np.isnan(X_transformed).any()

    def test_onehot_encoder_low_cardinality(self):
        """Test OneHotEncoder for low cardinality."""
        df = pd.DataFrame(
            {
                "low_card": ["A", "B", "C", "A", "B"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_low) == 1
        X_transformed = preprocessor.fit_transform(df)

        # OneHot should create 3 columns for 3 categories
        assert X_transformed.shape[1] == 3
        assert not np.isnan(X_transformed).any()

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
        assert not np.isnan(X_transformed).any()

    def test_target_encoder_unseen_categories(self):
        """Test TargetEncoder with unseen categories in test set."""
        np.random.seed(42)
        train_cats = ["A", "B", "C"]
        test_cats = ["A", "B", "D"]  # D is unseen

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
        assert not np.isnan(X_test_transformed).any()


class TestScalers:
    """Test different scaling strategies."""

    def test_standard_scaler(self):
        """Test StandardScaler."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling={"strategy": "standard"})

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # Standard scaling should have mean ~0 and std ~1
        assert abs(X_transformed.mean()) < 1e-10
        assert abs(X_transformed.std() - 1.0) < 1e-10

    def test_minmax_scaler(self):
        """Test MinMaxScaler."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling={"strategy": "minmax"})

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # MinMax scaling should be in [0, 1] (with small tolerance for floating point)
        assert X_transformed.min() >= -1e-10
        assert X_transformed.max() <= 1 + 1e-10

    def test_robust_scaler(self):
        """Test RobustScaler."""
        df = pd.DataFrame({"numeric": [1, 2, 3, 4, 100]})  # 100 is outlier
        cfg = FeatureEngineeringConfig(scaling={"strategy": "robust"})

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # RobustScaler should handle outliers better
        assert X_transformed.shape[0] == len(df)
        assert not np.isnan(X_transformed).any()

    def test_no_scaling(self):
        """Test no scaling option."""
        df = pd.DataFrame({"numeric": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling={"strategy": "none"})

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # Values should remain unchanged
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
            imputation={"strategy": "median"},
            scaling={"strategy": "none"},  # Disable scaling to check imputed value
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not np.isnan(X_transformed).any()
        # Median of [1,2,4,5] is 3
        assert 3 in X_transformed

    def test_mean_imputation(self):
        """Test mean imputation."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5],
            }
        )
        cfg = FeatureEngineeringConfig(imputation={"strategy": "mean"})

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not np.isnan(X_transformed).any()

    def test_knn_imputation(self):
        """Test KNN imputation."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5],
                "feat2": [10, 20, 30, 40, 50],
            }
        )
        cfg = FeatureEngineeringConfig(
            imputation={"strategy": "knn", "knn_neighbors": 2}
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert not np.isnan(X_transformed).any()

    def test_categorical_imputation(self):
        """Test categorical imputation with most_frequent."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", np.nan, "A", "A"] * 20,
            }
        )

        cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # Should impute with most frequent (A)
        assert not np.isnan(X_transformed).any()


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
            imputation={"strategy": "median"},
            scaling={"strategy": "standard"},
            encoding=EncodingConfig(
                high_cardinality_encoder="target",
                high_cardinality_number_threshold=10,
            ),
        )

        preprocessor, col_types = build_preprocessor(df, cfg, TaskType.regression)

        # Verify column detection
        assert len(col_types.numeric) == 1
        assert len(col_types.categorical_low) == 1
        assert len(col_types.categorical_high) == 1

        # Transform
        X_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed.shape[0] == len(df)
        assert not np.isnan(X_transformed).any()

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

        # Same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert not np.isnan(X_train_transformed).any()
        assert not np.isnan(X_test_transformed).any()
