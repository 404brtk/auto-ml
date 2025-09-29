"""Tests for build_preprocessor function."""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.config import FeatureEngineeringConfig


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
        df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"]})
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_low) == 1

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 5
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
                "numeric": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "B", "A"],
                "text": [
                    "This is the first sample text that should be long enough to trigger text processing in the categorization system",
                    "This is the second sample text with sufficient length to be properly classified as text content for processing",
                    "This is the third sample text that contains enough characters to meet the text length threshold requirement",
                    "This is the fourth sample text designed to be adequately long for proper text feature extraction and processing",
                    "This is the fifth sample text that provides comprehensive content for testing the text processing capabilities",
                ],
                "date": pd.date_range("2023-01-01", periods=5),
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
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] > 5


class TestBuildPreprocessorAdvanced:
    """Test advanced preprocessing scenarios."""

    def test_handles_missing_values(self):
        """Test preprocessor handles missing values correctly."""
        df = pd.DataFrame(
            {"numeric": [1, 2, np.nan, 4, 5], "category": ["A", "B", np.nan, "B", "A"]}
        )
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert not np.isnan(X_transformed).any()

    def test_high_cardinality_categorical(self):
        """Test preprocessor with high cardinality categorical."""
        df = pd.DataFrame({"high_card": [f"cat_{i}" for i in range(100)]})
        cfg = FeatureEngineeringConfig(encoding={"high_cardinality_threshold": 50})
        preprocessor, col_types = build_preprocessor(df, cfg)

        assert len(col_types.categorical_high) == 1
        X_transformed = preprocessor.fit_transform(df)
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
        """Test preprocessor with single row."""
        df = pd.DataFrame({"numeric": [1], "category": ["A"]})
        cfg = FeatureEngineeringConfig()
        preprocessor, col_types = build_preprocessor(df, cfg)

        X_transformed = preprocessor.fit_transform(df)
        assert X_transformed.shape[0] == 1

    def test_transform_consistency_on_new_data(self):
        """Test that transform produces consistent results on new data."""
        df_train = pd.DataFrame(
            {"numeric": [1, 2, 3, 4, 5], "category": ["A", "B", "A", "B", "A"]}
        )
        df_test = pd.DataFrame({"numeric": [6, 7], "category": ["A", "B"]})

        cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(df_train, cfg)

        preprocessor.fit(df_train)
        X_test = preprocessor.transform(df_test)

        assert X_test.shape[0] == 2
