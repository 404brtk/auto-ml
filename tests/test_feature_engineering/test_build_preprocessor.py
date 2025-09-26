import pandas as pd
from sklearn.compose import ColumnTransformer

from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.config import FeatureEngineeringConfig


def test_build_preprocessor_numeric_only():
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

    # Should be able to fit and transform
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5  # Same number of rows
    assert X_transformed.shape[1] == 2  # Same number of numeric columns


def test_build_preprocessor_categorical_low():
    """Test preprocessor with low-cardinality categorical columns."""
    df = pd.DataFrame({"category": ["A", "B", "A", "B", "A"]})
    cfg = FeatureEngineeringConfig()
    preprocessor, col_types = build_preprocessor(df, cfg)

    assert len(col_types.categorical_low) == 1

    # Should create one-hot encoded features
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] == 2  # 2 unique categories → 2 features


def test_build_preprocessor_text():
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

    # Should create TF-IDF features
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] <= 100  # Limited by max_features


def test_build_preprocessor_datetime():
    """Test preprocessor with datetime columns."""
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
    cfg = FeatureEngineeringConfig(extract_datetime=True)
    preprocessor, col_types = build_preprocessor(df, cfg)

    assert len(col_types.datetime) == 1

    # Should extract datetime features (year, month, day, etc.)
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    assert (
        X_transformed.shape[1] == 6
    )  # year, month, day, dayofweek, quarter, is_weekend


def test_build_preprocessor_time():
    """Test preprocessor with time columns."""
    df = pd.DataFrame(
        {"time": ["14:30:00", "09:15:30", "18:45:00", "12:00:00", "23:59:00"]}
    )
    cfg = FeatureEngineeringConfig(extract_time=True)
    preprocessor, col_types = build_preprocessor(df, cfg)

    assert len(col_types.time) == 1

    # Should extract time features
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    assert (
        X_transformed.shape[1] == 5
    )  # hour, minute, second, is_business_hours, time_category


def test_build_preprocessor_empty_columns():
    """Test preprocessor when no columns match any category."""
    df = pd.DataFrame()  # Empty DataFrame
    cfg = FeatureEngineeringConfig()
    preprocessor, col_types = build_preprocessor(df, cfg)

    # Should create passthrough transformer
    transformers = preprocessor.transformers
    assert len(transformers) == 1
    assert transformers[0][0] == "passthrough"


def test_build_preprocessor_mixed_types():
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

    # Should be able to fit and transform all together
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 5
    # Should have features from all transformers
    assert X_transformed.shape[1] > 5  # At least some features from each type
