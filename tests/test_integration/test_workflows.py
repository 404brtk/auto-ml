"""Integration tests for full auto-ML pipeline workflow."""

import pandas as pd
import numpy as np

from auto_ml_pipeline.config import CleaningConfig, FeatureEngineeringConfig, TaskType
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.feature_selection import build_selector
from auto_ml_pipeline.task_inference import infer_task


class TestEndToEndClassification:
    """Test end-to-end classification workflow."""

    def test_simple_classification_workflow(self):
        """Test simple classification from raw data to predictions."""
        # Create synthetic data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "numeric1": np.random.randn(100),
                "numeric2": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        # Data cleaning
        cleaning_cfg = CleaningConfig(drop_duplicates=True)
        df_clean = clean_data(df, "target", cleaning_cfg)

        # Feature engineering
        fe_cfg = FeatureEngineeringConfig()
        X = df_clean.drop("target", axis=1)

        preprocessor, col_types = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Verify transformations
        assert X_transformed.shape[0] == len(df_clean)
        assert X_transformed.shape[1] > 0
        assert not np.isnan(X_transformed).any()

    def test_workflow_with_missing_values(self):
        """Test workflow with missing values in features."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5] * 20,
                "feat2": [10, np.nan, 30, 40, 50] * 20,
                "category": ["A", "B", "A", np.nan, "C"] * 20,
                "target": [0, 1, 0, 1, 0] * 20,
            }
        )

        cleaning_cfg = CleaningConfig(drop_duplicates=False)
        df_clean = clean_data(df, "target", cleaning_cfg)

        fe_cfg = FeatureEngineeringConfig()
        X = df_clean.drop("target", axis=1)

        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Missing values should be imputed
        assert not np.isnan(X_transformed).any()

    def test_workflow_with_duplicates(self):
        """Test workflow removes duplicates."""
        df = pd.DataFrame({"feat": [1, 2, 1, 3], "target": [0, 1, 0, 1]})

        cleaning_cfg = CleaningConfig(drop_duplicates=True)
        df_clean = clean_data(df, "target", cleaning_cfg)

        # Should remove one duplicate
        assert len(df_clean) == 3

    def test_workflow_with_datetime(self):
        """Test workflow with datetime features."""
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-15", "2023-03-30", "2023-04-20"],
                "time": ["14:30:00", "09:15:30", "18:45:00", "12:00:00"],
                "target": [0, 1, 0, 1],
            }
        )

        # Disable ID column removal for small test dataset
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_clean = clean_data(df, "target", cleaning_cfg)

        fe_cfg = FeatureEngineeringConfig(extract_datetime=True, extract_time=True)
        X = df_clean.drop("target", axis=1)

        preprocessor, col_types = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Should extract datetime features
        assert X_transformed.shape[1] > 2  # More features than original

    def test_workflow_with_text(self):
        """Test workflow with text features."""
        df = pd.DataFrame(
            {
                "text": [
                    "This is a long sample text for classification task that needs to be really quite long to exceed the text length threshold for proper processing",
                    "Another text sample with different words and content and more words to make it sufficiently long for text feature extraction",
                    "Machine learning and natural language processing example with additional descriptive text to meet the minimum length requirements",
                    "Final text entry for testing purposes with extra content to ensure it is categorized as text and not as categorical variable",
                ],
                "target": [0, 1, 0, 1],
            }
        )

        # Disable ID column removal for small test dataset
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_clean = clean_data(df, "target", cleaning_cfg)

        fe_cfg = FeatureEngineeringConfig(
            handle_text=True, max_features_text=100, text_length_threshold=50
        )
        X = df_clean.drop("target", axis=1)

        preprocessor, col_types = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Should create TF-IDF features
        assert len(col_types.text) == 1
        assert X_transformed.shape[1] > 0


class TestEndToEndRegression:
    """Test end-to-end regression workflow."""

    def test_simple_regression_workflow(self):
        """Test simple regression workflow."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
                "feat3": np.random.choice(["low", "medium", "high"], 100),
                "target": np.random.randn(100) * 10 + 50,
            }
        )

        cleaning_cfg = CleaningConfig()
        df_clean = clean_data(df, "target", cleaning_cfg)

        fe_cfg = FeatureEngineeringConfig()
        X = df_clean.drop("target", axis=1)

        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == len(df_clean)
        assert not np.isnan(X_transformed).any()

    def test_regression_with_feature_selection(self):
        """Test regression with feature selection."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 20))
        y = pd.Series(np.random.randn(100))
        df = X.copy()
        df["target"] = y

        # Disable ID column removal for test dataset
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_clean = clean_data(df, "target", cleaning_cfg)

        fe_cfg = FeatureEngineeringConfig()
        X_clean = df_clean.drop("target", axis=1)
        y_clean = df_clean["target"]

        preprocessor, _ = build_preprocessor(X_clean, fe_cfg)
        X_transformed = preprocessor.fit_transform(X_clean)

        from auto_ml_pipeline.config import FeatureSelectionConfig

        fs_cfg = FeatureSelectionConfig(
            variance_threshold=0.1, correlation_threshold=0.95
        )
        selector = build_selector(fs_cfg, TaskType.regression)

        if selector is not None:
            X_selected = selector.fit_transform(X_transformed, y_clean)
            assert X_selected.shape[1] <= X_transformed.shape[1]


class TestTaskInference:
    """Test task type inference."""

    def test_infer_binary_classification(self):
        """Test inference of binary classification."""
        # Use larger dataset for proper inference
        df = pd.DataFrame({"target": [0, 1] * 50})  # 100 samples
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_infer_multiclass_classification(self):
        """Test inference of multiclass classification."""
        # Use larger dataset for proper inference
        df = pd.DataFrame({"target": [0, 1, 2] * 40})  # 120 samples
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_infer_regression(self):
        """Test inference of regression."""
        df = pd.DataFrame({"target": [1.5, 2.7, 3.2, 4.8, 5.1]})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_infer_integer_regression(self):
        """Test regression inference with integer targets."""
        # Many unique integer values should be regression
        df = pd.DataFrame({"target": list(range(100))})
        task = infer_task(df, "target")
        assert task == TaskType.regression


class TestDataCleaningIntegration:
    """Integration tests for data cleaning module."""

    def test_clean_with_all_operations(self):
        """Test cleaning with all operations enabled."""
        df = pd.DataFrame(
            {
                "feat1": [1, 2, 1, 4, np.nan, 6],  # Has duplicate and NaN
                "feat2": [10, 20, 10, 40, np.nan, 60],
                "numeric_str": ["1.5", "2.5", "1.5", "4.5", "5.5", "6.5"],
                "target": [0, 1, 0, 1, np.nan, 1],  # Has missing target
            }
        )

        cfg = CleaningConfig(
            drop_duplicates=True,
            max_missing_row_ratio=0.3,
            remove_id_columns=False,  # Disable for test dataset
        )

        result = clean_data(df, "target", cfg)

        # Should remove rows with missing target and duplicates
        assert len(result) < len(df)
        assert not result["target"].isna().any()
        # Check for duplicates properly
        has_duplicates = result.duplicated().any()
        assert not bool(has_duplicates)

    def test_clean_preserves_datatypes(self):
        """Test that cleaning preserves appropriate datatypes."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        # Disable ID column removal for small test dataset
        cfg = CleaningConfig(remove_id_columns=False)
        result = clean_data(df, "target", cfg)

        assert pd.api.types.is_integer_dtype(result["int_col"])
        assert pd.api.types.is_float_dtype(result["float_col"])
        assert result["str_col"].dtype == object

    def test_clean_with_datetime_conversion(self):
        """Test datetime conversion during cleaning."""
        df = pd.DataFrame(
            {
                "date_str": ["2023-01-01", "2023-02-15", "2023-03-30"],
                "target": [0, 1, 0],
            }
        )

        # Disable ID column removal for small test dataset
        cfg = CleaningConfig(remove_id_columns=False)
        result = clean_data(df, "target", cfg)

        # Date strings should be converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])

    def test_clean_with_numeric_coercion(self):
        """Test numeric coercion during cleaning."""
        df = pd.DataFrame({"numeric_str": ["1", "2", "3", "4"], "target": [0, 1, 0, 1]})

        # Disable ID column removal for small test dataset
        cfg = CleaningConfig(remove_id_columns=False)
        result = clean_data(df, "target", cfg)

        # Numeric strings should be converted to numbers
        assert pd.api.types.is_numeric_dtype(result["numeric_str"])


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering module."""

    def test_mixed_feature_types(self):
        """Test preprocessing with mixed feature types."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5] * 20,
                "categorical": ["A", "B", "A", "C", "B"] * 20,
                "datetime": pd.date_range("2023-01-01", periods=100),
                "text": [
                    "This is a longer text sample that exceeds the minimum length threshold for text processing",
                    "Another longer text sample with sufficient words to be categorized as text feature for machine learning",
                    "Third text sample with many words to ensure proper text categorization and feature extraction",
                    "Fourth text sample designed to be long enough to trigger text processing in the pipeline",
                    "Final text sample with adequate length to meet the text length requirements for classification",
                ]
                * 20,
            }
        )

        cfg = FeatureEngineeringConfig(
            extract_datetime=True, handle_text=True, text_length_threshold=50
        )

        preprocessor, col_types = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        # Check that feature types were identified
        # At least some features should be detected
        total_features = (
            len(col_types.numeric)
            + len(col_types.categorical_low)
            + len(col_types.categorical_high)
            + len(col_types.datetime)
            + len(col_types.text)
        )
        assert total_features > 0

        # Output should have no missing values
        assert not np.isnan(X_transformed).any()

    def test_high_cardinality_encoding(self):
        """Test encoding of high cardinality features."""
        # Create high cardinality feature
        df = pd.DataFrame({"high_card": [f"cat_{i}" for i in range(100)]})
        y = np.random.randn(100)  # Add target for TargetEncoder

        from auto_ml_pipeline.config import EncodingConfig

        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(high_cardinality_threshold=50)
        )

        preprocessor, col_types = build_preprocessor(df, cfg)

        # Should be detected as high cardinality
        assert len(col_types.categorical_high) == 1

        X_transformed = preprocessor.fit_transform(df, y)
        assert X_transformed.shape[0] == 100

    def test_scaling_strategies(self):
        """Test different scaling strategies."""
        df = pd.DataFrame({"feat": [1, 100, 1000, 10000]})

        from auto_ml_pipeline.config import ScalingConfig

        for strategy in ["standard", "minmax", "robust"]:
            cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy=strategy))

            preprocessor, _ = build_preprocessor(df, cfg)
            X_transformed = preprocessor.fit_transform(df)

            # Should scale the values
            assert X_transformed.min() < df["feat"].min()
            assert X_transformed.max() < df["feat"].max()

    def test_imputation_strategies(self):
        """Test different imputation strategies."""
        df = pd.DataFrame({"feat": [1, 2, np.nan, 4, 5]})

        from auto_ml_pipeline.config import ImputationConfig

        for strategy in ["mean", "median"]:
            cfg = FeatureEngineeringConfig(
                imputation=ImputationConfig(strategy=strategy)
            )

            preprocessor, _ = build_preprocessor(df, cfg)
            X_transformed = preprocessor.fit_transform(df)

            # Should impute missing values
            assert not np.isnan(X_transformed).any()


class TestCompleteWorkflow:
    """Test complete workflow from raw data to model-ready data."""

    def test_complete_classification_pipeline(self):
        """Test complete pipeline for classification."""
        np.random.seed(42)

        # Create realistic messy data
        df = pd.DataFrame(
            {
                "id": range(100),  # Should be high cardinality
                "numeric1": np.random.randn(100),
                "numeric2": np.random.randn(100) * 100 + 50,
                "category": np.random.choice(["A", "B", "C"], 100),
                "date": pd.date_range("2023-01-01", periods=100),
                "text": ["sample text " + str(i) for i in range(100)],
                "target": np.random.randint(0, 3, 100),
            }
        )

        # Add some messiness
        df.loc[5, "numeric1"] = np.nan
        df.loc[10, "category"] = np.nan
        df.loc[15, "target"] = np.nan
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # Add duplicates

        # Step 1: Clean
        clean_cfg = CleaningConfig(drop_duplicates=True, max_missing_row_ratio=0.5)
        df_clean = clean_data(df, "target", clean_cfg)

        # Step 2: Split features and target
        X = df_clean.drop("target", axis=1)
        y = df_clean["target"]

        # Step 3: Feature engineering
        from auto_ml_pipeline.config import EncodingConfig

        fe_cfg = FeatureEngineeringConfig(
            extract_datetime=True,
            handle_text=False,  # Skip text for speed
            encoding=EncodingConfig(high_cardinality_threshold=50),
        )
        preprocessor, col_types = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Step 4: Feature selection
        from auto_ml_pipeline.config import FeatureSelectionConfig

        fs_cfg = FeatureSelectionConfig(variance_threshold=0.01)
        selector = build_selector(fs_cfg, TaskType.classification)

        if selector is not None:
            X_final = selector.fit_transform(X_transformed, y)
        else:
            X_final = X_transformed

        # Verify final data quality
        assert X_final.shape[0] == len(y)
        assert X_final.shape[0] < 100  # Some rows removed
        assert not np.isnan(X_final).any()
        assert not np.isinf(X_final).any()

    def test_complete_regression_pipeline(self):
        """Test complete pipeline for regression."""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
                "feat3": np.random.choice(["low", "medium", "high"], 100),
                "outlier_feat": np.concatenate(
                    [np.random.randn(95), [100, 200, 300, 400, 500]]
                ),
                "target": np.random.randn(100) * 10 + 50,
            }
        )

        # Add outliers in target
        df.loc[90:95, "target"] = 1000

        # Clean
        clean_cfg = CleaningConfig(drop_duplicates=True)
        df_clean = clean_data(df, "target", clean_cfg)

        X = df_clean.drop("target", axis=1)
        y = df_clean["target"]

        # Feature engineering
        fe_cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        # Feature selection
        from auto_ml_pipeline.config import FeatureSelectionConfig

        fs_cfg = FeatureSelectionConfig(correlation_threshold=0.95)
        selector = build_selector(fs_cfg, TaskType.regression)

        if selector is not None:
            X_final = selector.fit_transform(X_transformed, y)
        else:
            X_final = X_transformed

        # Verify - use appropriate check for array/dataframe
        assert X_final.shape[0] == len(y)
        # Check for NaN values properly
        if hasattr(X_final, "values"):
            assert not np.isnan(X_final.values).any()
        else:
            assert not np.isnan(X_final).any()

    def test_pipeline_with_edge_cases(self):
        """Test pipeline handles edge cases gracefully."""
        # Small dataset
        df_small = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})

        clean_cfg = CleaningConfig()
        df_clean = clean_data(df_small, "target", clean_cfg)

        X = df_clean.drop("target", axis=1)
        fe_cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == 3

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(50),
                "feat2": np.random.randn(50),
                "target": np.random.randint(0, 2, 50),
            }
        )

        # Run pipeline twice
        results = []
        for _ in range(2):
            clean_cfg = CleaningConfig()
            df_clean = clean_data(df, "target", clean_cfg)

            X = df_clean.drop("target", axis=1)
            fe_cfg = FeatureEngineeringConfig()
            preprocessor, _ = build_preprocessor(X, fe_cfg)
            X_transformed = preprocessor.fit_transform(X)
            results.append(X_transformed)

        # Should produce same results
        np.testing.assert_array_almost_equal(results[0], results[1])
