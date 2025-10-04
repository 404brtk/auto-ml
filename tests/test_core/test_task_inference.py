"""Tests for task_inference module."""

import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.task_inference import infer_task
from auto_ml_pipeline.config import TaskType, CleaningConfig
from auto_ml_pipeline.data_cleaning import clean_data


class TestInferTaskBasic:
    """Test basic task inference scenarios."""

    def test_binary_classification(self):
        """Test inference of binary classification.

        Binary targets (2 unique values) always -> classification.
        """
        df = pd.DataFrame({"target": [0, 1] * 50})  # 100 samples, 2 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_multiclass_classification(self):
        """Test inference of multiclass classification.

        3 unique values <= 20 (max_categories_absolute) -> classification.
        """
        df = pd.DataFrame({"target": [0, 1, 2] * 40})  # 120 samples, 3 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_float_regression(self):
        """Test inference of regression with non-integer float values."""
        df = pd.DataFrame(
            {"target": [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.1, 9.5, 10.2]}
        )
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_high_cardinality_integer_regression(self):
        """Test regression inference with many unique integers.

        100 unique values > 20 (max_categories_absolute) -> check ratio.
        100/100 = 100% uniqueness > 5% threshold -> regression.
        """
        df = pd.DataFrame({"target": list(range(100))})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_low_cardinality_integer_classification(self):
        """Test classification inference with few unique integers.

        3 unique values <= 20 (max_categories_absolute) -> classification.
        """
        df = pd.DataFrame({"target": [1, 2, 3] * 40})  # 120 samples, 3 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_boolean_classification(self):
        """Test classification inference with boolean values."""
        df = pd.DataFrame({"target": [True, False, True, False, True, False]})
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_categorical_dtype_classification(self):
        """Test classification inference with categorical dtype."""
        df = pd.DataFrame({"target": pd.Categorical(["A", "B", "C", "A", "B", "C"])})
        task = infer_task(df, "target")
        assert task == TaskType.classification


class TestInferTaskEdgeCases:
    """Test edge cases in task inference."""

    def test_missing_target_column_raises_error(self):
        """Test that missing target column raises KeyError."""
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(KeyError, match="Target column 'target' not found"):
            infer_task(df, "target")

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"target": []})
        task = infer_task(df, "target")
        # Should default to classification
        assert task == TaskType.classification

    def test_all_nan_values(self):
        """Test with all NaN values in target."""
        df = pd.DataFrame({"target": [np.nan, np.nan, np.nan]})
        task = infer_task(df, "target")
        # Should default to classification
        assert task == TaskType.classification

    def test_single_unique_value(self):
        """Test with constant target (single unique value)."""
        df = pd.DataFrame({"target": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]})
        task = infer_task(df, "target")
        # Constant prediction -> classification
        assert task == TaskType.classification

    def test_few_samples_warning(self):
        """Test that few samples generates warning but still infers.

        [0.0, 1.0, 0.0] = 2 unique integer-like float values -> classification.
        """
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0]})  # Only 3 samples (< 10)
        task = infer_task(df, "target")
        # 2 unique values (binary) -> classification
        assert task == TaskType.classification

    def test_mixed_with_nan(self):
        """Test inference with some NaN values."""
        target_data = ["cat", "dog", "bird"] * 40 + [np.nan] * 10
        df = pd.DataFrame({"target": target_data})
        task = infer_task(df, "target")
        # String categorical values (object dtype) -> classification
        assert task == TaskType.classification


class TestInferTaskStringTargets:
    """Test task inference with string targets."""

    def test_string_categorical(self):
        """Test categorical strings."""
        df = pd.DataFrame(
            {
                "feature": list(range(6)),
                "target": ["cat", "dog", "cat", "dog", "cat", "dog"],
            }
        )
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)
        task = infer_task(df_cleaned, "target")
        assert task == TaskType.classification

    def test_numeric_strings_classification(self):
        """Test numeric strings with low cardinality.

        After cleaning, 3 unique numeric values <= 20 -> classification.
        """
        df = pd.DataFrame(
            {
                "feature": list(range(120)),
                "target": ["1", "2", "3"] * 40,  # 120 samples, 3 classes
            }
        )
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)
        task = infer_task(df_cleaned, "target")
        assert task == TaskType.classification

    def test_numeric_strings_regression_after_cleaning(self):
        """Test numeric strings with high cardinality.

        After cleaning: 100 unique values > 20, 100% uniqueness > 5% -> regression.
        """
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": [str(i) for i in range(100)],  # 100% numeric strings
            }
        )

        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)
        task = infer_task(df_cleaned, "target")
        # After cleaning, 100 unique numeric values -> regression
        assert task == TaskType.regression

    def test_numeric_strings_with_formatting(self):
        """Test numeric strings with comma formatting.

        After cleaning: 5 unique values <= 20 -> classification.
        """
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": ["1,000", "2,000", "3,000", "4,000", "5,000"] * 20,
            }
        )

        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)
        task = infer_task(df_cleaned, "target")
        # 5 unique values <= 20 -> classification
        assert task == TaskType.classification

    def test_mixed_numeric_and_text_strings(self):
        """Test strings that are partially numeric."""
        df = pd.DataFrame(
            {
                "feature1": list(range(12)),
                "target": ["text1", "text2", "123", "456"] * 3,
            }
        )

        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)
        task = infer_task(df_cleaned, "target")
        # After cleaning, if still object type -> classification
        assert task == TaskType.classification


class TestInferTaskCustomParameters:
    """Test task inference with custom parameters."""

    def test_custom_max_categories_absolute_classification(self):
        """Test with custom max_categories_absolute for classification.

        15 unique values <= 30 (custom max_categories_absolute) -> classification.
        """
        df = pd.DataFrame({"target": list(range(15)) * 10})  # 150 samples, 15 unique
        task = infer_task(df, "target", max_categories_absolute=30)
        assert task == TaskType.classification

    def test_custom_max_categories_absolute_regression(self):
        """Test with custom max_categories_absolute for regression.

        25 unique values > 20 (default), but need to check ratio.
        25/100 = 25% uniqueness > 5% threshold -> regression.
        """
        df = pd.DataFrame({"target": list(range(25)) * 4})  # 100 samples, 25 unique
        task = infer_task(df, "target", max_categories_absolute=20)
        # 25 > 20, so check ratio: 25/100 = 0.25 > 0.05 -> regression
        assert task == TaskType.regression

    def test_custom_uniqueness_ratio_threshold_classification(self):
        """Test with custom uniqueness ratio threshold for classification.

        30 unique values > 20 (max_categories_absolute), so check ratio.
        30/100 = 30% uniqueness < 40% threshold -> classification.
        """
        # Create exactly 30 unique values in 100 samples
        df = pd.DataFrame(
            {"target": list(range(30)) * 3 + list(range(10))}
        )  # 100 samples, 30 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.4)
        # 30 > 20, but 30/100 = 0.30 < 0.40 -> classification
        assert task == TaskType.classification

    def test_custom_uniqueness_ratio_threshold_regression(self):
        """Test with custom uniqueness ratio threshold for regression.

        30 unique values > 20 (max_categories_absolute), so check ratio.
        30/100 = 30% uniqueness >= 20% threshold -> regression.
        """
        # Create exactly 30 unique values in 100 samples
        df = pd.DataFrame(
            {"target": list(range(30)) * 3 + list(range(10))}
        )  # 100 samples, 30 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.2)
        # 30 > 20, and 30/100 = 0.30 >= 0.20 -> regression
        assert task == TaskType.regression

    def test_large_dataset_low_ratio_classification(self):
        """Test large dataset with low uniqueness ratio.

        50 unique values > 20, so check ratio.
        50/1000 = 5% uniqueness >= 5% threshold -> regression (boundary case).
        """
        n_samples = 1000
        n_unique = 50
        # Create exactly 50 unique values distributed across 1000 samples
        target_values = list(range(n_unique)) * (n_samples // n_unique) + list(
            range(n_samples % n_unique)
        )
        df = pd.DataFrame({"target": target_values})

        # With default 0.05 threshold: 50/1000 = 0.05 >= 0.05 -> regression
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.05)
        assert task == TaskType.regression

        # With 0.10 threshold: 50/1000 = 0.05 < 0.10 -> classification
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.10)
        assert task == TaskType.classification

    def test_large_dataset_high_ratio_regression(self):
        """Test large dataset with high uniqueness ratio.

        800 unique values > 20, so check ratio.
        800/1000 = 80% uniqueness > 5% threshold -> regression.
        """
        n_samples = 1000
        desired_unique = 800
        np.random.seed(42)

        # Include all unique values once, then sample additional
        unique_vals = list(range(desired_unique))
        additional_vals = np.random.choice(
            range(desired_unique), size=n_samples - desired_unique, replace=True
        ).tolist()
        all_vals = unique_vals + additional_vals
        np.random.shuffle(all_vals)
        df = pd.DataFrame({"target": all_vals})

        task = infer_task(df, "target", uniqueness_ratio_threshold=0.05)
        # 800/1000 = 0.80 > 0.05 -> regression
        assert task == TaskType.regression


class TestInferTaskIntegerBoundary:
    """Test boundary cases for integer targets."""

    def test_exactly_at_max_categories_classification(self):
        """Test with exactly max_categories_absolute unique values.

        20 unique values == 20 (max_categories_absolute) -> classification.
        """
        df = pd.DataFrame({"target": list(range(20)) * 5})  # 100 samples, 20 unique
        task = infer_task(df, "target", max_categories_absolute=20)
        assert task == TaskType.classification

    def test_one_above_max_categories_checks_ratio(self):
        """Test with max_categories_absolute + 1 unique values.

        21 unique values > 20, so check ratio.
        21/100 = 21% uniqueness > 5% threshold -> regression.
        """
        df = pd.DataFrame(
            {"target": list(range(21)) + list(range(79))}
        )  # 100 samples, 21 unique
        task = infer_task(df, "target", max_categories_absolute=20)
        # 21 > 20, so check ratio: 21/100 = 0.21 > 0.05 -> regression
        assert task == TaskType.regression

    def test_just_below_ratio_threshold_classification(self):
        """Test with uniqueness ratio just below threshold.

        4 unique values in 100 samples = 4% < 5% threshold -> classification.
        But also 4 <= 20, so classification anyway.
        """
        df = pd.DataFrame({"target": [0, 1, 2, 3] * 25})  # 100 samples, 4 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.05)
        assert task == TaskType.classification

    def test_just_above_ratio_threshold_regression(self):
        """Test with uniqueness ratio just above threshold.

        30 unique values > 20, so check ratio.
        30/100 = 30% > 5% threshold -> regression.
        """
        df = pd.DataFrame(
            {"target": list(range(30)) + list(range(70))}
        )  # 100 samples, 30 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.05)
        # 30 > 20, and 30/100 = 0.30 > 0.05 -> regression
        assert task == TaskType.regression

    def test_age_like_values_regression(self):
        """Test age-like values (0-100) infer as regression.

        101 unique values > 20, and high uniqueness ratio -> regression.
        """
        df = pd.DataFrame({"target": list(range(101)) * 2})  # 202 samples, 101 unique
        task = infer_task(df, "target")
        # 101 > 20, and 101/202 ≈ 50% > 5% -> regression
        assert task == TaskType.regression

    def test_year_like_values_regression(self):
        """Test year-like values (1990-2025) infer as regression.

        36 unique values > 20, and high uniqueness ratio -> regression.
        """
        years = list(range(1990, 2026))  # 36 years
        df = pd.DataFrame({"target": years * 3})  # 108 samples, 36 unique
        task = infer_task(df, "target")
        # 36 > 20, and 36/108 = 33% > 5% -> regression
        assert task == TaskType.regression


class TestInferTaskFloatTypes:
    """Test inference specifically for float types."""

    def test_integer_like_float_low_cardinality_classification(self):
        """Test integer-like floats with low cardinality.

        [1.0, 2.0, 3.0] are integer-like with 3 unique values <= 20 -> classification.
        """
        df = pd.DataFrame(
            {"target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] * 10}
        )  # 60 samples, 3 unique
        task = infer_task(df, "target")
        # Integer-like floats with 3 unique values <= 20 -> classification
        assert task == TaskType.classification

    def test_integer_like_float_binary_classification(self):
        """Test binary integer-like floats.

        [0.0, 1.0] are integer-like with 2 unique values -> classification.
        """
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 20})
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_non_integer_float_regression(self):
        """Test non-integer floats always infer as regression."""
        df = pd.DataFrame({"target": [1.1, 2.2, 3.3, 1.1, 2.2, 3.3] * 10})
        task = infer_task(df, "target")
        # Non-integer floats -> regression
        assert task == TaskType.regression

    def test_float_with_many_decimals(self):
        """Test floats with many decimal places."""
        df = pd.DataFrame(
            {"target": [1.234567, 2.345678, 3.456789, 4.567890, 5.678901]}
        )
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_float_with_scientific_notation(self):
        """Test floats in scientific notation."""
        df = pd.DataFrame({"target": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_mixed_integer_and_decimal_floats(self):
        """Test mix of integer-like and non-integer floats.

        If any value has decimals, all are treated as continuous -> regression.
        """
        df = pd.DataFrame({"target": [1.0, 2.0, 3.5, 4.0, 5.0] * 10})
        task = infer_task(df, "target")
        # Contains 3.5 (non-integer), so -> regression
        assert task == TaskType.regression

    def test_integer_like_float_high_cardinality_regression(self):
        """Test integer-like floats with high cardinality.

        30 unique integer-like floats > 20 -> check ratio -> regression.
        """
        df = pd.DataFrame(
            {"target": [float(i) for i in range(30)] * 4}
        )  # 120 samples, 30 unique
        task = infer_task(df, "target")
        # 30 > 20, but all are integer-like, 30/120 = 25% > 5% -> regression
        assert task == TaskType.regression


class TestInferTaskRealWorldScenarios:
    """Test realistic machine learning scenarios."""

    def test_iris_species_classification(self):
        """Simulate Iris dataset - 3 species."""
        df = pd.DataFrame({"target": [0, 1, 2] * 50})  # 150 samples, 3 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_house_prices_regression(self):
        """Simulate house prices - continuous values."""
        np.random.seed(42)
        prices = np.random.uniform(100000, 500000, 1000)
        df = pd.DataFrame({"target": prices})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_customer_churn_binary_classification(self):
        """Simulate customer churn - binary outcome."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1] * 100})
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_star_rating_classification(self):
        """Simulate product ratings 1-5 stars."""
        df = pd.DataFrame({"target": [1, 2, 3, 4, 5] * 50})  # 250 samples, 5 ratings
        task = infer_task(df, "target")
        # 5 unique values <= 20 -> classification
        assert task == TaskType.classification

    def test_temperature_regression(self):
        """Simulate temperature measurements - continuous."""
        np.random.seed(42)
        temps = np.random.normal(20, 5, 500)  # Mean 20°C, std 5
        df = pd.DataFrame({"target": temps})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_count_data_low_values_classification(self):
        """Simulate count data with low values (e.g., number of purchases 0-10)."""
        np.random.seed(42)
        counts = np.random.poisson(2, 1000)  # Poisson with lambda=2 (mostly 0-10)
        df = pd.DataFrame({"target": counts})
        task = infer_task(df, "target")
        # Likely <= 20 unique values -> classification
        # But if more, check ratio
        nunique = df["target"].nunique()
        if nunique <= 20:
            assert task == TaskType.classification
        # If somehow > 20 unique, depends on ratio

    def test_count_data_high_values_regression(self):
        """Simulate count data with high values (e.g., daily visitors 0-1000)."""
        np.random.seed(42)
        counts = np.random.poisson(100, 500)  # Poisson with lambda=100
        df = pd.DataFrame({"target": counts})
        task = infer_task(df, "target")
        # Many unique values, high ratio -> regression
        assert task == TaskType.regression
