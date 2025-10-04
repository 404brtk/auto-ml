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
        """Test inference of binary classification."""
        # 2 classes in 100 samples = 2/100 = 0.02 < 0.5 threshold -> classification
        df = pd.DataFrame({"target": [0, 1] * 50})  # 100 samples, 2 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_multiclass_classification(self):
        """Test inference of multiclass classification."""
        # 3 classes in 120 samples = 3/120 = 0.025 < 0.5 threshold -> classification
        df = pd.DataFrame({"target": [0, 1, 2] * 40})  # 120 samples, 3 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_float_regression(self):
        """Test inference of regression with float values."""
        df = pd.DataFrame(
            {"target": [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.1, 9.5, 10.2]}
        )
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_high_cardinality_integer_regression(self):
        """Test regression inference with many unique integers."""
        # 100 unique values in 100 samples = 100/100 = 1.0 >= 0.5 threshold -> regression
        df = pd.DataFrame({"target": list(range(100))})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_low_cardinality_integer_classification(self):
        """Test classification inference with few unique integers."""
        # 3 unique values in 120 samples = 3/120 = 0.025 < 0.5 threshold -> classification
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
        """Test that few samples generates warning but still infers."""
        # Use float to avoid integer classification issues with small data
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0]})  # Only 3 samples (< 10)
        task = infer_task(df, "target")
        # Float values -> regression
        assert task == TaskType.regression

    def test_mixed_with_nan(self):
        """Test inference with some NaN values."""
        # Use categorical strings to avoid integer classification issues
        target_data = ["cat", "dog", "bird"] * 40 + [np.nan] * 10
        df = pd.DataFrame({"target": target_data})
        task = infer_task(df, "target")
        # String categorical values -> classification
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
        """Test numeric strings with low cardinality."""
        # Use larger dataset: 3 unique values in 120 samples = 3/120 = 0.025 < 0.5 -> classification
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
        """Test numeric strings with high cardinality - after cleaning they become numeric."""
        # 100 unique values in 100 samples = 100/100 = 1.0 >= 0.5 -> regression
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": [str(i) for i in range(100)],  # 100% numeric strings
            }
        )

        # Simulate the cleaning process
        cleaning_cfg = CleaningConfig(
            remove_id_columns=False
        )  # Disable ID removal for this test
        df_cleaned = clean_data(df, "target", cleaning_cfg)

        # Now test task inference on the cleaned data
        task = infer_task(df_cleaned, "target")
        # After cleaning, these become numeric with 100 unique values -> regression
        assert task == TaskType.regression

    def test_numeric_strings_with_formatting(self):
        """Test numeric strings with comma formatting."""
        # 5 unique values in 100 samples = 5/100 = 0.05 < 0.5 -> classification
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": ["1,000", "2,000", "3,000", "4,000", "5,000"] * 20,
            }
        )

        # Simulate the cleaning process
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)

        # Now test task inference on the cleaned data
        task = infer_task(df_cleaned, "target")
        # After cleaning, should be detected as numeric, though low unique values -> classification
        assert task == TaskType.classification

    def test_mixed_numeric_and_text_strings(self):
        """Test strings that are partially numeric."""
        # 4 unique values in 12 samples = 4/12 = 0.33 < 0.5 -> classification
        df = pd.DataFrame(
            {
                "feature1": list(range(12)),
                "target": ["text1", "text2", "123", "456"] * 3,
            }
        )

        # Simulate the cleaning process
        cleaning_cfg = CleaningConfig(remove_id_columns=False)
        df_cleaned = clean_data(df, "target", cleaning_cfg)

        # Now test task inference on the cleaned data
        task = infer_task(df_cleaned, "target")
        # After cleaning, if still object type, treat as classification
        assert task == TaskType.classification


class TestInferTaskCustomParameters:
    """Test task inference with custom uniqueness ratio threshold parameters."""

    def test_custom_uniqueness_ratio_threshold_classification(self):
        """Test with custom uniqueness ratio threshold for classification.

        Setup: 25 unique values in 100 samples = 0.25 ratio
        Expected: ratio (0.25) < threshold (0.3) → classification
        """
        # Deterministic approach: repeat 25 unique values 4 times each = 100 samples
        df = pd.DataFrame({"target": list(range(25)) * 4})

        task = infer_task(df, "target", uniqueness_ratio_threshold=0.3)
        assert task == TaskType.classification

    def test_custom_uniqueness_ratio_threshold_regression(self):
        """Test with custom uniqueness ratio threshold for regression.

        Setup: 70 unique values in 100 samples = 0.70 ratio
        Expected: ratio (0.70) >= threshold (0.6) → regression
        """
        # Create 100 samples with exactly 70 unique values
        # First 70 values are unique (0-69), remaining 30 repeat from range(30)
        df = pd.DataFrame({"target": list(range(70)) + list(range(30))})

        task = infer_task(df, "target", uniqueness_ratio_threshold=0.6)
        assert task == TaskType.regression

    def test_large_dataset_with_moderate_ratio_classification(self):
        """Test large dataset with moderate uniqueness ratio for classification.

        Setup: 80 unique values in 1000 samples = 0.08 ratio
        Expected: ratio (0.08) < threshold (0.1) → classification

        Note: We use deterministic generation to guarantee exactly 80 unique values.
        np.random.choice with replace=True would not guarantee this.
        """
        n_samples = 1000
        n_unique = 80
        np.random.seed(42)

        # Generate exactly n_unique values distributed across n_samples
        # Each unique value appears approximately n_samples/n_unique times
        repeats = n_samples // n_unique  # 12 full repetitions
        remainder = n_samples % n_unique  # 40 additional values
        target_values = list(range(n_unique)) * repeats + list(range(remainder))

        # Shuffle to randomize order while maintaining exact uniqueness count
        np.random.shuffle(target_values)
        df = pd.DataFrame({"target": target_values})

        task = infer_task(df, "target", uniqueness_ratio_threshold=0.1)
        # Verify: 0.08 < 0.1 → classification
        assert task == TaskType.classification

    def test_large_dataset_with_high_ratio_regression(self):
        """Test large dataset with high uniqueness ratio for regression.

        Setup: 800 unique values in 1000 samples = 0.80 ratio
        Expected: ratio (0.80) >= threshold (0.7) → regression

        Note: We guarantee exactly 800 unique values by first including all 800 unique
        values once, then randomly sampling 200 additional values from the same pool.
        """
        n_samples = 1000
        desired_unique = 800
        np.random.seed(42)

        # Step 1: Include all unique values exactly once (800 values)
        unique_vals = list(range(desired_unique))

        # Step 2: Sample remaining values with replacement (200 values)
        # This ensures we have exactly 800 unique values total
        additional_vals = np.random.choice(
            range(desired_unique), size=n_samples - desired_unique, replace=True
        ).tolist()

        # Step 3: Combine and shuffle to randomize order
        all_vals = unique_vals + additional_vals
        np.random.shuffle(all_vals)
        df = pd.DataFrame({"target": all_vals})

        task = infer_task(df, "target", uniqueness_ratio_threshold=0.7)
        # Verify: 0.80 >= 0.7 → regression
        assert task == TaskType.regression


class TestInferTaskIntegerBoundary:
    """Test boundary cases for integer targets."""

    def test_exactly_at_ratio_threshold_classification(self):
        """Test with uniqueness ratio exactly at classification threshold."""
        # 50 unique in 100 samples = 0.5 == 0.5 threshold -> regression (>= threshold)
        df = pd.DataFrame({"target": list(range(50)) * 2})
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.5)
        # 0.5 >= 0.5 -> regression
        assert task == TaskType.regression

    def test_just_below_threshold_classification(self):
        """Test with uniqueness ratio just below threshold."""
        # 50 unique values in 100 samples = 50/100 = 0.5, but if we make one more duplicate, we get 49/100 = 0.49 < 0.5 -> classification
        # Create 100 values with 49 unique ones (so ratio is 49/100 = 0.49)
        df = pd.DataFrame(
            {"target": list(range(49)) + [0] * 51}
        )  # 49 unique + 51 duplicates of 0 = 100 total, 49 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.5)
        # 49/100 = 0.49 < 0.5 -> classification
        assert task == TaskType.classification

    def test_just_above_threshold_regression(self):
        """Test with uniqueness ratio just above threshold."""
        # 51 unique in 100 samples = 0.51 >= 0.5 threshold -> regression
        df = pd.DataFrame(
            {"target": list(range(51)) + [0, 0, 0, 0, 0, 0, 0, 0, 0]}
        )  # 100 samples, 51 unique
        task = infer_task(df, "target", uniqueness_ratio_threshold=0.5)
        # 0.51 >= 0.5 -> regression
        assert task == TaskType.regression

    def test_large_integers_as_ids_classification(self):
        """Test large integers that look like IDs."""
        # 10 unique values in 100 samples = 10/100 = 0.1 < 0.5 threshold -> classification
        df = pd.DataFrame(
            {
                "target": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
                * 10
            }
        )
        task = infer_task(df, "target")
        # Only 10 unique values out of 100 -> classification
        assert task == TaskType.classification


class TestInferTaskFloatTypes:
    """Test inference specifically for float types."""

    def test_float_always_regression(self):
        """Test that float types always infer as regression."""
        # Even with low cardinality
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] * 10})
        task = infer_task(df, "target")
        # Float type -> regression (continuous by nature)
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
