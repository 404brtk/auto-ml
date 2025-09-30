"""Tests for task_inference module."""

import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.task_inference import infer_task
from auto_ml_pipeline.config import TaskType


class TestInferTaskBasic:
    """Test basic task inference scenarios."""

    def test_binary_classification(self):
        """Test inference of binary classification."""
        # Use larger dataset so dynamic threshold works properly
        df = pd.DataFrame({"target": [0, 1] * 50})  # 100 samples, 2 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_multiclass_classification(self):
        """Test inference of multiclass classification."""
        # Use larger dataset so dynamic threshold works properly
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
        df = pd.DataFrame({"target": list(range(100))})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_low_cardinality_integer_classification(self):
        """Test classification inference with few unique integers."""
        # Use larger dataset so dynamic threshold works properly
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
        df = pd.DataFrame({"target": ["cat", "dog", "cat", "dog", "cat", "dog"]})
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_numeric_strings_classification(self):
        """Test numeric strings with low cardinality."""
        # Use larger dataset
        df = pd.DataFrame({"target": ["1", "2", "3"] * 40})  # 120 samples, 3 classes
        task = infer_task(df, "target")
        assert task == TaskType.classification

    def test_numeric_strings_regression(self):
        """Test numeric strings with high cardinality."""
        df = pd.DataFrame({"target": [str(i) for i in range(100)]})
        task = infer_task(df, "target")
        assert task == TaskType.regression

    def test_numeric_strings_with_formatting(self):
        """Test numeric strings with comma formatting."""
        df = pd.DataFrame(
            {"target": ["1,000", "2,000", "3,000", "4,000", "5,000"] * 20}
        )
        task = infer_task(df, "target")
        # After removing commas, should be detected as numeric
        assert task == TaskType.classification  # Low unique values

    def test_mixed_numeric_and_text_strings(self):
        """Test strings that are partially numeric."""
        df = pd.DataFrame({"target": ["text1", "text2", "123", "456"] * 3})
        task = infer_task(df, "target")
        # Not mostly numeric -> classification
        assert task == TaskType.classification


class TestInferTaskCustomParameters:
    """Test task inference with custom parameters."""

    def test_custom_classification_threshold(self):
        """Test with custom cardinality threshold."""
        # 25 unique in 250 samples = 10% uniqueness
        df = pd.DataFrame({"target": list(range(25)) * 10})  # 250 samples, 25 unique

        # With default threshold (30), should be classification (25 <= 30 and 10% < 50%)
        task_default = infer_task(df, "target")
        assert task_default == TaskType.classification

        # With lower threshold (20), should be regression (25 > 20)
        task_custom = infer_task(df, "target", classification_cardinality_threshold=20)
        assert task_custom == TaskType.regression

    def test_custom_numeric_coercion_threshold(self):
        """Test with custom numeric coercion threshold."""
        # 70% numeric, 30% text - use larger dataset
        target_data = (["1", "2", "3", "4", "5", "6", "7"] * 10) + (["text"] * 30)
        df = pd.DataFrame({"target": target_data})

        # With 95% threshold, should be classification (not enough numeric)
        task_high = infer_task(df, "target", numeric_coercion_threshold=0.95)
        assert task_high == TaskType.classification

        # With 60% threshold, should treat as numeric
        task_low = infer_task(df, "target", numeric_coercion_threshold=0.6)
        # After coercion, has low cardinality (7 unique) -> classification
        assert task_low == TaskType.classification

    def test_large_dataset_with_moderate_cardinality(self):
        """Test large dataset with moderate unique values."""
        # 80 unique values in 1000 samples = 8% uniqueness
        n_samples = 1000
        n_unique = 80
        np.random.seed(42)
        df = pd.DataFrame({"target": np.random.choice(range(n_unique), size=n_samples)})

        task = infer_task(df, "target")
        # 80 > 30 (default threshold) -> regression
        assert task == TaskType.regression


class TestInferTaskIntegerBoundary:
    """Test boundary cases for integer targets."""

    def test_exactly_at_threshold(self):
        """Test with exactly 20 unique values (at threshold)."""
        df = pd.DataFrame({"target": list(range(20)) * 10})
        task = infer_task(df, "target", classification_cardinality_threshold=20)
        # At threshold -> classification
        assert task == TaskType.classification

    def test_just_above_threshold(self):
        """Test with 21 unique values (just above threshold)."""
        df = pd.DataFrame({"target": list(range(21)) * 10})
        task = infer_task(df, "target", classification_cardinality_threshold=20)
        # Above threshold -> regression
        assert task == TaskType.regression

    def test_large_integers_as_ids(self):
        """Test large integers that look like IDs."""
        df = pd.DataFrame(
            {
                "target": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
                * 10
            }
        )
        task = infer_task(df, "target")
        # Only 10 unique values -> classification
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
