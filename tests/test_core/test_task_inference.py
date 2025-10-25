"""Tests for task_inference module."""

import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.task_inference import infer_task
from auto_ml_pipeline.config import TaskType, CleaningConfig
from auto_ml_pipeline.data_cleaning import clean_data


@pytest.fixture
def cleaning_config():
    """Default cleaning config for tests."""
    return CleaningConfig(remove_id_columns=False)


class TestInferTaskBasic:
    """Test basic task inference scenarios."""

    def test_binary_classification(self):
        df = pd.DataFrame({"target": [0, 1] * 50})
        assert infer_task(df, "target") == TaskType.classification

    def test_multiclass_classification(self):
        df = pd.DataFrame({"target": [0, 1, 2] * 40})
        assert infer_task(df, "target") == TaskType.classification

    def test_float_regression(self):
        df = pd.DataFrame(
            {"target": [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.1, 9.5, 10.2]}
        )
        assert infer_task(df, "target") == TaskType.regression

    def test_high_cardinality_integer_regression(self):
        df = pd.DataFrame({"target": list(range(100))})
        assert infer_task(df, "target") == TaskType.regression

    def test_low_cardinality_integer_classification(self):
        df = pd.DataFrame({"target": [1, 2, 3] * 40})
        assert infer_task(df, "target") == TaskType.classification

    def test_boolean_classification(self):
        df = pd.DataFrame({"target": [True, False, True, False, True, False]})
        assert infer_task(df, "target") == TaskType.classification

    def test_categorical_dtype_classification(self):
        df = pd.DataFrame({"target": pd.Categorical(["A", "B", "C", "A", "B", "C"])})
        assert infer_task(df, "target") == TaskType.classification


class TestInferTaskEdgeCases:
    """Test edge cases in task inference."""

    def test_missing_target_column_raises_error(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(KeyError, match="Target column 'target' not found"):
            infer_task(df, "target")

    def test_empty_dataframe(self):
        df = pd.DataFrame({"target": []})
        assert infer_task(df, "target") == TaskType.classification

    def test_all_nan_values(self):
        df = pd.DataFrame({"target": [np.nan, np.nan, np.nan]})
        assert infer_task(df, "target") == TaskType.classification

    def test_single_unique_value(self):
        df = pd.DataFrame({"target": [5] * 10})
        assert infer_task(df, "target") == TaskType.classification

    def test_few_samples_warning(self):
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0]})
        assert infer_task(df, "target") == TaskType.classification

    def test_mixed_with_nan(self):
        target_data = ["cat", "dog", "bird"] * 40 + [np.nan] * 10
        df = pd.DataFrame({"target": target_data})
        assert infer_task(df, "target") == TaskType.classification


class TestInferTaskStringTargets:
    """Test task inference with string targets."""

    def test_string_categorical(self, cleaning_config):
        df = pd.DataFrame(
            {
                "feature": list(range(6)),
                "target": ["cat", "dog", "cat", "dog", "cat", "dog"],
            }
        )
        df_cleaned, target, _ = clean_data(df, "target", cleaning_config)
        assert infer_task(df_cleaned, target) == TaskType.classification

    def test_numeric_strings_classification(self, cleaning_config):
        df = pd.DataFrame(
            {
                "feature": list(range(120)),
                "target": ["1", "2", "3"] * 40,
            }
        )
        df_cleaned, target, _ = clean_data(df, "target", cleaning_config)
        assert infer_task(df_cleaned, target) == TaskType.classification

    def test_numeric_strings_regression_after_cleaning(self, cleaning_config):
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": [str(i) for i in range(100)],
            }
        )
        df_cleaned, target, _ = clean_data(df, "target", cleaning_config)
        assert infer_task(df_cleaned, target) == TaskType.regression

    def test_numeric_strings_with_formatting(self, cleaning_config):
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "target": ["1,000", "2,000", "3,000", "4,000", "5,000"] * 20,
            }
        )
        df_cleaned, target, _ = clean_data(df, "target", cleaning_config)
        assert infer_task(df_cleaned, target) == TaskType.classification

    def test_mixed_numeric_and_text_strings(self, cleaning_config):
        df = pd.DataFrame(
            {
                "feature1": list(range(12)),
                "target": ["text1", "text2", "123", "456"] * 3,
            }
        )
        df_cleaned, target, _ = clean_data(df, "target", cleaning_config)
        assert infer_task(df_cleaned, target) == TaskType.classification


class TestInferTaskCustomParameters:
    """Test task inference with custom parameters."""

    def test_custom_max_categories_absolute_classification(self):
        df = pd.DataFrame({"target": list(range(15)) * 10})
        assert (
            infer_task(df, "target", max_categories_absolute=30)
            == TaskType.classification
        )

    def test_custom_max_categories_absolute_regression(self):
        df = pd.DataFrame({"target": list(range(25)) * 4})
        assert (
            infer_task(df, "target", max_categories_absolute=20) == TaskType.regression
        )

    def test_custom_uniqueness_ratio_threshold_classification(self):
        df = pd.DataFrame({"target": list(range(30)) * 3 + list(range(10))})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=0.4)
            == TaskType.classification
        )

    def test_custom_uniqueness_ratio_threshold_regression(self):
        df = pd.DataFrame({"target": list(range(30)) * 3 + list(range(10))})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=0.2)
            == TaskType.regression
        )

    @pytest.mark.parametrize(
        "threshold,expected",
        [
            (0.05, TaskType.regression),
            (0.10, TaskType.classification),
        ],
    )
    def test_large_dataset_boundary(self, threshold, expected):
        n_samples = 1000
        n_unique = 50
        target_values = list(range(n_unique)) * (n_samples // n_unique) + list(
            range(n_samples % n_unique)
        )
        df = pd.DataFrame({"target": target_values})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=threshold) == expected
        )

    def test_large_dataset_high_ratio_regression(self):
        n_samples = 1000
        desired_unique = 800
        np.random.seed(42)
        unique_vals = list(range(desired_unique))
        additional_vals = np.random.choice(
            range(desired_unique), size=n_samples - desired_unique, replace=True
        ).tolist()
        all_vals = unique_vals + additional_vals
        np.random.shuffle(all_vals)
        df = pd.DataFrame({"target": all_vals})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=0.05)
            == TaskType.regression
        )


class TestInferTaskIntegerBoundary:
    """Test boundary cases for integer targets."""

    def test_exactly_at_max_categories_classification(self):
        df = pd.DataFrame({"target": list(range(20)) * 5})
        assert (
            infer_task(df, "target", max_categories_absolute=20)
            == TaskType.classification
        )

    def test_one_above_max_categories_checks_ratio(self):
        df = pd.DataFrame({"target": list(range(21)) + list(range(79))})
        assert (
            infer_task(df, "target", max_categories_absolute=20) == TaskType.regression
        )

    def test_just_below_ratio_threshold_classification(self):
        df = pd.DataFrame({"target": [0, 1, 2, 3] * 25})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=0.05)
            == TaskType.classification
        )

    def test_just_above_ratio_threshold_regression(self):
        df = pd.DataFrame({"target": list(range(30)) + list(range(70))})
        assert (
            infer_task(df, "target", uniqueness_ratio_threshold=0.05)
            == TaskType.regression
        )

    def test_age_like_values_regression(self):
        df = pd.DataFrame({"target": list(range(101)) * 2})
        assert infer_task(df, "target") == TaskType.regression

    def test_year_like_values_regression(self):
        years = list(range(1990, 2026))
        df = pd.DataFrame({"target": years * 3})
        assert infer_task(df, "target") == TaskType.regression


class TestInferTaskFloatTypes:
    """Test inference specifically for float types."""

    def test_integer_like_float_low_cardinality_classification(self):
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] * 10})
        assert infer_task(df, "target") == TaskType.classification

    def test_integer_like_float_binary_classification(self):
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 20})
        assert infer_task(df, "target") == TaskType.classification

    def test_non_integer_float_regression(self):
        df = pd.DataFrame({"target": [1.1, 2.2, 3.3, 1.1, 2.2, 3.3] * 10})
        assert infer_task(df, "target") == TaskType.regression

    def test_float_with_many_decimals(self):
        df = pd.DataFrame(
            {"target": [1.234567, 2.345678, 3.456789, 4.567890, 5.678901]}
        )
        assert infer_task(df, "target") == TaskType.regression

    def test_float_with_scientific_notation(self):
        df = pd.DataFrame({"target": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]})
        assert infer_task(df, "target") == TaskType.regression

    def test_mixed_integer_and_decimal_floats(self):
        df = pd.DataFrame({"target": [1.0, 2.0, 3.5, 4.0, 5.0] * 10})
        assert infer_task(df, "target") == TaskType.regression

    def test_integer_like_float_high_cardinality_regression(self):
        df = pd.DataFrame({"target": [float(i) for i in range(30)] * 4})
        assert infer_task(df, "target") == TaskType.regression


class TestInferTaskRealWorldScenarios:
    """Test realistic machine learning scenarios."""

    @pytest.mark.parametrize(
        "n_classes,samples_per_class",
        [
            (3, 50),  # Iris
            (2, 100),  # Binary churn
            (5, 50),  # Star ratings
        ],
    )
    def test_classification_scenarios(self, n_classes, samples_per_class):
        df = pd.DataFrame({"target": list(range(n_classes)) * samples_per_class})
        assert infer_task(df, "target") == TaskType.classification

    @pytest.mark.parametrize(
        "seed,n_samples",
        [
            (42, 1000),  # House prices
            (123, 500),  # Temperature
        ],
    )
    def test_regression_scenarios(self, seed, n_samples):
        np.random.seed(seed)
        values = np.random.uniform(100000, 500000, n_samples)
        df = pd.DataFrame({"target": values})
        assert infer_task(df, "target") == TaskType.regression

    def test_count_data_high_values_regression(self):
        np.random.seed(42)
        counts = np.random.poisson(100, 500)
        df = pd.DataFrame({"target": counts})
        assert infer_task(df, "target") == TaskType.regression
