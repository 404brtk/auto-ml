"""Tests for remove_constant_features function."""

import pandas as pd
import numpy as np
from auto_ml_pipeline.data_cleaning import remove_constant_features


class TestRemoveConstantFeatures:
    """Test remove_constant_features function."""

    def test_remove_pure_constant(self):
        """Test removal of pure constant features (all same value)."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],  # Constant
                "feature2": [1, 2, 3, 4, 5],  # Variable
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        assert "feature1" not in result.columns
        assert "feature2" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_quasi_constant_with_high_tolerance(self):
        """Test removal of quasi-constant features with lower tolerance."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 2],  # 80% same value
                "feature2": [1, 2, 3, 4, 5],  # Variable
                "target": [0, 1, 0, 1, 0],
            }
        )

        # With tolerance=0.8, feature1 should be removed (80%+ same value)
        result = remove_constant_features(df, "target", constant_tolerance=0.8)

        assert "feature1" not in result.columns
        assert "feature2" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_quasi_constant_below_tolerance(self):
        """Test that quasi-constant below tolerance is kept."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 2, 3],  # 60% same value
                "feature2": [1, 2, 3, 4, 5],  # Variable
                "target": [0, 1, 0, 1, 0],
            }
        )

        # With tolerance=0.8, feature1 should be kept (only 60% same value)
        result = remove_constant_features(df, "target", constant_tolerance=0.8)

        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 3

    def test_multiple_constant_features(self):
        """Test removal of multiple constant features."""
        df = pd.DataFrame(
            {
                "const1": [5, 5, 5, 5, 5],
                "const2": ["a", "a", "a", "a", "a"],
                "const3": [True, True, True, True, True],
                "variable": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        assert "const1" not in result.columns
        assert "const2" not in result.columns
        assert "const3" not in result.columns
        assert "variable" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_target_not_removed(self):
        """Test that target column is never removed."""
        df = pd.DataFrame(
            {
                "feature1": [0, 1, 1, 1, 1],
                "target": [0, 0, 0, 0, 0],  # Constant target
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        # Target should be preserved even if constant
        assert "target" in result.columns
        assert "feature1" in result.columns
        assert len(result.columns) == 2

    def test_no_constant_features(self):
        """Test when there are no constant features."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        # All features should be kept
        assert len(result.columns) == 3
        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert "target" in result.columns

    def test_constant_with_missing_values(self):
        """Test constant feature with missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, np.nan, 1, 1],  # Constant except NaN
                "feature2": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        # feature1 should be removed (constant non-NaN values)
        assert "feature1" not in result.columns
        assert "feature2" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_string_constant_features(self):
        """Test removal of constant string features."""
        df = pd.DataFrame(
            {
                "str_const": ["same", "same", "same", "same", "same"],
                "str_var": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        assert "str_const" not in result.columns
        assert "str_var" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_boolean_constant_features(self):
        """Test removal of constant boolean features."""
        df = pd.DataFrame(
            {
                "bool_const": [True, True, True, True, True],
                "bool_var": [True, False, True, False, True],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        assert "bool_const" not in result.columns
        assert "bool_var" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 2

    def test_tolerance_edge_cases(self):
        """Test tolerance edge cases."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1, 1],  # 100% same
                "feature2": [1, 1, 1, 1, 2],  # 80% same
                "feature3": [1, 1, 1, 2, 3],  # 60% same
                "target": [0, 1, 0, 1, 0],
            }
        )

        # tolerance=1.0: only pure constants (100% same value)
        result_1 = remove_constant_features(df, "target", constant_tolerance=1.0)
        # feature-engine with tol=1.0 removes features where all values are the same
        assert "feature1" not in result_1.columns
        assert "feature2" in result_1.columns
        assert "feature3" in result_1.columns
        assert "target" in result_1.columns
        assert len(result_1.columns) == 3

        # tolerance=0.8: 80%+ same value
        result_08 = remove_constant_features(df, "target", constant_tolerance=0.8)
        assert "feature1" not in result_08.columns
        assert "feature2" not in result_08.columns
        assert "feature3" in result_08.columns
        assert "target" in result_08.columns
        assert len(result_08.columns) == 2

    def test_data_integrity(self):
        """Test that data values are not modified."""
        df = pd.DataFrame(
            {
                "const": [5, 5, 5, 5, 5],
                "feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        # Data should be unchanged
        assert result["feature"].tolist() == [1, 2, 3, 4, 5]
        assert result["target"].tolist() == [0, 1, 0, 1, 0]

    def test_index_preserved(self):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame(
            {
                "const": [5, 5, 5, 5, 5],
                "feature": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            },
            index=[10, 20, 30, 40, 50],
        )

        result = remove_constant_features(df, "target", constant_tolerance=1.0)

        assert result.index.tolist() == [10, 20, 30, 40, 50]
