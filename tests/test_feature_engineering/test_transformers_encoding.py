import pandas as pd
import numpy as np
from auto_ml_pipeline.transformers.encoding import FrequencyEncoder


class TestFrequencyEncoder:
    """Test FrequencyEncoder transformer."""

    def test_basic_frequency_encoding(self):
        """Test basic frequency encoding."""
        df = pd.DataFrame(
            {
                "category": [
                    "A",
                    "B",
                    "A",
                    "C",
                    "A",
                    "B",
                ]  # A=3/6=0.5, B=2/6=0.33, C=1/6=0.17
            }
        )

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # Check frequencies
        assert np.isclose(result[0, 0], 0.5)  # A
        assert np.isclose(result[1, 0], 0.333, atol=0.01)  # B
        assert np.isclose(result[3, 0], 0.167, atol=0.01)  # C

    def test_multiple_columns(self):
        """Test encoding with multiple columns."""
        df = pd.DataFrame({"cat1": ["A", "B", "A", "B"], "cat2": ["X", "X", "Y", "Y"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        assert result.shape == (4, 2)
        # cat1: A=0.5, B=0.5
        assert np.isclose(result[0, 0], 0.5)
        # cat2: X=0.5, Y=0.5
        assert np.isclose(result[0, 1], 0.5)

    def test_unseen_categories(self):
        """Test handling of unseen categories in transform."""
        df_train = pd.DataFrame({"cat": ["A", "B", "A"]})
        df_test = pd.DataFrame({"cat": ["A", "C"]})  # C is unseen

        encoder = FrequencyEncoder()
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        # A should have frequency 2/3
        assert np.isclose(result[0, 0], 0.667, atol=0.01)
        # C is unseen, should be filled with 0
        assert result[1, 0] == 0.0

    def test_missing_values_in_fit(self):
        """Test handling of NaN values during fit."""
        df = pd.DataFrame({"cat": ["A", "B", np.nan, "A", "B", np.nan]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # Should encode NaN as well
        assert result.shape == (6, 1)
        # A=2/6, B=2/6, NaN=2/6
        assert np.isclose(result[0, 0], 0.333, atol=0.01)

    def test_missing_values_in_transform(self):
        """Test handling of NaN values during transform."""
        df_train = pd.DataFrame({"cat": ["A", "B", "A"]})
        df_test = pd.DataFrame({"cat": ["A", np.nan]})

        encoder = FrequencyEncoder()
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        # NaN was not seen in training, should be 0
        assert result[1, 0] == 0.0

    def test_single_category(self):
        """Test with single category value."""
        df = pd.DataFrame({"cat": ["A", "A", "A"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # All should be 1.0
        assert np.all(result == 1.0)

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        arr = np.array([["A"], ["B"], ["A"], ["B"]])

        encoder = FrequencyEncoder()
        encoder.fit(arr)
        result = encoder.transform(arr)

        assert result.shape == (4, 1)
        assert np.isclose(result[0, 0], 0.5)

    def test_1d_array_input(self):
        """Test with 1D array input."""
        arr = np.array(["A", "B", "A", "B"])

        encoder = FrequencyEncoder()
        encoder.fit(arr)
        result = encoder.transform(arr)

        assert result.shape == (4, 1)
        assert np.isclose(result[0, 0], 0.5)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"cat": []})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        assert result.shape == (0, 1)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        df = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)

        feature_names = encoder.get_feature_names_out()
        assert list(feature_names) == ["cat1", "cat2"]

    def test_transform_preserves_order(self):
        """Test that transform preserves row order."""
        df = pd.DataFrame({"cat": ["C", "A", "B", "A", "C"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # C=2/5, A=2/5, B=1/5
        assert np.isclose(result[0, 0], 0.4)  # C
        assert np.isclose(result[1, 0], 0.4)  # A
        assert np.isclose(result[2, 0], 0.2)  # B
        assert np.isclose(result[3, 0], 0.4)  # A
        assert np.isclose(result[4, 0], 0.4)  # C

    def test_numeric_categories(self):
        """Test encoding of numeric categories."""
        df = pd.DataFrame({"cat": [1, 2, 1, 3, 1]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # 1=3/5=0.6, 2=1/5=0.2, 3=1/5=0.2
        assert np.isclose(result[0, 0], 0.6)
        assert np.isclose(result[1, 0], 0.2)

    def test_mixed_type_categories(self):
        """Test with mixed type values."""
        df = pd.DataFrame({"cat": ["A", 1, "A", 1]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # Should handle mixed types
        assert result.shape == (4, 1)

    def test_high_cardinality(self):
        """Test with high cardinality data."""
        # Many unique values
        df = pd.DataFrame({"cat": [f"cat_{i}" for i in range(100)]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # Each category appears once, so all frequencies should be 1/100 = 0.01
        assert np.allclose(result, 0.01)

    def test_transform_with_missing_columns(self):
        """Test transform when some columns are missing."""
        df_train = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})
        df_test = pd.DataFrame({"cat1": ["A", "A"]})  # cat2 missing

        encoder = FrequencyEncoder()
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        # Should have 2 columns (cat2 filled with 0)
        assert result.shape == (2, 2)
        assert np.all(result[:, 1] == 0.0)  # cat2 column filled with 0

    def test_transform_with_extra_columns(self):
        """Test transform when test has extra columns."""
        df_train = pd.DataFrame({"cat1": ["A", "B"]})
        df_test = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})

        encoder = FrequencyEncoder()
        encoder.fit(df_train)
        result = encoder.transform(df_test)

        # Should only return fitted columns
        assert result.shape == (2, 1)

    def test_very_rare_categories(self):
        """Test encoding with very rare categories."""
        # One category appears 99 times, another once
        df = pd.DataFrame({"cat": ["common"] * 99 + ["rare"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # common = 99/100 = 0.99
        assert np.isclose(result[0, 0], 0.99)
        # rare = 1/100 = 0.01
        assert np.isclose(result[-1, 0], 0.01)

    def test_consistent_encoding_across_transforms(self):
        """Test that encoding is consistent across multiple transforms."""
        df_train = pd.DataFrame({"cat": ["A", "B", "A"]})
        df_test1 = pd.DataFrame({"cat": ["A", "B"]})
        df_test2 = pd.DataFrame({"cat": ["B", "A"]})

        encoder = FrequencyEncoder()
        encoder.fit(df_train)
        result1 = encoder.transform(df_test1)
        result2 = encoder.transform(df_test2)

        # A should always be 2/3, B should always be 1/3
        assert np.isclose(result1[0, 0], result2[1, 0])  # Both A
        assert np.isclose(result1[1, 0], result2[0, 0])  # Both B

    def test_string_numbers(self):
        """Test with string representations of numbers."""
        df = pd.DataFrame({"cat": ["1", "2", "1", "2", "1"]})

        encoder = FrequencyEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # "1" = 3/5 = 0.6
        assert np.isclose(result[0, 0], 0.6)
