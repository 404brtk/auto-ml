import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.transformers.cleaning import NumericLikeCoercer


class TestNumericLikeCoercer:
    """Test NumericLikeCoercer transformer."""

    def test_basic_numeric_conversion(self):
        """Test basic string to numeric conversion."""
        df = pd.DataFrame(
            {"num_str": ["1", "2", "3", "4"], "other": ["a", "b", "c", "d"]}
        )

        coercer = NumericLikeCoercer(threshold=0.8)
        coercer.fit(df)
        result = coercer.transform(df)

        assert "num_str" in coercer.convert_cols_
        assert pd.api.types.is_numeric_dtype(result["num_str"])
        assert result["num_str"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_decimal_numbers(self):
        """Test conversion of decimal numbers."""
        df = pd.DataFrame({"decimals": ["1.5", "2.7", "3.14", "4.0"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["decimals"])
        assert result["decimals"].tolist() == [1.5, 2.7, 3.14, 4.0]

    def test_us_number_format_with_commas(self):
        """Test US format: 1,234.56"""
        df = pd.DataFrame({"numbers": ["1,234.56", "2,345.67", "3,456.78", "4,567.89"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234.56, 2345.67, 3456.78, 4567.89]

    def test_european_number_format(self):
        """Test European format: 1.234,56"""
        df = pd.DataFrame({"numbers": ["1.234,56", "2.345,67", "3.456,78", "4.567,89"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234.56, 2345.67, 3456.78, 4567.89]

    def test_scientific_notation(self):
        """Test scientific notation conversion."""
        df = pd.DataFrame({"sci": ["1.23e10", "4.56e-5", "7.89e2", "1e3"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["sci"])
        assert np.isclose(result["sci"].iloc[0], 1.23e10)
        assert np.isclose(result["sci"].iloc[1], 4.56e-5)

    def test_threshold_filtering(self):
        """Test that columns below threshold are not converted."""
        df = pd.DataFrame(
            {
                "mostly_numeric": ["1", "2", "3", "not_a_number"],  # 75% numeric
                "all_text": ["a", "b", "c", "d"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.8)  # Need 80%
        coercer.fit(df)
        result = coercer.transform(df)

        # mostly_numeric should NOT be converted (only 75% < 80%)
        assert "mostly_numeric" not in coercer.convert_cols_
        assert not pd.api.types.is_numeric_dtype(result["mostly_numeric"])

        # all_text should definitely not be converted
        assert "all_text" not in coercer.convert_cols_

    def test_high_threshold(self):
        """Test that high threshold converts appropriate columns."""
        df = pd.DataFrame(
            {
                "perfect": ["1", "2", "3", "4"],  # 100% numeric
                "mostly": ["1", "2", "3", "bad"],  # 75% numeric
            }
        )

        coercer = NumericLikeCoercer(threshold=0.95)
        coercer.fit(df)

        assert "perfect" in coercer.convert_cols_
        assert "mostly" not in coercer.convert_cols_

    def test_empty_strings_and_nan(self):
        """Test handling of empty strings and NaN values."""
        df = pd.DataFrame(
            {
                "with_empty": ["1", "2", "", "4", np.nan],
                "with_nan": ["1.5", np.nan, "3.5", "4.5", "5.5"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.6)
        result = coercer.fit_transform(df)

        # Should convert and replace empty/invalid with NaN
        assert pd.api.types.is_numeric_dtype(result["with_empty"])
        assert pd.api.types.is_numeric_dtype(result["with_nan"])
        assert result["with_empty"].isna().sum() == 2  # Empty and original NaN

    def test_thousand_separator_explicit(self):
        """Test explicit thousand separator."""
        df = pd.DataFrame({"numbers": ["1,234", "5,678", "9,012"]})

        coercer = NumericLikeCoercer(threshold=0.8, thousand_sep=",")
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234.0, 5678.0, 9012.0]

    def test_already_numeric_columns_ignored(self):
        """Test that already numeric columns are not processed."""
        df = pd.DataFrame(
            {"already_numeric": [1, 2, 3, 4], "string_numeric": ["5", "6", "7", "8"]}
        )

        coercer = NumericLikeCoercer(threshold=0.8)
        coercer.fit(df)

        # Should only convert the string column
        assert "string_numeric" in coercer.convert_cols_
        assert "already_numeric" not in coercer.convert_cols_

    def test_non_dataframe_input(self):
        """Test that non-DataFrame input is handled gracefully."""
        array = np.array([[1, 2], [3, 4]])

        coercer = NumericLikeCoercer()
        result = coercer.fit_transform(array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, array)

    def test_sampling_large_dataset(self):
        """Test that large datasets are sampled for efficiency."""
        # Create a large dataset
        large_df = pd.DataFrame({"col": [str(i) for i in range(20000)]})

        coercer = NumericLikeCoercer(threshold=0.95, sample_size=5000)
        coercer.fit(large_df)

        # Should still detect and convert
        assert "col" in coercer.convert_cols_
        assert coercer.conversion_stats_["col"]["sample_size"] == 5000

    def test_conversion_stats(self):
        """Test that conversion statistics are recorded."""
        df = pd.DataFrame({"nums": ["1", "2", "3", "4"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        coercer.fit(df)

        assert "nums" in coercer.conversion_stats_
        stats = coercer.conversion_stats_["nums"]
        assert "conversion_rate" in stats
        assert "format_info" in stats
        assert stats["conversion_rate"] == 1.0

    def test_negative_numbers(self):
        """Test conversion of negative numbers."""
        df = pd.DataFrame({"negatives": ["-1", "-2.5", "-3.14", "-100"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["negatives"])
        assert result["negatives"].tolist() == [-1.0, -2.5, -3.14, -100.0]

    def test_mixed_formats_same_column(self):
        """Test column with mixed number formats."""
        df = pd.DataFrame({"mixed": ["1,234.56", "2345.67", "3.456,78", "4567"]})

        coercer = NumericLikeCoercer(threshold=0.5)
        result = coercer.fit_transform(df)

        # Should attempt conversion using detected dominant format
        assert "mixed" in coercer.convert_cols_
        assert result.shape == df.shape

    def test_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be in"):
            NumericLikeCoercer(threshold=0)

        with pytest.raises(ValueError, match="threshold must be in"):
            NumericLikeCoercer(threshold=1.5)

    def test_transform_without_fit(self):
        """Test that transform after fit works correctly."""
        df_train = pd.DataFrame({"nums": ["1", "2", "3"]})
        df_test = pd.DataFrame({"nums": ["4", "5", "6"]})

        coercer = NumericLikeCoercer()
        coercer.fit(df_train)
        result = coercer.transform(df_test)

        assert pd.api.types.is_numeric_dtype(result["nums"])
        assert result["nums"].tolist() == [4.0, 5.0, 6.0]

    def test_missing_column_in_transform(self):
        """Test transform when fitted column is missing."""
        df_train = pd.DataFrame({"nums": ["1", "2", "3"]})
        df_test = pd.DataFrame({"other": ["a", "b", "c"]})

        coercer = NumericLikeCoercer()
        coercer.fit(df_train)
        result = coercer.transform(df_test)

        # Should return unchanged
        assert "nums" not in result.columns
        assert "other" in result.columns

    def test_spaces_and_quotes_in_numbers(self):
        """Test numbers with spaces and quotes."""
        df = pd.DataFrame(
            {
                "with_spaces": ["1 234", "2 345", "3 456"],
                "with_quotes": ["1'234", "2'345", "3'456"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Spaces and quotes should be stripped
        assert pd.api.types.is_numeric_dtype(result["with_spaces"])
        assert pd.api.types.is_numeric_dtype(result["with_quotes"])

    def test_category_dtype_column(self):
        """Test that category dtype columns are processed."""
        df = pd.DataFrame({"cat": pd.Categorical(["1", "2", "3", "4"])})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert "cat" in coercer.convert_cols_
        assert pd.api.types.is_numeric_dtype(result["cat"])
