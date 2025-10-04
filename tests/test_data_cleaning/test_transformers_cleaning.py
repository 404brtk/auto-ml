import pandas as pd
import numpy as np
import pytest
from auto_ml_pipeline.transformers.cleaning import NumericLikeCoercer


class TestNumericLikeCoercer:
    """Test NumericLikeCoercer transformer."""

    def test_basic_numeric_conversion(self):
        """Test basic string to numeric conversion with integer detection."""
        df = pd.DataFrame(
            {"num_str": ["1", "2", "3", "4"], "other": ["a", "b", "c", "d"]}
        )

        coercer = NumericLikeCoercer(threshold=0.8, detect_integers=True)
        coercer.fit(df)
        result = coercer.transform(df)

        assert "num_str" in coercer.convert_cols_
        assert pd.api.types.is_numeric_dtype(result["num_str"])
        # Should detect as integers (no NaN values), uses int32/int64
        assert pd.api.types.is_integer_dtype(result["num_str"])
        assert result["num_str"].dtype in [np.int32, np.int64]
        assert result["num_str"].tolist() == [1, 2, 3, 4]

    def test_decimal_numbers(self):
        """Test conversion of decimal numbers."""
        df = pd.DataFrame({"decimals": ["1.5", "2.7", "3.14", "4.0"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["decimals"])
        assert pd.api.types.is_float_dtype(result["decimals"])
        assert result["decimals"].tolist() == [1.5, 2.7, 3.14, 4.0]

    def test_integer_detection_disabled(self):
        """Test that integer detection can be disabled."""
        df = pd.DataFrame({"nums": ["1", "2", "3", "4"]})

        coercer = NumericLikeCoercer(threshold=0.8, detect_integers=False)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_float_dtype(result["nums"])
        assert result["nums"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_nullable_integers_with_nan(self):
        """Test that float64 is used when NaN present (sklearn compatibility)."""
        df = pd.DataFrame({"nums": ["1", "2", np.nan, "4"]})

        coercer = NumericLikeCoercer(threshold=0.5, detect_integers=True)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["nums"])
        # Should use float64 when NaN present (not nullable integers for sklearn compatibility)
        assert pd.api.types.is_float_dtype(result["nums"])
        assert result["nums"].dtype == np.float64
        assert result["nums"].iloc[0] == 1.0
        assert pd.isna(result["nums"].iloc[2])

    def test_us_number_format_with_commas(self):
        """Test US format: 1,234.56"""
        df = pd.DataFrame({"numbers": ["1,234.56", "2,345.67", "3,456.78", "4,567.89"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234.56, 2345.67, 3456.78, 4567.89]
        # Verify US format was detected
        assert coercer.conversion_stats_["numbers"]["format_info"]["decimal_sep"] == "."

    def test_us_number_format_thousands_only(self):
        """Test US format with thousands separator only: 1,234"""
        df = pd.DataFrame({"numbers": ["1,234", "5,678", "9,012", "12,345"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234, 5678, 9012, 12345]

    def test_us_number_format_multiple_thousands(self):
        """Test US format with multiple thousands separators: 1,234,567"""
        df = pd.DataFrame({"numbers": ["1,234,567", "2,345,678", "3,456,789"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234567, 2345678, 3456789]

    def test_european_number_format(self):
        """Test European format: 1.234,56"""
        df = pd.DataFrame({"numbers": ["1.234,56", "2.345,67", "3.456,78", "4.567,89"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234.56, 2345.67, 3456.78, 4567.89]
        # Verify EU format was detected
        assert coercer.conversion_stats_["numbers"]["format_info"]["decimal_sep"] == ","

    def test_european_number_format_thousands_only(self):
        """Test European format with thousands separator only: 1.234"""
        df = pd.DataFrame({"numbers": ["1.234", "5.678", "9.012", "12.345"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        # Should be detected as EU thousands separator
        assert result["numbers"].tolist() == [1234, 5678, 9012, 12345]

    def test_european_number_format_multiple_thousands(self):
        """Test European format with multiple thousands separators: 1.234.567"""
        df = pd.DataFrame({"numbers": ["1.234.567", "2.345.678", "3.456.789"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [1234567, 2345678, 3456789]

    def test_european_decimal_only(self):
        """Test European decimal format: 12,34"""
        df = pd.DataFrame({"numbers": ["12,34", "45,67", "89,01", "23,45"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["numbers"])
        assert result["numbers"].tolist() == [12.34, 45.67, 89.01, 23.45]

    def test_scientific_notation(self):
        """Test scientific notation conversion."""
        df = pd.DataFrame({"sci": ["1.23e10", "4.56e-5", "7.89e2", "1e3"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["sci"])
        assert np.isclose(result["sci"].iloc[0], 1.23e10)
        assert np.isclose(result["sci"].iloc[1], 4.56e-5)
        assert np.isclose(result["sci"].iloc[2], 789.0)
        assert np.isclose(result["sci"].iloc[3], 1000.0)

    def test_threshold_filtering(self):
        """Test that columns below threshold are not converted."""
        df = pd.DataFrame(
            {
                "mostly_numeric": ["1", "2", "3", "not_a_number"],  # 75% numeric
                "all_text": ["a", "b", "c", "d"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.8)  # Need 80%
        result = coercer.fit_transform(df)

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
        result = coercer.fit_transform(df)

        assert "perfect" in coercer.convert_cols_
        assert "mostly" not in coercer.convert_cols_
        # Verify actual conversion happened
        assert pd.api.types.is_integer_dtype(result["perfect"])

    def test_empty_strings_and_nan(self):
        """Test handling of empty strings and NaN values."""
        df = pd.DataFrame(
            {
                "with_empty": ["1", "2", "", "4", np.nan],
                "with_nan": ["1.5", np.nan, "3.5", "4.5", "5.5"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.6, detect_integers=True)
        result = coercer.fit_transform(df)

        # Should convert and replace empty/invalid with NaN
        assert pd.api.types.is_numeric_dtype(result["with_empty"])
        assert pd.api.types.is_numeric_dtype(result["with_nan"])

        # with_empty has NaN, so should be float64 even though values are integer-like
        assert pd.api.types.is_float_dtype(result["with_empty"])
        assert result["with_empty"].dtype == np.float64
        assert result["with_empty"].isna().sum() == 2  # Empty and original NaN

        # with_nan should be float (has NaN and decimals)
        assert pd.api.types.is_float_dtype(result["with_nan"])

    def test_already_numeric_columns_ignored(self):
        """Test that already numeric columns are not processed."""
        df = pd.DataFrame(
            {"already_numeric": [1, 2, 3, 4], "string_numeric": ["5", "6", "7", "8"]}
        )

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Should only convert the string column
        assert "string_numeric" in coercer.convert_cols_
        assert "already_numeric" not in coercer.convert_cols_
        # Verify the transformation actually happened
        assert pd.api.types.is_integer_dtype(result["string_numeric"])

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
        result = coercer.fit_transform(large_df)

        # Should still detect and convert
        assert "col" in coercer.convert_cols_
        assert coercer.conversion_stats_["col"]["sample_size"] == 5000
        # Verify conversion worked
        assert pd.api.types.is_numeric_dtype(result["col"])

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
        assert "decimal_sep" in stats["format_info"]
        assert "thousands_sep" in stats["format_info"]

    def test_negative_numbers(self):
        """Test conversion of negative numbers."""
        df = pd.DataFrame({"negatives": ["-1", "-2.5", "-3.14", "-100"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["negatives"])
        assert result["negatives"].tolist() == [-1.0, -2.5, -3.14, -100.0]

    def test_negative_integers(self):
        """Test conversion of negative integers with integer detection."""
        df = pd.DataFrame({"negatives": ["-1", "-2", "-3", "-100"]})

        coercer = NumericLikeCoercer(threshold=0.8, detect_integers=True)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_integer_dtype(result["negatives"])
        assert result["negatives"].dtype in [np.int32, np.int64]
        assert result["negatives"].tolist() == [-1, -2, -3, -100]

    def test_mixed_formats_same_column(self):
        """Test column with mixed number formats."""
        # Mostly US format with one ambiguous value
        df = pd.DataFrame({"mixed": ["1,234.56", "2,345.67", "3456.78", "4,567.89"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Should attempt conversion using detected dominant format (US)
        assert "mixed" in coercer.convert_cols_
        assert pd.api.types.is_numeric_dtype(result["mixed"])
        # All should convert successfully with US format
        assert result["mixed"].notna().all()

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

        coercer = NumericLikeCoercer(detect_integers=True)
        coercer.fit(df_train)
        result = coercer.transform(df_test)

        assert pd.api.types.is_numeric_dtype(result["nums"])
        assert result["nums"].tolist() == [4, 5, 6]

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

    def test_spaces_in_numbers(self):
        """Test numbers with spaces (common in some locales)."""
        df = pd.DataFrame({"with_spaces": ["1 234", "2 345", "3 456", "10 000"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Spaces should be stripped
        assert pd.api.types.is_numeric_dtype(result["with_spaces"])
        assert result["with_spaces"].tolist() == [1234, 2345, 3456, 10000]

    def test_quotes_in_numbers(self):
        """Test numbers with apostrophes (Swiss format)."""
        df = pd.DataFrame({"with_quotes": ["1'234", "2'345", "3'456", "10'000"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Apostrophes should be stripped
        assert pd.api.types.is_numeric_dtype(result["with_quotes"])
        assert result["with_quotes"].tolist() == [1234, 2345, 3456, 10000]

    def test_currency_symbols(self):
        """Test numbers with currency symbols."""
        df = pd.DataFrame(
            {
                "dollars": ["$1,234.56", "$2,345.67", "$3,456.78"],
                "euros": ["€1.234,56", "€2.345,67", "€3.456,78"],
            }
        )

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["dollars"])
        assert pd.api.types.is_numeric_dtype(result["euros"])
        assert result["dollars"].tolist() == [1234.56, 2345.67, 3456.78]
        assert result["euros"].tolist() == [1234.56, 2345.67, 3456.78]

    def test_category_dtype_column(self):
        """Test that category dtype columns are processed."""
        df = pd.DataFrame({"cat": pd.Categorical(["1", "2", "3", "4"])})

        coercer = NumericLikeCoercer(threshold=0.8, detect_integers=True)
        result = coercer.fit_transform(df)

        assert "cat" in coercer.convert_cols_
        assert pd.api.types.is_numeric_dtype(result["cat"])
        assert pd.api.types.is_integer_dtype(result["cat"])
        assert result["cat"].dtype in [np.int32, np.int64]

    def test_integer_dtype_optimization(self):
        """Test that int32 or int64 is used (sklearn compatibility)."""
        df_small = pd.DataFrame({"small": ["1", "2", "3", "4"]})
        df_medium = pd.DataFrame({"medium": ["1000", "2000", "3000"]})
        df_large = pd.DataFrame({"large": ["100000", "200000", "300000"]})
        df_very_large = pd.DataFrame({"very_large": ["10000000000", "20000000000"]})

        # All should use int32 or int64 only (sklearn compatibility)
        coercer_small = NumericLikeCoercer(detect_integers=True)
        result_small = coercer_small.fit_transform(df_small)
        assert result_small["small"].dtype in [np.int32, np.int64]

        coercer_medium = NumericLikeCoercer(detect_integers=True)
        result_medium = coercer_medium.fit_transform(df_medium)
        assert result_medium["medium"].dtype in [np.int32, np.int64]

        coercer_large = NumericLikeCoercer(detect_integers=True)
        result_large = coercer_large.fit_transform(df_large)
        assert result_large["large"].dtype in [np.int32, np.int64]

        # Values that fit in int32 should use int32
        assert result_large["large"].dtype == np.int32

        # Very large values should use int64
        coercer_very_large = NumericLikeCoercer(detect_integers=True)
        result_very_large = coercer_very_large.fit_transform(df_very_large)
        assert result_very_large["very_large"].dtype == np.int64

    def test_get_feature_names_out(self):
        """Test sklearn compatibility method."""
        coercer = NumericLikeCoercer()

        # With input features
        features = ["col1", "col2", "col3"]
        output = coercer.get_feature_names_out(features)
        assert list(output) == features

        # Without input features
        output_empty = coercer.get_feature_names_out()
        assert len(output_empty) == 0

    def test_target_column_warning(self):
        """Test that target column warnings are generated."""
        df = pd.DataFrame({"target": ["1", "2", "invalid", "4"]})

        coercer = NumericLikeCoercer(threshold=0.5, target_col="target")
        # Should log a warning about target column
        coercer.fit(df)

        # Should still convert despite warning
        assert "target" in coercer.convert_cols_

    def test_all_null_column(self):
        """Test handling of all-null columns."""
        df = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan]})

        coercer = NumericLikeCoercer(threshold=0.8)
        result = coercer.fit_transform(df)

        # Should skip all-null columns (they're already float64 with NaN by default)
        assert "all_null" not in coercer.convert_cols_
        # Pandas creates all-NaN columns as float64 by default
        assert pd.api.types.is_float_dtype(result["all_null"])
        assert result["all_null"].isna().all()

    def test_format_confidence(self):
        """Test that format detection reports confidence."""
        df = pd.DataFrame({"numbers": ["1,234.56", "2,345.67", "3,456.78"]})

        coercer = NumericLikeCoercer(threshold=0.8)
        coercer.fit(df)

        stats = coercer.conversion_stats_["numbers"]
        assert "confidence" in stats["format_info"]
        assert 0.0 <= stats["format_info"]["confidence"] <= 1.0
