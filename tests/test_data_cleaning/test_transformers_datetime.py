import pandas as pd
import numpy as np
from auto_ml_pipeline.transformers.datetime import (
    DateTimeConverter,
    TimeConverter,
    _detect_datetime_patterns,
    _detect_time_patterns,
)


class TestDateTimeConverter:
    """Test DateTimeConverter transformer."""

    def test_iso_format_datetime(self):
        """Test ISO format datetime conversion."""
        df = pd.DataFrame({"date": ["2023-01-01", "2023-02-15", "2023-12-31"]})

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "date" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_us_format_datetime(self):
        """Test US format MM/DD/YYYY."""
        df = pd.DataFrame({"date": ["01/15/2023", "02/20/2023", "12/31/2023"]})

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "date" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_european_format_datetime(self):
        """Test European format DD/MM/YYYY."""
        df = pd.DataFrame({"date": ["15/01/2023", "20/02/2023", "31/12/2023"]})

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "date" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_datetime_with_time(self):
        """Test datetime with time component."""
        df = pd.DataFrame(
            {
                "datetime": [
                    "2023-01-01 14:30:00",
                    "2023-02-15 09:15:30",
                    "2023-12-31 23:59:59",
                ]
            }
        )

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "datetime" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_month_names(self):
        """Test dates with month names."""
        df = pd.DataFrame({"date": ["Jan 15, 2023", "Feb 20, 2023", "Dec 31, 2023"]})

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "date" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_mixed_valid_invalid(self):
        """Test column with some invalid dates."""
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-02-15", "not_a_date", "2023-12-31"]}
        )

        converter = DateTimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        # Should still convert if 75% are valid (above 70% threshold)
        assert "date" in converter.datetime_cols_
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        # Invalid entries become NaT
        assert pd.isna(result["date"].iloc[2])

    def test_low_confidence_not_converted(self):
        """Test that low confidence columns are not converted."""
        df = pd.DataFrame(
            {"mostly_text": ["text1", "text2", "2023-01-01", "text3", "text4"]}
        )

        converter = DateTimeConverter(confidence_threshold=0.7)
        converter.fit(df)

        # Only 20% are dates, below 70% threshold
        assert "mostly_text" not in converter.datetime_cols_

    def test_confidence_threshold(self):
        """Test custom confidence threshold."""
        # Use more samples for reliable detection
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-15"] * 5
                + ["bad"] * 3  # ~77% valid (10/13)
            }
        )

        # With 80% threshold, should not convert
        converter_high = DateTimeConverter(confidence_threshold=0.8)
        converter_high.fit(df)
        assert "date" not in converter_high.datetime_cols_

        # With 70% threshold, should convert
        converter_low = DateTimeConverter(confidence_threshold=0.7)
        converter_low.fit(df)
        assert "date" in converter_low.datetime_cols_

    def test_already_datetime_column(self):
        """Test that already datetime columns are not processed."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3)})

        converter = DateTimeConverter()
        converter.fit(df)

        # Already datetime, should not be in object columns
        assert "date" not in converter.datetime_cols_

    def test_time_only_not_detected_as_datetime(self):
        """Test that time-only columns are not detected as datetime."""
        df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "18:45:00", "12:00:00"]})

        converter = DateTimeConverter()
        converter.fit(df)

        # Should not be detected as datetime (should be handled by TimeConverter)
        assert "time" not in converter.datetime_cols_

    def test_non_dataframe_input(self):
        """Test that non-DataFrame input is handled gracefully."""
        array = np.array([[1, 2], [3, 4]])

        converter = DateTimeConverter()
        result = converter.fit_transform(array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, array)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        converter = DateTimeConverter()
        result = converter.fit_transform(df)

        assert len(converter.datetime_cols_) == 0
        assert len(result) == 0

    def test_all_nan_column(self):
        """Test column with all NaN values."""
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})

        converter = DateTimeConverter()
        converter.fit(df)

        assert "all_nan" not in converter.datetime_cols_

    def test_sample_size_parameter(self):
        """Test that sample size parameter works."""
        # Create large dataset
        large_df = pd.DataFrame({"date": ["2023-01-01"] * 5000})

        converter = DateTimeConverter(sample_size=100)
        converter.fit(large_df)

        assert "date" in converter.datetime_cols_

    def test_different_separators(self):
        """Test dates with different separators."""
        df_dash = pd.DataFrame({"date": ["2023-01-15", "2023-02-20"]})
        df_slash = pd.DataFrame({"date": ["2023/01/15", "2023/02/20"]})
        df_dot = pd.DataFrame({"date": ["2023.01.15", "2023.02.20"]})

        for df in [df_dash, df_slash, df_dot]:
            converter = DateTimeConverter()
            result = converter.fit_transform(df)
            assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_transform_on_new_data(self):
        """Test transform on different data than fit."""
        df_train = pd.DataFrame({"date": ["2023-01-01", "2023-02-15"]})
        df_test = pd.DataFrame({"date": ["2023-03-20", "2023-04-25"]})

        converter = DateTimeConverter()
        converter.fit(df_train)
        result = converter.transform(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_missing_column_in_transform(self):
        """Test transform when fitted column is missing."""
        df_train = pd.DataFrame({"date": ["2023-01-01", "2023-02-15"]})
        df_test = pd.DataFrame({"other": ["a", "b"]})

        converter = DateTimeConverter()
        converter.fit(df_train)
        result = converter.transform(df_test)

        # Should return unchanged
        assert "date" not in result.columns


class TestTimeConverter:
    """Test TimeConverter transformer."""

    def test_basic_time_conversion(self):
        """Test basic time conversion HH:MM:SS."""
        df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "18:45:00"]})

        converter = TimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "time" in converter.time_cols_
        # Should be normalized to HH:MM:SS strings
        assert result["time"].iloc[0] == "14:30:00"

    def test_time_without_seconds(self):
        """Test time format HH:MM."""
        df = pd.DataFrame({"time": ["14:30", "09:15", "18:45"]})

        converter = TimeConverter(confidence_threshold=0.7)
        converter.fit_transform(df)

        assert "time" in converter.time_cols_

    def test_12hour_format_am_pm(self):
        """Test 12-hour format with AM/PM."""
        df = pd.DataFrame({"time": ["02:30 PM", "09:15 AM", "11:45 PM"]})

        converter = TimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "time" in converter.time_cols_
        # Should convert to 24-hour format
        assert result["time"].iloc[0] == "14:30:00"  # 2:30 PM -> 14:30:00

    def test_mixed_valid_invalid_times(self):
        """Test column with some invalid times."""
        df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "not_a_time", "18:45:00"]})

        converter = TimeConverter(confidence_threshold=0.7)
        result = converter.fit_transform(df)

        assert "time" in converter.time_cols_
        # Invalid entries should be None
        assert result["time"].iloc[2] is None

    def test_low_confidence_not_converted(self):
        """Test that low confidence columns are not converted."""
        df = pd.DataFrame(
            {"mostly_text": ["text1", "text2", "14:30:00", "text3", "text4"]}
        )

        converter = TimeConverter(confidence_threshold=0.7)
        converter.fit(df)

        # Only 20% are times, below 70% threshold
        assert "mostly_text" not in converter.time_cols_

    def test_already_time_string(self):
        """Test that already normalized time strings work."""
        df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "18:45:00"]})

        converter = TimeConverter()
        result = converter.fit_transform(df)

        assert "time" in converter.time_cols_
        # Should remain as HH:MM:SS
        assert result["time"].iloc[0] == "14:30:00"

    def test_non_dataframe_input(self):
        """Test that non-DataFrame input is handled gracefully."""
        array = np.array([[1, 2], [3, 4]])

        converter = TimeConverter()
        result = converter.fit_transform(array)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, array)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        converter = TimeConverter()
        result = converter.fit_transform(df)

        assert len(converter.time_cols_) == 0
        assert len(result) == 0

    def test_all_nan_column(self):
        """Test column with all NaN values."""
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})

        converter = TimeConverter()
        converter.fit(df)

        assert "all_nan" not in converter.time_cols_

    def test_confidence_threshold(self):
        """Test custom confidence threshold."""
        # Use more samples for reliable detection
        df = pd.DataFrame(
            {"time": ["14:30:00", "09:15:30"] * 5 + ["bad"] * 3}  # ~77% valid (10/13)
        )

        # With 80% threshold, should not convert
        converter_high = TimeConverter(confidence_threshold=0.8)
        converter_high.fit(df)
        assert "time" not in converter_high.time_cols_

        # With 70% threshold, should convert
        converter_low = TimeConverter(confidence_threshold=0.7)
        converter_low.fit(df)
        assert "time" in converter_low.time_cols_

    def test_transform_on_new_data(self):
        """Test transform on different data than fit."""
        df_train = pd.DataFrame({"time": ["14:30:00", "09:15:30"]})
        df_test = pd.DataFrame({"time": ["18:45:00", "12:00:00"]})

        converter = TimeConverter()
        converter.fit(df_train)
        result = converter.transform(df_test)

        assert result["time"].iloc[0] == "18:45:00"

    def test_midnight_and_noon(self):
        """Test midnight and noon times."""
        df = pd.DataFrame({"time": ["00:00:00", "12:00:00", "23:59:59"]})

        converter = TimeConverter()
        result = converter.fit_transform(df)

        assert result["time"].iloc[0] == "00:00:00"  # Midnight
        assert result["time"].iloc[1] == "12:00:00"  # Noon
        assert result["time"].iloc[2] == "23:59:59"  # End of day


class TestDetectDatetimePatterns:
    """Test _detect_datetime_patterns function."""

    def test_detect_iso_format(self):
        """Test detection of ISO format."""
        series = pd.Series(["2023-01-01", "2023-02-15", "2023-12-31"])

        result = _detect_datetime_patterns(series)

        assert result["is_datetime"] is True
        assert result["confidence"] >= 0.7
        assert result["format"] is not None

    def test_detect_no_datetime(self):
        """Test detection when no datetime present."""
        series = pd.Series(["text1", "text2", "text3"])

        result = _detect_datetime_patterns(series)

        assert result["is_datetime"] is False
        assert result["confidence"] == 0.0

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([])

        result = _detect_datetime_patterns(series)

        assert result["is_datetime"] is False
        assert result["confidence"] == 0.0


class TestDetectTimePatterns:
    """Test _detect_time_patterns function."""

    def test_detect_time_format(self):
        """Test detection of time format."""
        series = pd.Series(["14:30:00", "09:15:30", "18:45:00"])

        result = _detect_time_patterns(series)

        assert result["is_time"] is True
        assert result["confidence"] >= 0.7
        assert result["format"] is not None

    def test_detect_no_time(self):
        """Test detection when no time present."""
        series = pd.Series(["text1", "text2", "text3"])

        result = _detect_time_patterns(series)

        assert result["is_time"] is False
        assert result["confidence"] == 0.0

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([])

        result = _detect_time_patterns(series)

        assert result["is_time"] is False
        assert result["confidence"] == 0.0
