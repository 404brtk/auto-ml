"""Tests for SimpleDateTimeFeatures and SimpleTimeFeatures transformers."""

import pandas as pd
import numpy as np
import pytest

from auto_ml_pipeline.transformers import SimpleDateTimeFeatures, SimpleTimeFeatures


@pytest.fixture
def datetime_transformer():
    """Fixture for SimpleDateTimeFeatures transformer."""
    return SimpleDateTimeFeatures()


@pytest.fixture
def time_transformer():
    """Fixture for SimpleTimeFeatures transformer."""
    return SimpleTimeFeatures()


class TestSimpleDateTimeFeatures:
    """Test SimpleDateTimeFeatures transformer."""

    def test_datetime_feature_extraction(self, datetime_transformer):
        """Test basic datetime feature extraction."""
        df = pd.DataFrame(
            {"date_col": pd.date_range("2023-01-01", periods=5, freq="D")}
        )
        datetime_transformer.fit(df)
        features = datetime_transformer.transform(df)

        assert features.shape == (5, 6)
        feature_names = datetime_transformer.get_feature_names_out()
        expected_names = [
            "date_col_year",
            "date_col_month",
            "date_col_day",
            "date_col_dayofweek",
            "date_col_quarter",
            "date_col_is_weekend",
        ]
        assert list(feature_names) == expected_names

    def test_datetime_feature_values(self, datetime_transformer):
        """Test that extracted datetime features have correct values."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-02-15", "2023-12-31"])}
        )
        features = datetime_transformer.fit_transform(df)

        assert features[0, 0] == 2023
        assert features[0, 1] == 1
        assert features[0, 2] == 1
        assert features[0, 3] == 6
        assert features[0, 4] == 1
        assert features[0, 5] == 1

    def test_datetime_with_missing_values(self, datetime_transformer):
        """Test datetime feature extraction with NaN values."""
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", None, "2023-01-03"])})
        features = datetime_transformer.fit_transform(df)

        assert not np.isnan(features[0, :]).any()
        assert np.isnan(features[1, 0])
        assert features[1, 5] == 0
        assert not np.isnan(features[2, :]).any()


class TestSimpleTimeFeatures:
    """Test SimpleTimeFeatures transformer."""

    def test_feature_names(self, time_transformer):
        """Test that feature names are correctly generated."""
        df = pd.DataFrame({"time_col": ["14:30:00"]})
        time_transformer.fit(df)
        feature_names = time_transformer.get_feature_names_out()
        expected_names = [
            "time_col_hour",
            "time_col_minute",
            "time_col_second",
            "time_col_is_business_hours",
            "time_col_time_category",
        ]
        assert list(feature_names) == expected_names

    @pytest.mark.parametrize(
        "time_str, expected",
        [
            ("14:30:05", [14, 30, 5, 1, 2]),
            ("09:15:30", [9, 15, 30, 1, 1]),
            ("18:45", [18, 45, 0, 0, 3]),
            ("23:59", [23, 59, 0, 0, 3]),
            ("8", [8, 0, 0, 0, 1]),
        ],
    )
    def test_time_feature_values(self, time_transformer, time_str, expected):
        """Test that time features are extracted correctly for various formats."""
        df = pd.DataFrame({"time": [time_str]})
        features = time_transformer.fit_transform(df)
        np.testing.assert_array_equal(features[0], expected)

    def test_time_with_missing_values(self, time_transformer):
        """Test time feature extraction with missing values."""
        df = pd.DataFrame({"time": ["14:30:00", None, "18:45:00"]})
        features = time_transformer.fit_transform(df)
        assert not np.isnan(features).any()
        assert features[1, 0] == 0

    def test_time_category_mapping(self, time_transformer):
        """Test that time categories are correctly assigned."""
        df = pd.DataFrame({"time": ["03:00:00", "08:00:00", "15:00:00", "20:00:00"]})
        features = time_transformer.fit_transform(df)
        assert features[0, 4] == 0
        assert features[1, 4] == 1
        assert features[2, 4] == 2
        assert features[3, 4] == 3

    def test_invalid_time_format(self, time_transformer):
        """Test that invalid time formats are handled gracefully."""
        df = pd.DataFrame({"time": ["invalid-time", "14:30:00"]})
        features = time_transformer.fit_transform(df)
        assert (features[0] == 0).all()
        assert features[1, 0] == 14

    def test_all_nan_column(self, time_transformer):
        """Test that a column with all NaNs is handled correctly."""
        df = pd.DataFrame({"time": [None, np.nan]})
        features = time_transformer.fit_transform(df)
        assert features.shape == (2, 5)
        assert (features == 0).all()
