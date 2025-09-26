import pandas as pd
import numpy as np

from auto_ml_pipeline.transformers import SimpleDateTimeFeatures, SimpleTimeFeatures


class TestSimpleDateTimeFeatures:
    """Test SimpleDateTimeFeatures transformer."""

    def test_datetime_feature_extraction(self):
        """Test basic datetime feature extraction."""
        df = pd.DataFrame(
            {"date_col": pd.date_range("2023-01-01", periods=5, freq="D")}
        )

        transformer = SimpleDateTimeFeatures()
        transformer.fit(df)
        features = transformer.transform(df)

        # Should extract 6 features: year, month, day, dayofweek, quarter, is_weekend
        assert features.shape == (5, 6)
        assert features.shape[1] == 6

        # Check feature names
        feature_names = transformer.get_feature_names_out()
        expected_names = [
            "date_col_year",
            "date_col_month",
            "date_col_day",
            "date_col_dayofweek",
            "date_col_quarter",
            "date_col_is_weekend",
        ]
        assert list(feature_names) == expected_names

    def test_datetime_feature_values(self):
        """Test that extracted datetime features have correct values."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-02-15", "2023-12-31"])}
        )

        transformer = SimpleDateTimeFeatures()
        features = transformer.fit_transform(df)

        # Check specific values
        assert features[0, 0] == 2023  # year
        assert features[0, 1] == 1  # month
        assert features[0, 2] == 1  # day
        assert features[0, 3] == 6  # dayofweek (Sunday = 6)
        assert features[0, 4] == 1  # quarter
        assert features[0, 5] == 1  # is_weekend (Sunday = True)

        assert features[1, 1] == 2  # February
        assert features[1, 2] == 15  # 15th
        assert features[2, 1] == 12  # December
        assert features[2, 4] == 4  # Q4

    def test_datetime_with_missing_values(self):
        """Test datetime feature extraction with NaN values."""
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", None, "2023-01-03"])})

        transformer = SimpleDateTimeFeatures()
        features = transformer.fit_transform(df)

        # NaN datetime values result in NaN for most features, but is_weekend becomes 0
        # Row 0: valid date
        assert not np.isnan(features[0, :]).any()
        # Row 1: NaT date - most features become NaN, but is_weekend becomes 0
        assert np.isnan(features[1, 0])  # year
        assert np.isnan(features[1, 1])  # month
        assert np.isnan(features[1, 2])  # day
        assert np.isnan(features[1, 3])  # dayofweek
        assert np.isnan(features[1, 4])  # quarter
        assert features[1, 5] == 0  # is_weekend (False for NaT)
        # Row 2: valid date
        assert not np.isnan(features[2, :]).any()


class TestSimpleTimeFeatures:
    """Test SimpleTimeFeatures transformer."""

    def test_time_feature_extraction(self):
        """Test basic time feature extraction."""
        df = pd.DataFrame(
            {"time_col": ["14:30:00", "09:15:30", "18:45:00", "12:00:00", "23:59:00"]}
        )

        transformer = SimpleTimeFeatures()
        transformer.fit(df)
        features = transformer.transform(df)

        # Should extract 5 features: hour, minute, second, is_business_hours, time_category
        assert features.shape == (5, 5)

        # Check feature names
        feature_names = transformer.get_feature_names_out()
        expected_names = [
            "time_col_hour",
            "time_col_minute",
            "time_col_second",
            "time_col_is_business_hours",
            "time_col_time_category",
        ]
        assert list(feature_names) == expected_names

    def test_time_feature_values(self):
        """Test that extracted time features have correct values."""
        df = pd.DataFrame({"time": ["14:30:00", "09:15:30", "18:45:00", "23:59:00"]})

        transformer = SimpleTimeFeatures()
        features = transformer.fit_transform(df)

        # Check specific values
        assert features[0, 0] == 14  # hour
        assert features[0, 1] == 30  # minute
        assert features[0, 2] == 0  # second
        assert features[0, 3] == 1  # is_business_hours (2-6 PM = True)
        assert features[0, 4] == 2  # afternoon category

        assert features[1, 0] == 9  # morning hour
        assert features[1, 3] == 1  # business hours
        assert features[1, 4] == 1  # morning category

        assert features[2, 0] == 18  # evening hour
        assert features[2, 3] == 0  # not business hours
        assert features[2, 4] == 3  # evening category

        assert features[3, 0] == 23  # late night hour
        assert features[3, 4] == 3  # evening category (18-24)

    def test_time_with_missing_values(self):
        """Test time feature extraction with missing values."""
        df = pd.DataFrame({"time": ["14:30:00", None, "18:45:00"]})

        transformer = SimpleTimeFeatures()
        features = transformer.fit_transform(df)

        # Should handle NaN by filling with most frequent
        assert not np.isnan(features).any()

    def test_time_category_mapping(self):
        """Test that time categories are correctly assigned."""
        df = pd.DataFrame(
            {"time": ["06:00:00", "12:00:00", "18:00:00", "00:00:00", "03:00:00"]}
        )

        transformer = SimpleTimeFeatures()
        features = transformer.fit_transform(df)

        # Categories: 0=night (0-6), 1=morning (6-12), 2=afternoon (12-18), 3=evening (18-24)
        assert features[0, 4] == 1  # 6 AM = morning
        assert features[1, 4] == 2  # 12 PM = afternoon
        assert features[2, 4] == 3  # 6 PM = evening
        assert features[3, 4] == 0  # 12 AM = night
        assert features[4, 4] == 0  # 3 AM = night
