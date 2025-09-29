"""Tests for ColumnTypes dataclass."""

from auto_ml_pipeline.feature_engineering import ColumnTypes


class TestColumnTypes:
    """Test ColumnTypes dataclass."""

    def test_column_types_creation(self):
        """Test basic ColumnTypes instantiation."""
        ct = ColumnTypes(
            numeric=["col1"],
            categorical_low=["col2"],
            categorical_high=["col3"],
            datetime=["col4"],
            time=["col5"],
            text=["col6"],
        )
        assert ct.numeric == ["col1"]
        assert ct.categorical_low == ["col2"]
        assert ct.categorical_high == ["col3"]
        assert ct.datetime == ["col4"]
        assert ct.time == ["col5"]
        assert ct.text == ["col6"]

    def test_column_types_empty_lists(self):
        """Test ColumnTypes with empty lists."""
        ct = ColumnTypes(
            numeric=[],
            categorical_low=[],
            categorical_high=[],
            datetime=[],
            time=[],
            text=[],
        )
        assert ct.numeric == []
        assert ct.categorical_low == []
        assert ct.categorical_high == []
        assert ct.datetime == []
        assert ct.time == []
        assert ct.text == []

    def test_column_types_multiple_columns(self):
        """Test ColumnTypes with multiple columns per type."""
        ct = ColumnTypes(
            numeric=["num1", "num2", "num3"],
            categorical_low=["cat1", "cat2"],
            categorical_high=["high1"],
            datetime=["dt1", "dt2"],
            time=["time1"],
            text=["text1", "text2"],
        )
        assert len(ct.numeric) == 3
        assert len(ct.categorical_low) == 2
        assert len(ct.categorical_high) == 1
        assert len(ct.datetime) == 2
        assert len(ct.time) == 1
        assert len(ct.text) == 2
