from auto_ml_pipeline.feature_engineering import ColumnTypes


def test_column_types_creation():
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
