import pandas as pd
from auto_ml_pipeline.data_cleaning import drop_duplicates


class TestDropDuplicates:
    """Test drop_duplicates function."""

    def test_no_duplicates(self):
        """Test with no duplicate rows."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }
        )

        result = drop_duplicates(df)
        pd.testing.assert_frame_equal(result, df)

    def test_remove_duplicates(self):
        """Test removing duplicate rows."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 1, 3],
                "feature2": [4, 5, 4, 6],
            }
        )

        result = drop_duplicates(df)
        expected = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            },
            index=[0, 1, 3],
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_all_duplicates(self):
        """Test when all rows are duplicates."""
        df = pd.DataFrame(
            {
                "feature1": [1, 1, 1],
                "feature2": [4, 4, 4],
            }
        )

        result = drop_duplicates(df)
        expected = pd.DataFrame(
            {
                "feature1": [1],
                "feature2": [4],
            }
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )
