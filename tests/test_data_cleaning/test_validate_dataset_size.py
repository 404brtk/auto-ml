"""Tests for validate_dataset_size function."""

import pandas as pd
import pytest
from auto_ml_pipeline.data_cleaning import validate_dataset_size


class TestValidateDatasetSize:
    """Test validate_dataset_size function."""

    def test_sufficient_rows_and_cols(self):
        """Test with sufficient rows and columns."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        initial_shape = (20, 5)

        # Should not raise any error
        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

    def test_insufficient_rows_raises_error(self):
        """Test that insufficient rows raises ValueError."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [0, 1, 0],
            }
        )
        initial_shape = (100, 2)

        with pytest.raises(ValueError, match="Dataset has only 3 rows after cleaning"):
            validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

    def test_insufficient_cols_raises_error(self):
        """Test that insufficient columns raises ValueError."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        initial_shape = (10, 10)

        with pytest.raises(
            ValueError, match="Dataset has only 1 feature columns after cleaning"
        ):
            validate_dataset_size(df, initial_shape, min_rows=10, min_cols=2)

    def test_exactly_minimum_rows(self):
        """Test with exactly the minimum number of rows."""
        df = pd.DataFrame(
            {
                "feature1": list(range(10)),
                "target": [0, 1] * 5,
            }
        )
        initial_shape = (10, 2)

        # Should not raise error (10 >= 10)
        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

    def test_exactly_minimum_cols(self):
        """Test with exactly the minimum number of columns."""
        df = pd.DataFrame(
            {
                "feature1": list(range(10)),
                "target": [0, 1] * 5,
            }
        )
        initial_shape = (10, 2)

        # Should not raise error (2 >= 2)
        validate_dataset_size(df, initial_shape, min_rows=5, min_cols=2)

    def test_custom_minimums(self):
        """Test with custom minimum thresholds."""
        df = pd.DataFrame(
            {
                "feature1": list(range(50)),
                "feature2": list(range(50)),
                "target": [0, 1] * 25,
            }
        )
        initial_shape = (100, 5)

        # Should not raise with 50 rows, 3 cols
        validate_dataset_size(df, initial_shape, min_rows=50, min_cols=3)

        # Should raise with requirement of 100 rows
        with pytest.raises(ValueError):
            validate_dataset_size(df, initial_shape, min_rows=100, min_cols=1)

    def test_default_minimums(self):
        """Test default minimum values (10 rows, 1 col)."""
        df_pass = pd.DataFrame(
            {
                "feature1": list(range(10)),
                "target": [0, 1] * 5,
            }
        )
        initial_shape = (20, 3)

        # Should pass with defaults (10 rows, 1 col)
        validate_dataset_size(df_pass, initial_shape)

        df_fail = pd.DataFrame(
            {
                "target": list(range(5)),
            }
        )

        # Should fail with only 5 rows (default min is 10)
        with pytest.raises(ValueError):
            validate_dataset_size(df_fail, initial_shape)

    def test_excessive_row_loss_warning(self, caplog):
        """Test warning for excessive row loss."""
        df = pd.DataFrame(
            {
                "feature1": list(range(20)),
                "target": [0, 1] * 10,
            }
        )
        initial_shape = (100, 2)  # Lost 80 rows (80%)

        # Should not raise but should log warning
        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

        # Check that warning was logged (80% > 50%)
        assert any(
            "Lost" in record.message and "80.0%" in record.message
            for record in caplog.records
        )

    def test_excessive_col_loss_warning(self, caplog):
        """Test warning for excessive column loss."""
        df = pd.DataFrame(
            {
                "feature1": list(range(20)),
                "target": [0, 1] * 10,
            }
        )
        initial_shape = (20, 10)  # Lost 8 columns (80%)

        # Should not raise but should log warning
        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

        # Check that warning was logged (80% > 50%)
        assert any(
            "Lost" in record.message and "80.0%" in record.message
            for record in caplog.records
        )

    def test_moderate_row_loss_no_warning(self, caplog):
        """Test no warning for moderate row loss."""
        df = pd.DataFrame(
            {
                "feature1": list(range(60)),
                "target": [0, 1] * 30,
            }
        )
        initial_shape = (100, 2)  # Lost 40 rows (40%)

        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

        # Should not warn (40% < 50%)
        assert not any("Lost" in record.message for record in caplog.records)

    def test_no_data_loss(self):
        """Test with no data loss."""
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "feature2": list(range(100)),
                "target": [0, 1] * 50,
            }
        )
        initial_shape = (100, 3)

        # Should pass without warnings
        validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

    def test_error_message_includes_suggestions(self):
        """Test that error messages include helpful suggestions."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0],
            }
        )
        initial_shape = (100, 5)

        # Check row error message
        with pytest.raises(ValueError) as exc_info:
            validate_dataset_size(df, initial_shape, min_rows=10, min_cols=1)

        assert "Consider relaxing cleaning parameters" in str(exc_info.value)
        assert "min_rows parameter" in str(exc_info.value)

        # Check column error message
        with pytest.raises(ValueError) as exc_info:
            validate_dataset_size(df, initial_shape, min_rows=1, min_cols=5)

        assert "Consider relaxing cleaning parameters" in str(exc_info.value)
        assert "min_cols parameter" in str(exc_info.value)

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        initial_shape = (100, 5)

        with pytest.raises(ValueError):
            validate_dataset_size(df, initial_shape, min_rows=1, min_cols=1)

    def test_zero_initial_shape_no_division_error(self):
        """Test with zero initial shape doesn't cause division errors."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [0, 1, 0],
            }
        )
        initial_shape = (0, 0)

        # Should not raise division by zero error
        validate_dataset_size(df, initial_shape, min_rows=1, min_cols=1)
