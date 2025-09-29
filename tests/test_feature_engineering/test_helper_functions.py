"""Tests for helper functions in feature engineering module."""

from auto_ml_pipeline.feature_engineering import get_imputer, get_scaler


class TestGetImputer:
    """Test get_imputer helper function."""

    def test_imputer_mean_strategy(self):
        """Test get_imputer with mean strategy."""
        imputer = get_imputer("mean")
        assert imputer.strategy == "mean"

    def test_imputer_median_strategy(self):
        """Test get_imputer with median strategy."""
        imputer = get_imputer("median")
        assert imputer.strategy == "median"

    def test_imputer_knn_strategy(self):
        """Test get_imputer with KNN strategy."""
        imputer = get_imputer("knn", knn_neighbors=7)
        assert imputer.n_neighbors == 7

    def test_imputer_knn_default_neighbors(self):
        """Test get_imputer with KNN using default neighbors."""
        imputer = get_imputer("knn")
        assert imputer.n_neighbors == 5

    def test_imputer_invalid_strategy_defaults_to_median(self):
        """Test get_imputer with invalid strategy defaults to median."""
        imputer = get_imputer("invalid")
        assert imputer.strategy == "median"


class TestGetScaler:
    """Test get_scaler helper function."""

    def test_scaler_standard_strategy(self):
        """Test get_scaler with standard strategy."""
        scaler = get_scaler("standard")
        assert hasattr(scaler, "fit")
        assert hasattr(scaler, "transform")

    def test_scaler_none_strategy(self):
        """Test get_scaler with none strategy returns passthrough."""
        scaler = get_scaler("none")
        assert scaler == "passthrough"

    def test_scaler_none_string(self):
        """Test get_scaler with None value."""
        scaler = get_scaler(None)
        assert scaler == "passthrough"

    def test_scaler_minmax_strategy(self):
        """Test get_scaler with minmax strategy."""
        scaler = get_scaler("minmax")
        assert hasattr(scaler, "feature_range")

    def test_scaler_robust_strategy(self):
        """Test get_scaler with robust strategy."""
        scaler = get_scaler("robust")
        assert hasattr(scaler, "quantile_range")

    def test_scaler_invalid_strategy_defaults_to_standard(self):
        """Test get_scaler with invalid strategy defaults to standard."""
        scaler = get_scaler("invalid")
        assert hasattr(scaler, "fit")
        assert hasattr(scaler, "transform")
