from auto_ml_pipeline.feature_engineering import get_imputer, get_scaler


def test_get_imputer_mean():
    """Test get_imputer with mean strategy."""
    imputer = get_imputer("mean")
    assert imputer.strategy == "mean"


def test_get_imputer_median():
    """Test get_imputer with median strategy."""
    imputer = get_imputer("median")
    assert imputer.strategy == "median"


def test_get_imputer_knn():
    """Test get_imputer with KNN strategy."""
    imputer = get_imputer("knn", knn_neighbors=7)
    assert imputer.n_neighbors == 7


def test_get_imputer_invalid():
    """Test get_imputer with invalid strategy defaults to median."""
    imputer = get_imputer("invalid")
    assert imputer.strategy == "median"


def test_get_scaler_standard():
    """Test get_scaler with standard strategy."""
    scaler = get_scaler("standard")
    assert hasattr(scaler, "fit")  # Should be StandardScaler instance


def test_get_scaler_none():
    """Test get_scaler with none strategy."""
    scaler = get_scaler("none")
    assert scaler == "passthrough"


def test_get_scaler_minmax():
    """Test get_scaler with minmax strategy."""
    scaler = get_scaler("minmax")
    assert hasattr(scaler, "feature_range")  # MinMaxScaler has feature_range


def test_get_scaler_robust():
    """Test get_scaler with robust strategy."""
    scaler = get_scaler("robust")
    assert hasattr(scaler, "quantile_range")  # RobustScaler has quantile_range


def test_get_scaler_invalid():
    """Test get_scaler with invalid strategy defaults to standard."""
    scaler = get_scaler("invalid")
    assert hasattr(scaler, "fit")  # Should be StandardScaler instance
