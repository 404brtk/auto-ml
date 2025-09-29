"""Tests for models module."""

from auto_ml_pipeline.models import (
    available_models_classification,
    available_models_regression,
    model_search_space,
)


class TestAvailableModelsClassification:
    """Test available_models_classification function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = available_models_classification()
        assert isinstance(models, dict)

    def test_contains_basic_models(self):
        """Test that basic models are always available."""
        models = available_models_classification()

        # These should always be available (sklearn models)
        assert "random_forest" in models
        assert "gradient_boosting" in models

    def test_models_are_instantiated(self):
        """Test that returned values are model instances."""
        models = available_models_classification()

        for name, model in models.items():
            # Should have fit and predict methods
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_random_state_parameter(self):
        """Test that random_state parameter is applied."""
        models = available_models_classification(random_state=123)

        # Check random_forest has the specified random_state
        rf = models["random_forest"]
        assert hasattr(rf, "random_state")
        assert rf.random_state == 123

    def test_n_jobs_parameter(self):
        """Test that n_jobs parameter is applied."""
        models = available_models_classification(n_jobs=4)

        rf = models["random_forest"]
        assert hasattr(rf, "n_jobs")
        assert rf.n_jobs == 4

    def test_xgboost_availability(self):
        """Test XGBoost availability (optional dependency)."""
        models = available_models_classification()

        # XGBoost may or may not be available depending on installation
        if "xgboost" in models:
            xgb = models["xgboost"]
            assert hasattr(xgb, "fit")
            assert hasattr(xgb, "predict")

    def test_lightgbm_availability(self):
        """Test LightGBM availability (optional dependency)."""
        models = available_models_classification()

        # LightGBM may or may not be available depending on installation
        if "lightgbm" in models:
            lgbm = models["lightgbm"]
            assert hasattr(lgbm, "fit")
            assert hasattr(lgbm, "predict")

    def test_minimum_models_available(self):
        """Test that at least sklearn models are available."""
        models = available_models_classification()

        # Should have at least 2 models (sklearn ones)
        assert len(models) >= 2


class TestAvailableModelsRegression:
    """Test available_models_regression function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = available_models_regression()
        assert isinstance(models, dict)

    def test_contains_basic_models(self):
        """Test that basic models are always available."""
        models = available_models_regression()

        # These should always be available (sklearn models)
        assert "random_forest" in models
        assert "gradient_boosting" in models

    def test_models_are_instantiated(self):
        """Test that returned values are model instances."""
        models = available_models_regression()

        for name, model in models.items():
            # Should have fit and predict methods
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_random_state_parameter(self):
        """Test that random_state parameter is applied."""
        models = available_models_regression(random_state=456)

        rf = models["random_forest"]
        assert hasattr(rf, "random_state")
        assert rf.random_state == 456

    def test_n_jobs_parameter(self):
        """Test that n_jobs parameter is applied."""
        models = available_models_regression(n_jobs=8)

        rf = models["random_forest"]
        assert hasattr(rf, "n_jobs")
        assert rf.n_jobs == 8

    def test_xgboost_availability(self):
        """Test XGBoost regressor availability."""
        models = available_models_regression()

        if "xgboost" in models:
            xgb = models["xgboost"]
            assert hasattr(xgb, "fit")
            assert hasattr(xgb, "predict")

    def test_lightgbm_availability(self):
        """Test LightGBM regressor availability."""
        models = available_models_regression()

        if "lightgbm" in models:
            lgbm = models["lightgbm"]
            assert hasattr(lgbm, "fit")
            assert hasattr(lgbm, "predict")

    def test_minimum_models_available(self):
        """Test that at least sklearn models are available."""
        models = available_models_regression()

        # Should have at least 2 models (sklearn ones)
        assert len(models) >= 2


class TestModelSearchSpace:
    """Test model_search_space function."""

    def test_is_callable(self):
        """Test that model_search_space is a callable function."""
        assert callable(model_search_space)

    def test_returns_dict_for_valid_model(self):
        """Test that function returns a dictionary for valid models."""
        rf_space = model_search_space("random_forest")
        assert isinstance(rf_space, dict)

    def test_random_forest_hyperparameters(self):
        """Test random_forest has expected hyperparameters."""
        rf_space = model_search_space("random_forest")

        # Should have typical RF hyperparameters
        assert isinstance(rf_space, dict)
        assert len(rf_space) > 0

        # Check for some expected parameters
        assert "n_estimators" in rf_space
        assert "max_depth" in rf_space

    def test_gradient_boosting_hyperparameters(self):
        """Test gradient_boosting has expected hyperparameters."""
        gb_space = model_search_space("gradient_boosting")

        assert isinstance(gb_space, dict)
        assert len(gb_space) > 0

        # Check for some expected parameters
        assert "n_estimators" in gb_space
        assert "learning_rate" in gb_space

    def test_xgboost_hyperparameters(self):
        """Test XGBoost hyperparameters."""
        xgb_space = model_search_space("xgboost")

        assert isinstance(xgb_space, dict)
        # May be empty if XGBoost not in search spaces, but should be dict
        if xgb_space:
            assert "n_estimators" in xgb_space

    def test_lightgbm_hyperparameters(self):
        """Test LightGBM hyperparameters."""
        lgbm_space = model_search_space("lightgbm")

        assert isinstance(lgbm_space, dict)
        # May be empty if LightGBM not in search spaces, but should be dict
        if lgbm_space:
            assert "n_estimators" in lgbm_space

    def test_search_space_structure(self):
        """Test that search spaces have proper structure."""
        for model_name in ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]:
            space = model_search_space(model_name)
            assert isinstance(space, dict)

            # Check structure for non-empty spaces
            if space:
                for param_name, param_spec in space.items():
                    assert isinstance(param_spec, tuple)
                    assert len(param_spec) == 2

                    param_type, bounds = param_spec
                    assert isinstance(param_type, str)
                    assert isinstance(bounds, tuple)
                    assert len(bounds) == 2

    def test_unknown_model_returns_empty_dict(self):
        """Test that unknown model names return empty dict."""
        space = model_search_space("unknown_model_xyz")
        assert isinstance(space, dict)
        assert len(space) == 0


class TestModelsIntegration:
    """Integration tests for models."""

    def test_classification_models_can_fit(self):
        """Test that classification models can be fitted."""
        import numpy as np
        import pandas as pd

        models = available_models_classification()
        # Use DataFrame with feature names to avoid LGBM warnings
        X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=["feat1", "feat2"])
        y = np.array([0, 1, 0, 1])

        for name, model in models.items():
            # Should be able to fit without errors
            model.fit(X, y)
            predictions = model.predict(X)

            assert len(predictions) == len(y)
            assert all(p in [0, 1] for p in predictions)

    def test_regression_models_can_fit(self):
        """Test that regression models can be fitted."""
        import numpy as np
        import pandas as pd

        models = available_models_regression()
        # Use DataFrame with feature names to avoid LGBM warnings
        X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=["feat1", "feat2"])
        y = np.array([1.5, 2.5, 3.5, 4.5])

        for name, model in models.items():
            # Should be able to fit without errors
            model.fit(X, y)
            predictions = model.predict(X)

            assert len(predictions) == len(y)
            assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_models_work_with_different_data_sizes(self):
        """Test models work with various data sizes."""
        import numpy as np
        import pandas as pd

        models = available_models_classification()

        # Small dataset with feature names to avoid LGBM warnings
        X_small = pd.DataFrame([[1, 2], [3, 4]], columns=["feat1", "feat2"])
        y_small = np.array([0, 1])

        for name, model in models.items():
            model.fit(X_small, y_small)
            pred = model.predict(X_small)
            assert len(pred) == 2

    def test_consistent_model_names(self):
        """Test that classification and regression have same model names."""
        clf_models = set(available_models_classification().keys())
        reg_models = set(available_models_regression().keys())

        # Should have the same model names available
        assert clf_models == reg_models
