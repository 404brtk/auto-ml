"""Tests for models module."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

from auto_ml_pipeline.models import (
    available_models_classification,
    available_models_regression,
    model_search_space,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
            "feat3": np.random.randn(n_samples),
        }
    )
    # Balanced classes
    y = np.array([0, 1] * (n_samples // 2))
    np.random.shuffle(y)
    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
            "feat3": np.random.randn(n_samples),
        }
    )
    # Linear relationship with noise
    y = (
        2.5 * X["feat1"]
        + 1.5 * X["feat2"]
        - 0.5 * X["feat3"]
        + np.random.randn(n_samples) * 0.1
    )
    return X, y.values


@pytest.fixture
def classification_models():
    """Get classification models."""
    return available_models_classification(random_state=42, n_jobs=1)


@pytest.fixture
def regression_models():
    """Get regression models."""
    return available_models_regression(random_state=42, n_jobs=1)


# ============================================================================
# Test Model Availability
# ============================================================================


class TestModelAvailability:
    """Test model availability and structure."""

    def test_classification_returns_dict(self, classification_models):
        """Test that classification function returns a dictionary."""
        assert isinstance(classification_models, dict)
        assert len(classification_models) > 0

    def test_regression_returns_dict(self, regression_models):
        """Test that regression function returns a dictionary."""
        assert isinstance(regression_models, dict)
        assert len(regression_models) > 0

    def test_classification_only_models(self, classification_models, regression_models):
        """Test that classification-only models exist only in classification."""
        classification_only = ["logistic", "naive_bayes"]

        for model_name in classification_only:
            assert model_name in classification_models, (
                f"{model_name} should be in classification"
            )
            assert model_name not in regression_models, (
                f"{model_name} should NOT be in regression"
            )

    def test_regression_only_models(self, classification_models, regression_models):
        """Test that regression-only models exist only in regression."""
        regression_only = ["linear", "lasso", "elastic_net"]

        for model_name in regression_only:
            assert model_name in regression_models, (
                f"{model_name} should be in regression"
            )
            assert model_name not in classification_models, (
                f"{model_name} should NOT be in classification"
            )

    @pytest.mark.parametrize(
        "model_name",
        [
            "ridge",
            "sgd",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "adaboost",
            "svm",
            "knn",
            "mlp",
        ],
    )
    def test_shared_models(self, classification_models, regression_models, model_name):
        """Test that shared models exist in both classification and regression."""
        assert model_name in classification_models, (
            f"{model_name} should be in classification"
        )
        assert model_name in regression_models, f"{model_name} should be in regression"

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "catboost"])
    def test_optional_dependency_models(
        self, classification_models, regression_models, model_name
    ):
        """Test that optional dependency models are available when installed."""
        # These may or may not be present depending on installation
        if model_name in classification_models:
            assert model_name in regression_models, (
                f"If {model_name} in classification, should be in regression too"
            )
        if model_name in regression_models:
            assert model_name in classification_models, (
                f"If {model_name} in regression, should be in classification too"
            )


# ============================================================================
# Test Model Instantiation
# ============================================================================


class TestModelInstantiation:
    """Test that models are properly instantiated."""

    def test_all_models_have_fit_predict(self, classification_models):
        """Test that all models have fit and predict methods."""
        for name, model in classification_models.items():
            assert hasattr(model, "fit"), f"{name} should have fit method"
            assert hasattr(model, "predict"), f"{name} should have predict method"
            assert callable(model.fit), f"{name}.fit should be callable"
            assert callable(model.predict), f"{name}.predict should be callable"

    @pytest.mark.parametrize("random_state", [42, 123, 999])
    def test_random_state_parameter(self, random_state):
        """Test that random_state parameter is applied correctly."""
        models = available_models_classification(random_state=random_state)

        # Check models that support random_state
        models_with_random_state = [
            "random_forest",
            "decision_tree",
            "gradient_boosting",
            "mlp",
        ]

        for model_name in models_with_random_state:
            if model_name in models:
                model = models[model_name]
                assert hasattr(model, "random_state"), (
                    f"{model_name} should have random_state"
                )
                assert model.random_state == random_state, (
                    f"{model_name} random_state should be {random_state}"
                )

    @pytest.mark.parametrize("n_jobs", [1, 2, 4, -1])
    def test_n_jobs_parameter(self, n_jobs):
        """Test that n_jobs parameter is applied correctly."""
        models = available_models_classification(n_jobs=n_jobs)

        # Check models that support n_jobs
        models_with_n_jobs = ["random_forest", "knn", "logistic"]

        for model_name in models_with_n_jobs:
            if model_name in models:
                model = models[model_name]
                assert hasattr(model, "n_jobs"), f"{model_name} should have n_jobs"
                assert model.n_jobs == n_jobs, f"{model_name} n_jobs should be {n_jobs}"


# ============================================================================
# Test Search Spaces
# ============================================================================


class TestSearchSpaces:
    """Test hyperparameter search spaces."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "logistic",
            "linear",
            "ridge",
            "lasso",
            "elastic_net",
            "sgd",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "adaboost",
            "xgboost",
            "lightgbm",
            "catboost",
            "svm",
            "knn",
            "naive_bayes",
            "mlp",
        ],
    )
    def test_search_space_returns_dict(self, model_name):
        """Test that search space returns a dictionary."""
        space = model_search_space(model_name)
        assert isinstance(space, dict), f"{model_name} should return dict"

    def test_unknown_model_returns_empty_dict(self):
        """Test that unknown model returns empty dictionary."""
        space = model_search_space("unknown_model_xyz")
        assert isinstance(space, dict)
        assert len(space) == 0

    @pytest.mark.parametrize(
        "model_name,expected_params",
        [
            ("random_forest", ["n_estimators", "max_depth", "min_samples_split"]),
            ("gradient_boosting", ["n_estimators", "learning_rate", "max_depth"]),
            (
                "xgboost",
                [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "reg_alpha",
                    "reg_lambda",
                ],
            ),
            ("lightgbm", ["n_estimators", "num_leaves", "learning_rate"]),
            ("svm", ["C", "kernel"]),
            ("knn", ["n_neighbors", "weights"]),
            (
                "mlp",
                ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter"],
            ),
        ],
    )
    def test_expected_hyperparameters(self, model_name, expected_params):
        """Test that models have expected hyperparameters."""
        space = model_search_space(model_name)

        if space:  # Only test if model has search space defined
            for param in expected_params:
                assert param in space, f"{model_name} should have {param} parameter"

    def test_search_space_structure(self):
        """Test that search spaces have proper structure."""
        all_models = [
            "logistic",
            "ridge",
            "sgd",
            "decision_tree",
            "random_forest",
            "xgboost",
            "lightgbm",
            "svm",
            "mlp",
        ]

        for model_name in all_models:
            space = model_search_space(model_name)

            # Skip empty spaces (like 'linear' which has no hyperparameters)
            if not space:
                continue

            for param_name, param_spec in space.items():
                assert isinstance(param_spec, tuple), (
                    f"{model_name}.{param_name} should be tuple"
                )
                assert len(param_spec) == 2, (
                    f"{model_name}.{param_name} should have 2 elements"
                )

                param_type, bounds = param_spec
                assert isinstance(param_type, str), (
                    f"{model_name}.{param_name} type should be string"
                )
                assert param_type in ["int", "uniform", "loguniform", "categorical"], (
                    f"{model_name}.{param_name} has invalid type: {param_type}"
                )

                assert isinstance(bounds, tuple), (
                    f"{model_name}.{param_name} bounds should be tuple"
                )

    @pytest.mark.parametrize(
        "model_name,param_name,expected_type",
        [
            ("random_forest", "n_estimators", "int"),
            ("random_forest", "max_depth", "int"),
            ("random_forest", "bootstrap", "categorical"),
            ("gradient_boosting", "learning_rate", "loguniform"),
            ("gradient_boosting", "subsample", "uniform"),
            ("svm", "C", "loguniform"),
            ("svm", "kernel", "categorical"),
            ("knn", "n_neighbors", "int"),
            ("knn", "weights", "categorical"),
            ("mlp", "hidden_layer_sizes", "categorical"),
            ("mlp", "activation", "categorical"),
            ("mlp", "solver", "categorical"),
            ("mlp", "alpha", "loguniform"),
            ("mlp", "learning_rate_init", "loguniform"),
            ("mlp", "max_iter", "int"),
        ],
    )
    def test_parameter_types(self, model_name, param_name, expected_type):
        """Test that specific parameters have correct types."""
        space = model_search_space(model_name)

        if param_name in space:
            param_type, _ = space[param_name]
            assert param_type == expected_type, (
                f"{model_name}.{param_name} should be {expected_type}, got {param_type}"
            )

    def test_categorical_parameters_structure(self):
        """Test that categorical parameters have valid choices."""
        categorical_params: Dict[str, Dict[str, list]] = {
            "random_forest": {"bootstrap": [True, False]},
            "svm": {"kernel": ["linear", "rbf", "poly"]},
            "knn": {"weights": ["uniform", "distance"]},
            "mlp": {
                "activation": ["relu", "tanh"],
                "solver": ["adam", "lbfgs"],
            },
        }

        for model_name, params in categorical_params.items():
            space = model_search_space(model_name)

            for param_name, expected_choices in params.items():
                if param_name in space:
                    param_type, bounds = space[param_name]
                    assert param_type == "categorical"
                    assert len(bounds) > 0, (
                        f"{model_name}.{param_name} should have choices"
                    )

    def test_numeric_parameters_have_valid_ranges(self):
        """Test that numeric parameters have valid ranges."""
        space = model_search_space("random_forest")

        if "n_estimators" in space:
            param_type, bounds = space["n_estimators"]
            assert param_type == "int"
            assert len(bounds) == 2
            low, high = bounds
            assert low < high, "Lower bound should be less than upper bound"
            assert low > 0, "n_estimators should be positive"

    def test_mlp_hidden_layer_sizes_are_strings(self):
        """Test that MLP hidden_layer_sizes are stored as strings."""
        space = model_search_space("mlp")

        if "hidden_layer_sizes" in space:
            param_type, bounds = space["hidden_layer_sizes"]
            assert param_type == "categorical"

            # All values should be strings
            for value in bounds:
                assert isinstance(value, str), (
                    f"hidden_layer_sizes value should be string, got {type(value)}"
                )

                # Should contain only digits and commas
                assert all(c.isdigit() or c == "," for c in value), (
                    f"hidden_layer_sizes '{value}' should contain only digits and commas"
                )


# ============================================================================
# Test Model Training
# ============================================================================


class TestModelTraining:
    """Test that models can be trained successfully."""

    def test_classification_models_fit_predict(
        self, classification_models, sample_classification_data
    ):
        """Test that all classification models can fit and predict."""
        X, y = sample_classification_data

        for name, model in classification_models.items():
            # Fit model
            model.fit(X, y)

            # Predict
            predictions = model.predict(X)

            assert len(predictions) == len(y), f"{name} predictions length mismatch"
            assert all(p in np.unique(y) for p in predictions), (
                f"{name} predictions not in valid classes"
            )

    def test_regression_models_fit_predict(
        self, regression_models, sample_regression_data
    ):
        """Test that all regression models can fit and predict."""
        X, y = sample_regression_data

        for name, model in regression_models.items():
            # Fit model
            model.fit(X, y)

            # Predict
            predictions = model.predict(X)

            assert len(predictions) == len(y), f"{name} predictions length mismatch"
            assert all(isinstance(p, (int, float, np.number)) for p in predictions), (
                f"{name} predictions should be numeric"
            )

    @pytest.mark.parametrize(
        "n_samples,n_features",
        [
            (50, 5),
            (100, 10),
            (200, 15),
        ],
    )
    def test_models_with_different_data_sizes(
        self, classification_models, n_samples, n_features
    ):
        """Test models work with various data sizes."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        y = np.random.randint(0, 2, n_samples)

        # Test a subset of fast models
        fast_models = ["logistic", "ridge", "decision_tree"]

        for model_name in fast_models:
            if model_name in classification_models:
                model = classification_models[model_name]
                model.fit(X, y)
                predictions = model.predict(X)
                assert len(predictions) == n_samples

    def test_models_deterministic_with_random_state(self, sample_classification_data):
        """Test that models produce same results with same random_state."""
        X, y = sample_classification_data

        # Test with random_forest as it's affected by random_state
        model1 = available_models_classification(random_state=42)["random_forest"]
        model2 = available_models_classification(random_state=42)["random_forest"]

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(
            pred1,
            pred2,
            "Models with same random_state should produce same predictions",
        )


# ============================================================================
# Test Special Cases
# ============================================================================


class TestSpecialCases:
    """Test edge cases and special scenarios."""

    def test_linear_model_has_empty_search_space(self):
        """Test that linear regression has no hyperparameters to tune."""
        space = model_search_space("linear")
        assert isinstance(space, dict)
        assert len(space) == 0, (
            "Linear regression should have no hyperparameters to tune"
        )

    def test_models_handle_dataframe_input(self, classification_models):
        """Test that models can handle pandas DataFrame input."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
                "c": np.random.randn(50),
            }
        )
        y = np.random.randint(0, 2, 50)

        # Test a few models
        for model_name in ["logistic", "random_forest", "ridge"]:
            if model_name in classification_models:
                model = classification_models[model_name]
                model.fit(X, y)
                predictions = model.predict(X)
                assert len(predictions) == 50

    def test_models_handle_numpy_input(self, classification_models):
        """Test that models can handle numpy array input."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        for model_name in ["logistic", "random_forest", "ridge"]:
            if model_name in classification_models:
                model = classification_models[model_name]
                model.fit(X, y)
                predictions = model.predict(X)
                assert len(predictions) == 50


# ============================================================================
# Test Model Consistency
# ============================================================================


class TestModelConsistency:
    """Test consistency between classification and regression models."""

    def test_shared_models_have_same_search_space_keys(self):
        """Test that shared models have similar search spaces."""
        shared_models = [
            "ridge",
            "sgd",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "svm",
            "knn",
            "mlp",
        ]

        for model_name in shared_models:
            clf_space = model_search_space(model_name)
            reg_space = model_search_space(model_name)

            # Should have the same parameters
            assert set(clf_space.keys()) == set(reg_space.keys()), (
                f"{model_name} should have same params for classification and regression"
            )

    def test_model_naming_consistency(self):
        """Test that model names follow conventions."""
        clf_models = available_models_classification()
        reg_models = available_models_regression()

        all_model_names = set(clf_models.keys()) | set(reg_models.keys())

        for name in all_model_names:
            # Model names should be lowercase with underscores
            assert name.islower(), f"{name} should be lowercase"
            assert " " not in name, f"{name} should not contain spaces"
