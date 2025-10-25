"""Integration tests for full auto-ML pipeline workflow."""

import numpy as np
import pandas as pd
import pytest
import warnings

from auto_ml_pipeline.config import (
    CleaningConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    EncodingConfig,
    ScalingConfig,
    ImputationConfig,
    TaskType,
    PipelineConfig,
    OptimizationConfig,
    ModelsConfig,
    IOConfig,
    ReportConfig,
)
from auto_ml_pipeline.data_cleaning import clean_data
from auto_ml_pipeline.feature_engineering import build_preprocessor
from auto_ml_pipeline.feature_selection import build_selector
from auto_ml_pipeline.trainer import train
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def suppress_warnings():
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


@pytest.fixture(scope="session")
def random_seed():
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def classification_data():
    np.random.seed(42)
    n_samples = 300
    return pd.DataFrame(
        {
            "numeric1": np.random.randn(n_samples),
            "numeric2": np.random.randn(n_samples) * 100 + 50,
            "category_low": np.random.choice(["A", "B", "C"], n_samples),
            "category_high": [f"cat_{i % 30}" for i in range(n_samples)],
            "date": pd.date_range("2023-01-01", periods=n_samples),
            "target": np.random.randint(0, 3, n_samples),
        }
    )


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
            "feat3": np.random.choice(["low", "medium", "high"], n_samples),
            "target": np.random.randn(n_samples) * 10 + 50,
        }
    )


@pytest.fixture
def messy_data():
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "user_id": range(n_samples),
            "numeric1": np.random.randn(n_samples),
            "numeric2": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "date": pd.date_range("2023-01-01", periods=n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )
    df.loc[5, "numeric1"] = np.nan
    df.loc[10, "category"] = np.nan
    df.loc[15, "target"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


@pytest.fixture
def text_data():
    np.random.seed(42)
    texts = [
        "This is a long sample text for classification task that needs to be really quite long to exceed the text length threshold for proper processing",
        "Another text sample with different words and content and more words to make it sufficiently long for text feature extraction",
        "Machine learning and natural language processing example with additional descriptive text to meet the minimum length requirements",
        "Final text entry for testing purposes with extra content to ensure it is categorized as text and not as categorical variable",
    ]
    return pd.DataFrame(
        {
            "text": texts * 25,
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def cli_app():
    from auto_ml_pipeline.cli import app

    return app


def check_no_nan(array_like):
    if hasattr(array_like, "values"):
        return not np.isnan(array_like.values).any()
    return not np.isnan(array_like).any()


class TestDataCleaningIntegration:
    def test_clean_messy_data_end_to_end(self, messy_data):
        cfg = CleaningConfig(
            drop_duplicates=True, max_missing_row_ratio=0.3, remove_id_columns=True
        )
        result, target, _ = clean_data(messy_data, "target", cfg)

        assert len(result) < len(messy_data)
        assert not result[target].isna().any()
        assert not result.duplicated().any()
        assert len(result) > 50

    def test_clean_preserves_data_types(self, classification_data):
        cfg = CleaningConfig(remove_id_columns=False)
        result, target, _ = clean_data(classification_data, "target", cfg)

        assert pd.api.types.is_numeric_dtype(result["numeric1"])
        assert pd.api.types.is_numeric_dtype(result["numeric2"])
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_clean_with_datetime_and_numeric_conversion(self):
        df = pd.DataFrame(
            {
                "date_str": ["2023-01-01", "2023-02-15", "2023-03-30"],
                "numeric_str": ["1", "2", "3"],
                "target": [0, 1, 0],
            }
        )
        cfg = CleaningConfig(remove_id_columns=False)
        result, target, _ = clean_data(df, "target", cfg)

        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])
        assert pd.api.types.is_numeric_dtype(result["numeric_str"])


class TestFeatureEngineeringIntegration:
    def test_mixed_feature_types_preprocessing(self, classification_data):
        cfg = FeatureEngineeringConfig(extract_datetime=True)
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]

        preprocessor, col_types = build_preprocessor(X, cfg)
        X_transformed = preprocessor.fit_transform(X, y)

        total_features = (
            len(col_types.numeric)
            + len(col_types.categorical_low)
            + len(col_types.categorical_high)
            + len(col_types.datetime)
        )
        assert total_features > 0
        assert check_no_nan(X_transformed)
        assert X_transformed.shape[0] == len(X)

    def test_high_cardinality_encoding(self):
        np.random.seed(42)
        df = pd.DataFrame({"high_card": [f"cat_{i}" for i in range(100)]})
        y = np.random.randn(100)
        cfg = FeatureEngineeringConfig(
            encoding=EncodingConfig(high_cardinality_threshold=50)
        )

        preprocessor, col_types = build_preprocessor(df, cfg)
        assert len(col_types.categorical_high) == 1

        X_transformed = preprocessor.fit_transform(df, y)
        assert X_transformed.shape[0] == 100
        assert check_no_nan(X_transformed)

    @pytest.mark.parametrize("strategy", ["standard", "minmax", "robust"])
    def test_scaling_strategies(self, strategy):
        df = pd.DataFrame({"feat": [1, 100, 1000, 10000]})
        cfg = FeatureEngineeringConfig(scaling=ScalingConfig(strategy=strategy))

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.max() < df["feat"].max()
        assert check_no_nan(X_transformed)

    @pytest.mark.parametrize("strategy", ["mean", "median"])
    def test_imputation_strategies(self, strategy):
        df = pd.DataFrame({"feat": [1, 2, np.nan, 4, 5]})
        cfg = FeatureEngineeringConfig(
            imputation=ImputationConfig(strategy_num=strategy)
        )

        preprocessor, _ = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert check_no_nan(X_transformed)

    def test_text_feature_extraction(self, text_data):
        cfg = FeatureEngineeringConfig(
            handle_text=True, max_features_text=100, text_length_threshold=50
        )
        X = text_data.drop("target", axis=1)

        preprocessor, col_types = build_preprocessor(X, cfg)
        X_transformed = preprocessor.fit_transform(X)

        assert len(col_types.text) == 1
        assert X_transformed.shape[1] > 0

    def test_datetime_feature_extraction(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "time": ["14:30:00"] * 100,
            }
        )
        cfg = FeatureEngineeringConfig(extract_datetime=True, extract_time=True)

        preprocessor, col_types = build_preprocessor(df, cfg)
        X_transformed = preprocessor.fit_transform(df)

        assert len(col_types.datetime) > 0
        assert X_transformed.shape[1] > 2


class TestFullPipelineIntegration:
    def test_classification_pipeline_end_to_end(self, messy_data):
        clean_cfg = CleaningConfig(
            drop_duplicates=True, max_missing_row_ratio=0.5, remove_id_columns=True
        )
        df_clean, target, _ = clean_data(messy_data, "target", clean_cfg)
        assert len(df_clean) > 10

        X = df_clean.drop(target, axis=1)
        y = df_clean[target]

        fe_cfg = FeatureEngineeringConfig(extract_datetime=True)
        preprocessor, col_types = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X, y)

        fs_cfg = FeatureSelectionConfig(variance_threshold=0.01)
        selector = build_selector(fs_cfg, TaskType.classification)

        X_final = (
            selector.fit_transform(X_transformed, y) if selector else X_transformed
        )

        assert X_final.shape[0] == len(y)
        assert check_no_nan(X_final)
        assert not np.isinf(
            X_final if not hasattr(X_final, "values") else X_final.values
        ).any()

    def test_regression_pipeline_end_to_end(self, regression_data):
        clean_cfg = CleaningConfig(remove_id_columns=False)
        df_clean, target, _ = clean_data(regression_data, "target", clean_cfg)

        X = df_clean.drop(target, axis=1)
        y = df_clean[target]

        fe_cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X, y)

        fs_cfg = FeatureSelectionConfig(correlation_threshold=0.95)
        selector = build_selector(fs_cfg, TaskType.regression)

        X_final = (
            selector.fit_transform(X_transformed, y) if selector else X_transformed
        )

        assert X_final.shape[0] == len(y)
        assert check_no_nan(X_final)

    def test_pipeline_with_missing_values(self):
        df = pd.DataFrame(
            {
                "feat1": [1, 2, np.nan, 4, 5] * 20,
                "feat2": [10, np.nan, 30, 40, 50] * 20,
                "category": ["A", "B", "A", np.nan, "C"] * 20,
                "target": [0, 1, 0, 1, 0] * 20,
            }
        )
        clean_cfg = CleaningConfig(drop_duplicates=False, remove_id_columns=False)
        df_clean, target, _ = clean_data(df, "target", clean_cfg)

        X = df_clean.drop(target, axis=1)
        y = df_clean[target]

        fe_cfg = FeatureEngineeringConfig()
        preprocessor, _ = build_preprocessor(X, fe_cfg)
        X_transformed = preprocessor.fit_transform(X, y)

        assert check_no_nan(X_transformed)


class TestTrainingIntegration:
    def test_classification_training_workflow(self, classification_data, tmp_path):
        cfg = PipelineConfig(
            task=TaskType.classification,
            optimization=OptimizationConfig(enabled=True, n_trials=2),
            models=ModelsConfig(models=["logistic", "decision_tree"]),
            io=IOConfig(output_dir=tmp_path),
            reporting=ReportConfig(enabled=True),
        )
        result = train(classification_data, "target", cfg)

        assert result is not None
        assert result.best_model_name in ["logistic", "decision_tree"]
        assert result.best_score > -np.inf
        assert result.best_estimator is not None
        assert "f1_macro" in result.metrics

        run_dir = tmp_path / result.run_dir.name
        assert (run_dir / "training_report.html").exists()
        assert (run_dir / "eval_model.joblib").exists()
        assert (run_dir / "results.json").exists()

    def test_regression_training_workflow(self, regression_data, tmp_path):
        cfg = PipelineConfig(
            task=TaskType.regression,
            optimization=OptimizationConfig(enabled=True, n_trials=2),
            models=ModelsConfig(models=["linear", "ridge"]),
            io=IOConfig(output_dir=tmp_path),
            reporting=ReportConfig(enabled=False),
        )
        result = train(regression_data, "target", cfg)

        assert result is not None
        assert result.best_model_name in ["linear", "ridge"]
        assert result.best_estimator is not None
        assert "rmse" in result.metrics

        run_dir = tmp_path / result.run_dir.name
        assert not (run_dir / "training_report.html").exists()
        assert (run_dir / "eval_model.joblib").exists()


class TestCLIIntegration:
    @pytest.mark.parametrize(
        "task_type,n_samples",
        [
            ("classification", 100),
            ("regression", 100),
        ],
    )
    def test_cli_run_command(self, cli_runner, cli_app, tmp_path, task_type, n_samples):
        np.random.seed(42)
        if task_type == "classification":
            df = pd.DataFrame(
                {
                    "feature1": np.random.rand(n_samples),
                    "feature2": np.random.rand(n_samples),
                    "target": np.random.randint(0, 2, n_samples),
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "feature1": np.random.rand(n_samples),
                    "feature2": np.random.rand(n_samples),
                    "target": np.random.rand(n_samples) * 100,
                }
            )

        dataset_path = tmp_path / f"dummy_{task_type}.csv"
        df.to_csv(dataset_path, index=False)

        output_dir = tmp_path / f"cli_output_{task_type}"
        output_dir.mkdir()

        result = cli_runner.invoke(
            cli_app,
            [
                "run",
                "--dataset",
                str(dataset_path),
                "--target",
                "target",
                "--output",
                str(output_dir),
                "--config",
                "configs/quickstart.yaml",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"
        assert "Training completed!" in result.stdout

        run_dirs = list(output_dir.glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "eval_model.joblib").exists()
        assert (run_dir / "results.json").exists()

    def test_cli_export_command(self, cli_runner, cli_app, tmp_path):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "target": np.random.randint(0, 2, 100),
            }
        )
        dataset_path = tmp_path / "train_data.csv"
        df.to_csv(dataset_path, index=False)

        output_dir = tmp_path / "cli_export_source"
        output_dir.mkdir()

        run_result = cli_runner.invoke(
            cli_app,
            [
                "run",
                "--dataset",
                str(dataset_path),
                "--target",
                "target",
                "--output",
                str(output_dir),
                "--config",
                "configs/quickstart.yaml",
            ],
        )
        assert run_result.exit_code == 0, f"Training failed: {run_result.stderr}"

        source_run_dir = list(output_dir.glob("run_*"))[0]
        source_model_path = source_run_dir / "eval_model.joblib"
        export_destination = tmp_path / "exported_model.joblib"

        export_result = cli_runner.invoke(
            cli_app, ["export", str(source_model_path), "--to", str(export_destination)]
        )

        assert export_result.exit_code == 0, f"Export failed: {export_result.stderr}"
        assert "Model exported to" in export_result.stdout
        assert export_destination.exists()
