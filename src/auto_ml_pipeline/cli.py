from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from auto_ml_pipeline.config import PipelineConfig, load_config, default_config
from auto_ml_pipeline.trainer import train
from auto_ml_pipeline.io_utils import load_model, save_model
from auto_ml_pipeline.logging_utils import setup_logging
from auto_ml_pipeline.api import start_server

app = typer.Typer(add_completion=True)


def _load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".feather"}:
        return pd.read_feather(path)
    raise typer.BadParameter(f"Unsupported dataset format: {suffix}")


@app.command()
def run(
    dataset: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to dataset (csv/parquet/feather)"
    ),
    target: str = typer.Option(..., help="Target column name"),
    config: Optional[Path] = typer.Option(None, help="Path to config (toml/yaml/json)"),
    output: Optional[Path] = typer.Option(None, help="Output directory for artifacts"),
):
    """Run full training pipeline with optional config."""
    setup_logging()
    df = _load_data(dataset)
    cfg: PipelineConfig = load_config(config) if config else default_config()
    cfg.io.dataset_path = dataset
    cfg.io.target = target
    if output:
        cfg.io.output_dir = output
    res = train(df, target, cfg)
    typer.echo(f"Training completed! Results saved to: {res.run_dir}")


@app.command()
def export(
    model_path: Path = typer.Argument(..., exists=True),
    to: Path = typer.Option(..., help="Destination path for exported model (.joblib)"),
):
    """Re-save a trained model to a new location (e.g., for deployment)."""
    setup_logging()
    model = load_model(model_path)
    save_model(model, to)
    typer.echo(f"Model exported to {to}")


@app.command()
def deploy(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Path to a completed run directory."
    ),
    config: Optional[Path] = typer.Option(None, help="Path to config (toml/yaml/json)"),
    host: Optional[str] = typer.Option(None, help="Host to bind the server to."),
    port: Optional[int] = typer.Option(None, help="Port to run the server on."),
):
    """Launch a server to deploy and serve a trained model."""
    setup_logging()
    cfg: PipelineConfig = load_config(config) if config else default_config()

    server_host = host if host is not None else cfg.api.host
    server_port = port if port is not None else cfg.api.port

    start_server(run_dir, server_host, server_port)


if __name__ == "__main__":
    app()
