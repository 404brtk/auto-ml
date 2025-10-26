from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from auto_ml_pipeline.config import PipelineConfig, load_config, default_config
from auto_ml_pipeline.trainer import train
from auto_ml_pipeline.logging_utils import setup_logging
from auto_ml_pipeline.api import start_server

app = typer.Typer(add_completion=True, no_args_is_help=True)


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


@app.command(no_args_is_help=True)
def run(
    dataset: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to dataset (csv/parquet/feather)"
    ),
    target: str = typer.Option(..., help="Target column name"),
    config: Optional[Path] = typer.Option(None, help="Path to config (toml/yaml/json)"),
    output: Optional[Path] = typer.Option(None, help="Output directory for artifacts"),
):
    """Run full training pipeline."""
    setup_logging()

    try:
        df = _load_data(dataset)
        cfg: PipelineConfig = load_config(config) if config else default_config()
        cfg.io.dataset_path = dataset
        cfg.io.target = target
        if output:
            cfg.io.output_dir = output
        res = train(df, target, cfg)
        typer.echo(f"✅ Training completed! Results saved to: {res.run_dir}")
        typer.echo(f"💡 Deploy with: auto-ml deploy {res.run_dir}")
    except Exception as e:
        typer.secho(f"❌ Error during training: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(no_args_is_help=True)
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

    try:
        cfg: PipelineConfig = load_config(config) if config else default_config()

        server_host = host if host is not None else cfg.api.host
        server_port = port if port is not None else cfg.api.port

        typer.echo(f"🚀 Starting server at {server_host}:{server_port}")
        start_server(run_dir, server_host, server_port)
    except Exception as e:
        typer.secho(f"❌ Error starting server: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
