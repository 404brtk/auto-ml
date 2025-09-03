import json
from datetime import datetime
from pathlib import Path
from typing import Any
import joblib


def make_run_dir(base_dir: str | Path) -> Path:
    """Create timestamped run directory."""
    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Any, path: str | Path) -> None:
    """Save object as JSON with pretty formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir exists
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            obj, f, indent=2, ensure_ascii=False, default=str
        )  # Handle datetime objects


def save_model(model: Any, path: str | Path) -> None:
    """Save model using joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir exists
    joblib.dump(model, p)


def load_model(path: str | Path) -> Any:
    """Load model using joblib."""
    return joblib.load(Path(path))
