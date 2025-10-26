import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
from pathlib import Path
from auto_ml_pipeline.logging_utils import get_logger


logger = get_logger(__name__)

pipeline = None
label_encoder = None
feature_mapping = None
target = None

app = FastAPI(
    title="auto-ml",
    description="API for serving predictions from a trained ML model.",
    version="0.1.0",
)


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    inputs: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def load_artifacts(run_dir: Path) -> None:
    global pipeline, label_encoder, feature_mapping, target

    prod_model_path = run_dir / "production_model.joblib"
    name_mapping_path = run_dir / "name_mapping.json"
    results_path = run_dir / "results.json"

    if not prod_model_path.exists():
        raise FileNotFoundError(f"production_model.joblib not found in {run_dir}")

    if not name_mapping_path.exists():
        raise FileNotFoundError(f"name_mapping.json not found in {run_dir}")

    pipeline = joblib.load(prod_model_path)

    with open(name_mapping_path, "r") as f:
        feature_mapping = json.load(f)

    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
            task = results.get("task", "unknown")
            model_name = results.get("best_model", "unknown")
            target = results.get("target", "prediction")
            logger.info(f"Loaded {task} model: {model_name}, target: {target}")
    else:
        logger.warning(f"results.json not found in {run_dir}")
        target = "prediction"

    logger.info(f"Feature mapping: {len(feature_mapping)} features")
    logger.info(f"Expected features: {sorted(feature_mapping.keys())}")

    encoder_path = run_dir / "label_encoder.joblib"
    if encoder_path.exists():
        label_encoder = joblib.load(encoder_path)
        logger.info(f"Label encoder loaded: {len(label_encoder.classes_)} classes")


def get_pipeline_features():
    if pipeline is None:
        return None

    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


@app.get("/")
def root():
    return {
        "service": "auto-ml",
        "version": "0.1.0",
        "status": "running" if pipeline else "not loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    if pipeline is None or feature_mapping is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "target": target,
        "features": {
            "expected": sorted(feature_mapping.keys()),
            "count": len(feature_mapping),
            "pipeline": get_pipeline_features(),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionPayload):
    if pipeline is None or feature_mapping is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame(payload.inputs)

        received = set(input_df.columns)
        expected = set(feature_mapping.keys())

        missing = expected - received
        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}. Expected: {sorted(expected)}"
            )

        extra = received - expected
        if extra:
            logger.warning(f"Ignoring extra features: {sorted(extra)}")

        feature_df = input_df[list(feature_mapping.keys())].copy()
        feature_df.rename(columns=feature_mapping, inplace=True)

        predictions = pipeline.predict(feature_df)

        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)

        response = [
            {target: pred.item() if hasattr(pred, "item") else pred}
            for pred in predictions
        ]

        return {"predictions": response}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        import traceback

        logger.error(f"Prediction failed: {e}")
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        raise HTTPException(status_code=400, detail=str(error_detail))


def create_app(run_dir: Path) -> FastAPI:
    logger.info(f"Loading model from: {run_dir}")
    load_artifacts(run_dir)
    logger.info("Model loaded successfully")
    return app


def start_server(run_dir: Path, host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    try:
        create_app(run_dir)
        logger.info(f"Starting server at http://{host}:{port}")
        logger.info(f"Documentation: http://{host}:{port}/docs")
        logger.info(f"Health check: http://{host}:{port}/health")

        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
