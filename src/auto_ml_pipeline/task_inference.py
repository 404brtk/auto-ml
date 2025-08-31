import pandas as pd

from auto_ml_pipeline.config import TaskType


def infer_task(df: pd.DataFrame, target: str) -> TaskType:
    y = df[target]
    # classification if few unique values or object/bool/categorical
    if y.dtype.kind in {"b", "O"} or str(y.dtype).startswith("category"):
        return TaskType.classification
    nunique = y.nunique(dropna=False)
    if nunique <= 20 and y.dtype.kind in {"i", "u"}:
        return TaskType.classification
    # else regression
    return TaskType.regression
