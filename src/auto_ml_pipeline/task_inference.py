import pandas as pd

from auto_ml_pipeline.config import TaskType


def infer_task(df: pd.DataFrame, target: str) -> TaskType:
    y = df[target]
    # Try to robustly determine if a target is numeric even when stored as strings
    # e.g., values like "5,953" should be treated as numeric
    if y.dtype.kind in {"b"}:
        return TaskType.classification

    # Handle pandas categorical dtype
    if str(y.dtype).startswith("category"):
        return TaskType.classification

    # If object, attempt numeric coercion (remove common thousand separators)
    if y.dtype.kind in {"O"}:
        y_str = y.astype(str).str.strip()
        # Remove commas (thousand separators). Avoid aggressive symbol stripping to not
        # misclassify true categorical strings.
        y_num = pd.to_numeric(y_str.str.replace(",", "", regex=False), errors="coerce")
        non_na_ratio = float(y_num.notna().mean()) if len(y_num) else 0.0
        if non_na_ratio >= 0.95:
            # Largely numeric-like
            nunique_num = y_num.nunique(dropna=True)
            if nunique_num <= 20:
                # Small number of discrete numeric classes: classification
                return TaskType.classification
            # Many unique numeric values: regression
            return TaskType.regression
        # Mostly non-numeric strings -> classification
        return TaskType.classification

    # Numeric dtypes
    nunique = y.nunique(dropna=False)
    if y.dtype.kind in {"i", "u"} and nunique <= 20:
        return TaskType.classification
    # Floats or many unique integers -> regression
    return TaskType.regression
