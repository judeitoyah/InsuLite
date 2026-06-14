"""Data loading and validation."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "expenses"]


def load_raw_data(path) -> pd.DataFrame:
    """Load and validate the raw insurance dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def dataset_overview(df: pd.DataFrame, target: str = "expenses") -> dict:
    """Return summary statistics for the dataset."""
    return {
        "rows":               df.shape[0],
        "columns":            df.shape[1],
        "memory_kb":          df.memory_usage(deep=True).sum() / 1024,
        "total_missing":      int(df.isnull().sum().sum()),
        "missing_per_column": df.isnull().sum().to_dict(),
        "target_mean":        round(df[target].mean(), 2),
        "target_std":         round(df[target].std(), 2),
        "target_min":         round(df[target].min(), 2),
        "target_max":         round(df[target].max(), 2),
        "target_skewness":    round(df[target].skew(), 4),
    }
