"""Dataset loading and basic structural checks."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "expenses"]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw insurance CSV and validate its expected schema.

    Raises FileNotFoundError / ValueError early so pipeline failures point at
    the dataset rather than surfacing as confusing errors deep in modelling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place insuranceFINAL.csv there or "
            "pass --data /path/to/insuranceFINAL.csv."
        )

    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset at '{path}' is missing expected columns: {missing_cols}")

    logger.info(
        "Loaded %s: %d rows x %d columns, %d missing values",
        path, df.shape[0], df.shape[1], int(df.isnull().sum().sum()),
    )
    return df


def dataset_overview(df: pd.DataFrame, target: str) -> dict:
    """Summary stats used both for logging and for the EDA report."""
    return {
        "shape": df.shape,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "target_skew": float(df[target].skew()),
    }
