"""Categorical encoding, target transform and numeric scaling.

Mirrors notebook section 3 (Data Preprocessing), split into composable
functions so each transform can be unit-tested and reused at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import LOG_TARGET, NUMERIC_COLS, TARGET


def encode_categoricals(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Binary-encode sex/smoker and one-hot encode region.

    Returns the encoded frame plus the list of generated `region_*` columns,
    since downstream feature lists need to reference them by name.
    """
    df = df_raw.copy()
    df["sex_enc"] = (df["sex"] == "male").astype(int)
    df["smoker_enc"] = (df["smoker"] == "yes").astype(int)

    df = pd.get_dummies(df, columns=["region"], prefix="region", drop_first=False)
    region_cols = [c for c in df.columns if c.startswith("region_")]

    df = df.drop(columns=["sex", "smoker"])
    return df, region_cols


def add_log_target(df: pd.DataFrame, target: str = TARGET, log_col: str = LOG_TARGET) -> pd.DataFrame:
    """Add a log1p-transformed target column (the raw target is right-skewed)."""
    df = df.copy()
    df[log_col] = np.log1p(df[target])
    return df


def fit_scaler(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> StandardScaler:
    """Fit a StandardScaler on the given numeric columns."""
    numeric_cols = numeric_cols or NUMERIC_COLS
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Apply a fitted scaler, returning a copy with the scaled columns replaced."""
    numeric_cols = numeric_cols or NUMERIC_COLS
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def preprocess(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run the full encode -> log-target pipeline used before feature engineering."""
    df, region_cols = encode_categoricals(df_raw)
    df = add_log_target(df)
    return df, region_cols
