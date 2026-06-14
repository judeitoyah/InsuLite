"""Data preprocessing — encoding, scaling, log-transform."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import TARGET, LOG_TARGET, NUMERIC_COLS


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Binary-encode sex/smoker and one-hot-encode region."""
    df = df.copy()
    df["sex_enc"]    = (df["sex"]    == "male").astype(int)
    df["smoker_enc"] = (df["smoker"] == "yes").astype(int)

    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=False)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = region_dummies.columns.tolist()

    df = df.drop(["sex", "smoker", "region"], axis=1)
    return df, region_cols


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed target column."""
    df = df.copy()
    df[LOG_TARGET] = np.log1p(df[TARGET])
    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on numeric columns."""
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to numeric columns."""
    df = df.copy()
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing: encode categoricals + add log target."""
    df, _ = encode_categoricals(df)
    df    = add_log_target(df)
    return df
