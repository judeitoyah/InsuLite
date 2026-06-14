"""Feature engineering and selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression

from .config import TARGET, SEED


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and polynomial features."""
    df = df.copy()
    df["smoker_bmi"]  = df["smoker_enc"] * df["bmi"]
    df["smoker_age"]  = df["smoker_enc"] * df["age"]
    df["age_bmi"]     = df["age"] * df["bmi"]
    df["age_sq"]      = df["age"] ** 2
    df["bmi_sq"]      = df["bmi"] ** 2
    df["bmi_obese"]   = (df["bmi"] >= 30).astype(int)
    df["age_children"]= df["age"] * df["children"]
    return df


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features by mutual information with the target."""
    mi = mutual_info_regression(X, y, random_state=SEED)
    return (
        pd.DataFrame({"feature": X.columns, "mi_score": mi})
        .sort_values("mi_score", ascending=True)
        .reset_index(drop=True)
    )


def rfe_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> list[str]:
    """Select top features via Recursive Feature Elimination."""
    rf  = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rfe = RFE(rf, n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()


def select_final_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Union of MI top-10, RFE top-10, and core domain features."""
    mi_top   = mutual_information_ranking(X, y).tail(10)["feature"].tolist()
    rfe_sel  = rfe_selection(X, y, n_features=10)
    core     = ["age", "bmi", "smoker_enc", "smoker_bmi", "smoker_age"]
    return sorted(set(mi_top + rfe_sel + core))


def build_feature_matrix(df: pd.DataFrame) -> dict:
    """Orchestrate feature engineering and selection."""
    df = add_interaction_features(df)
    feature_cols = [c for c in df.columns if c != TARGET]
    X, y = df[feature_cols], df[TARGET]

    mi_ranking    = mutual_information_ranking(X, y)
    rfe_selected  = rfe_selection(X, y)
    final_features = select_final_features(X, y)

    return {
        "df_engineered":  df,
        "mi_ranking":     mi_ranking,
        "rfe_selected":   rfe_selected,
        "final_features": final_features,
    }
