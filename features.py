"""Domain-driven interaction features and feature selection.

Mirrors notebook sections 5 (Feature Engineering) and 6 (Feature Selection):
mutual information, RFE with a Random Forest, and the union rule that builds
the final feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression

from .config import INTERACTION_COLS, SEED, TARGET


def add_interaction_features(df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered interaction/polynomial columns in `INTERACTION_COLS`.

    `df_raw` supplies the un-scaled BMI used for the clinical obesity cutoff
    (>= 30), independent of whatever scaling has been applied to `df`.
    """
    df = df.copy()
    df["smoker_bmi"] = df["smoker_enc"] * df["bmi"]
    df["smoker_age"] = df["smoker_enc"] * df["age"]
    df["age_bmi"] = df["age"] * df["bmi"]
    df["age_sq"] = df["age"] ** 2
    df["bmi_sq"] = df["bmi"] ** 2
    df["bmi_obese"] = (df_raw["bmi"] >= 30).astype(int)
    df["smoker_obese"] = df["smoker_enc"] * df["bmi_obese"]
    df["age_children"] = df["age"] * df["children"]

    missing = [c for c in INTERACTION_COLS if c not in df.columns]
    assert not missing, f"Interaction feature build is out of sync with config: {missing}"
    return df


def base_feature_list(region_cols: list[str]) -> list[str]:
    return ["age", "bmi", "children", "sex_enc", "smoker_enc"] + region_cols


def mutual_information_ranking(X: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> pd.Series:
    """Mutual information of every candidate feature with the target, sorted ascending."""
    scores = mutual_info_regression(X, y, random_state=seed)
    return pd.Series(scores, index=X.columns).sort_values(ascending=True)


def rfe_selection(X: pd.DataFrame, y: np.ndarray, n_features: int = 10, seed: int = SEED) -> list[str]:
    """Recursive Feature Elimination using a Random Forest, returns selected names."""
    estimator = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    ranking = pd.Series(rfe.ranking_, index=X.columns)
    return ranking[ranking == 1].index.tolist()


def select_final_features(
    X_all: pd.DataFrame,
    y: np.ndarray,
    rfe_n_features: int = 10,
    seed: int = SEED,
    always_keep: tuple[str, ...] = ("age", "bmi", "smoker_enc", "smoker_bmi", "smoker_age"),
) -> list[str]:
    """Union of top-10 mutual-information features, RFE-selected features and a
    fixed core set known from domain knowledge to drive premiums (smoker/age/bmi).
    """
    mi_series = mutual_information_ranking(X_all, y, seed=seed)
    top10_mi = mi_series.tail(10).index.tolist()
    selected_rfe = rfe_selection(X_all, y, n_features=rfe_n_features, seed=seed)

    final = sorted(set(top10_mi) | set(selected_rfe) | set(always_keep))
    return final


def build_feature_matrix(df_feat: pd.DataFrame, region_cols: list[str], config_seed: int = SEED) -> dict:
    """Run the full feature-engineering + selection stage.

    Returns a dict with the candidate matrix, MI ranking, RFE selection and
    the final chosen feature list — everything the report/CLI needs to log.
    """
    base_feats = base_feature_list(region_cols)
    all_feats = base_feats + INTERACTION_COLS

    X_all = df_feat[all_feats].copy()
    y = df_feat[TARGET].values

    mi_series = mutual_information_ranking(X_all, y, seed=config_seed)
    selected_rfe = rfe_selection(X_all, y, seed=config_seed)
    final_feats = select_final_features(X_all, y, seed=config_seed)

    return {
        "base_feats": base_feats,
        "all_feats": all_feats,
        "mi_series": mi_series,
        "selected_rfe": selected_rfe,
        "final_feats": final_feats,
    }
