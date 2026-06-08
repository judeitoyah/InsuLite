"""SHAP-based explainability for the tuned XGBoost model.

Mirrors notebook section 11 — global feature importance (mean |SHAP|),
per-sample explanations for low/mid/high-risk profiles, and the values
needed to draw beeswarm/dependence/waterfall plots without recomputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_test: np.ndarray, feature_names: list[str]):
    """Run SHAP's TreeExplainer over the test set; returns (explainer, shap_values)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending — the global ranking."""
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    return mean_abs.sort_values(ascending=False)


def risk_profile_indices(y_test: np.ndarray) -> dict[str, int]:
    """Pick representative low/mid/high-risk sample indices from the sorted test targets."""
    order = np.argsort(y_test)
    n = len(order)
    return {
        "Low-risk profile": int(order[n // 6]),
        "Mid-risk profile": int(order[n // 2]),
        "High-risk profile": int(order[-(n // 6)]),
    }


def explain_profile(shap_values: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                     y_pred: np.ndarray, feature_names: list[str], idx: int, top_k: int = 8) -> dict:
    """Top-k SHAP contributors for a single prediction — the data behind a waterfall plot."""
    shap_row = shap_values[idx]
    feat_vals = X_test[idx]
    order = np.argsort(np.abs(shap_row))[::-1][:top_k]

    contributions = [
        {"feature": feature_names[i], "value": float(feat_vals[i]), "shap": float(shap_row[i])}
        for i in order
    ]
    return {
        "actual": float(y_test[idx]),
        "predicted": float(y_pred[idx]),
        "contributions": contributions,
    }


def explain_risk_profiles(shap_values: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                          y_pred: np.ndarray, feature_names: list[str]) -> dict[str, dict]:
    """Explanations for representative low/mid/high-risk samples (Fig 11.4 in the notebook)."""
    indices = risk_profile_indices(y_test)
    return {
        label: explain_profile(shap_values, X_test, y_test, y_pred, feature_names, idx)
        for label, idx in indices.items()
    }
