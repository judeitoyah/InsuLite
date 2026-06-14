"""SHAP explainability for tree-based models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_test: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)


def global_importance(shap_values: np.ndarray,
                      feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP per feature, descending."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_single(model, X_test: pd.DataFrame,
                   y_test, y_pred: np.ndarray,
                   idx: int, top_k: int = 5) -> dict:
    """Explain one prediction: actual, predicted, top SHAP drivers."""
    shap_vals = compute_shap_values(model, X_test)
    sv        = shap_vals[idx]
    top_idx   = np.argsort(np.abs(sv))[-top_k:]
    return {
        "actual":             float(y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]),
        "predicted":          float(y_pred[idx]),
        "top_features":       X_test.columns[top_idx].tolist(),
        "shap_contributions": sv[top_idx].tolist(),
    }


def explain_risk_profiles(model, X_test: pd.DataFrame,
                           y_test, y_pred: np.ndarray) -> dict:
    """Explain low / medium / high risk representative cases."""
    sorted_idx = np.argsort(y_test if isinstance(y_test, np.ndarray)
                            else y_test.values)
    n = len(sorted_idx)
    return {
        "low_risk":    explain_single(model, X_test, y_test, y_pred, sorted_idx[n // 4]),
        "medium_risk": explain_single(model, X_test, y_test, y_pred, sorted_idx[n // 2]),
        "high_risk":   explain_single(model, X_test, y_test, y_pred, sorted_idx[3 * n // 4]),
    }
