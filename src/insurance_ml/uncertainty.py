"""Quantile regression for prediction intervals."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def fit_quantile_models(X_train, y_train,
                        quantiles: tuple = (0.10, 0.50, 0.90)) -> dict:
    """Fit one GBR per quantile."""
    return {
        q: GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42,
        ).fit(X_train, y_train)
        for q in quantiles
    }


def predict_quantiles(models: dict, X_test) -> dict:
    """Return {quantile: predictions} for all fitted models."""
    return {q: m.predict(X_test) for q, m in models.items()}


def quantile_diagnostics(y_test, preds: dict) -> dict:
    """90% interval coverage and mean width."""
    lo, hi = preds[0.10], preds[0.90]
    return {
        "empirical_coverage": float(np.mean((y_test >= lo) & (y_test <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
    }
