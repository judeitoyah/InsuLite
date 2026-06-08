"""Quantile regression for prediction intervals (uncertainty quantification).

Mirrors notebook section 12 — separate Gradient Boosting quantile regressors
for the 10th/50th/90th percentiles give an [Q10, Q90] interval whose
empirical coverage can be checked against the nominal 80%.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .config import SEED


def fit_quantile_models(X_train, y_train, quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
                        seed: int = SEED) -> dict[float, GradientBoostingRegressor]:
    """Fit one GradientBoostingRegressor(loss='quantile') per requested quantile."""
    models = {}
    for q in quantiles:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300,
                                           max_depth=5, learning_rate=0.05, random_state=seed)
        model.fit(X_train, y_train)
        models[q] = model
    return models


def predict_quantiles(models: dict[float, GradientBoostingRegressor], X) -> dict[float, np.ndarray]:
    return {q: model.predict(X) for q, model in models.items()}


def quantile_diagnostics(y_test: np.ndarray, preds: dict[float, np.ndarray],
                         lower: float = 0.10, upper: float = 0.90) -> dict:
    """Empirical coverage of the [lower, upper] interval and per-quantile MAE.

    Coverage near (upper - lower) confirms the interval is well-calibrated;
    `mean_width` quantifies how informative (vs. merely wide) it is.
    """
    in_interval = (y_test >= preds[lower]) & (y_test <= preds[upper])
    coverage = float(in_interval.mean())
    width = preds[upper] - preds[lower]

    mae_by_quantile = {q: float(mean_absolute_error(y_test, p)) for q, p in preds.items()}

    return {
        "target_coverage": upper - lower,
        "empirical_coverage": coverage,
        "mean_interval_width": float(width.mean()),
        "mae_by_quantile": mae_by_quantile,
    }
