"""Statistical validation: K-fold CV distributions and Wilcoxon significance tests.

Mirrors notebook section 10 — a single point-estimate R² is not enough to
claim one model "wins"; this module produces the CV distributions and the
paired non-parametric test that backs up such a claim.
"""

from __future__ import annotations

import logging

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import SEED

logger = logging.getLogger(__name__)


def cv_model_zoo(seed: int, optuna_best_params: dict | None = None) -> dict:
    """Models compared via cross-validation (Deep FNN excluded — too slow for 10-fold CV)."""
    zoo = {
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": Pipeline([("sc", StandardScaler()),
                               ("svr", SVR(kernel="rbf", C=100, gamma="scale"))]),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                      random_state=seed, n_jobs=-1, verbose=-1),
    }
    if optuna_best_params:
        params = dict(optuna_best_params)
        params.update({"objective": "reg:squarederror", "random_state": seed,
                       "n_jobs": -1, "verbosity": 0})
        zoo["XGBoost (Optuna)"] = xgb.XGBRegressor(**params)
    return zoo


def run_kfold_cv(X, y, models: dict, n_folds: int = 10, seed: int = SEED) -> tuple[dict, dict]:
    """Run K-fold CV for every model in `models`, returning per-model R² and RMSE arrays."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_r2, cv_rmse = {}, {}

    for name, model in models.items():
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                                scoring="neg_mean_squared_error", n_jobs=-1))
        cv_r2[name] = r2_scores
        cv_rmse[name] = rmse_scores
        logger.info("%d-fold CV %-22s R2 = %.4f +/- %.4f", n_folds, name,
                    r2_scores.mean(), r2_scores.std())

    return cv_r2, cv_rmse


def wilcoxon_vs_best(cv_r2: dict, best_model: str) -> list[dict]:
    """Paired Wilcoxon signed-rank test of `best_model`'s CV R² against every
    other model's, returning rows ready for a results table.
    """
    rows = []
    for name, scores in cv_r2.items():
        if name == best_model:
            continue
        stat, p = stats.wilcoxon(cv_r2[best_model], scores)
        direction = "better" if cv_r2[best_model].mean() > scores.mean() else "worse"
        rows.append({
            "Comparison": f"{best_model} vs {name}",
            "p-value": round(float(p), 4),
            "Significant (p<0.05)": bool(p < 0.05),
            "Direction": direction,
        })
    return rows


def cv_summary_table(cv_r2: dict, cv_rmse: dict, n_folds: int = 10) -> list[dict]:
    """Mean +/- std and 95% CI for each model's CV R² and RMSE."""
    rows = []
    for name in cv_r2:
        r2_mean, r2_std = cv_r2[name].mean(), cv_r2[name].std()
        rmse_mean, rmse_std = cv_rmse[name].mean(), cv_rmse[name].std()
        ci95 = 1.96 * r2_std / np.sqrt(n_folds)
        rows.append({
            "Model": name,
            "Mean R2": round(float(r2_mean), 4),
            "Std R2": round(float(r2_std), 4),
            "95% CI": [round(float(r2_mean - ci95), 4), round(float(r2_mean + ci95), 4)],
            "Mean RMSE": round(float(rmse_mean), 2),
            "Std RMSE": round(float(rmse_std), 2),
        })
    return rows
