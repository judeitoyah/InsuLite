"""K-fold cross-validation and Wilcoxon statistical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVR


def cv_model_zoo() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "svr":               SVR(kernel="rbf"),
        "random_forest":     RandomForestRegressor(n_estimators=300, random_state=42),
        "xgboost":           xgb.XGBRegressor(n_estimators=500, random_state=42, verbosity=0),
        "lightgbm":          lgb.LGBMRegressor(n_estimators=500, random_state=42, verbosity=-1),
    }


def run_kfold_cv(X, y, models: dict,
                 cv_folds: int = 10, seed: int = 42) -> dict:
    """Execute K-fold CV and return per-model score arrays."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=kf,
            scoring=["r2", "neg_mean_squared_error"],
            return_train_score=False,
        )
        results[name] = {
            "r2_scores":   scores["test_r2"],
            "rmse_scores": np.sqrt(-scores["test_neg_mean_squared_error"]),
        }
    return results


def wilcoxon_vs_best(cv_results: dict) -> tuple[str, dict]:
    """Wilcoxon signed-rank test of each model vs the best."""
    best = max(cv_results, key=lambda n: np.mean(cv_results[n]["r2_scores"]))
    best_r2 = cv_results[best]["r2_scores"]
    tests = {}
    for name, res in cv_results.items():
        if name == best:
            continue
        stat, p = wilcoxon(best_r2, res["r2_scores"])
        tests[name] = {"p_value": round(p, 4), "significant": p < 0.05}
    return best, tests


def cv_summary_table(cv_results: dict) -> pd.DataFrame:
    """Build a tidy summary DataFrame from CV results."""
    rows = []
    for name, res in cv_results.items():
        r2   = res["r2_scores"]
        rmse = res["rmse_scores"]
        rows.append({
            "model":     name,
            "R2_mean":   round(np.mean(r2), 4),
            "R2_std":    round(np.std(r2), 4),
            "R2_CI95":   round(1.96 * np.std(r2) / np.sqrt(len(r2)), 4),
            "RMSE_mean": round(np.mean(rmse), 2),
            "RMSE_std":  round(np.std(rmse), 2),
        })
    return pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
