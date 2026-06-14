"""Baseline regression models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class EvalResult:
    r2:            float
    rmse:          float
    mae:           float
    mape:          float
    training_time: float

    def to_dict(self) -> dict:
        return {
            "R²":            round(self.r2, 4),
            "RMSE":          round(self.rmse, 2),
            "MAE":           round(self.mae, 2),
            "MAPE%":         round(self.mape, 3),
            "Train Time (s)":round(self.training_time, 3),
        }


def _evaluate(y_true, y_pred) -> tuple[float, float, float, float]:
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return r2, rmse, mae, mape


def train_linear_regression(X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    m  = LinearRegression().fit(X_tr, y_tr)
    return m, EvalResult(*_evaluate(y_te, m.predict(X_te)), time.time() - t0)


def train_svr(X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    sc = StandardScaler()
    m  = SVR(kernel="rbf").fit(sc.fit_transform(X_tr), y_tr)
    return m, EvalResult(*_evaluate(y_te, m.predict(sc.transform(X_te))), time.time() - t0)


def train_random_forest(X_tr, y_tr, X_te, y_te, seed: int = 42):
    t0 = time.time()
    m  = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
    return m, EvalResult(*_evaluate(y_te, m.predict(X_te)), time.time() - t0)


def train_xgboost(X_tr, y_tr, X_te, y_te, seed: int = 42):
    t0 = time.time()
    m  = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbosity=0,
    ).fit(X_tr, y_tr)
    return m, EvalResult(*_evaluate(y_te, m.predict(X_te)), time.time() - t0)


def train_lightgbm(X_tr, y_tr, X_te, y_te, seed: int = 42):
    t0 = time.time()
    m  = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=seed, verbosity=-1,
    ).fit(X_tr, y_tr)
    return m, EvalResult(*_evaluate(y_te, m.predict(X_te)), time.time() - t0)


def train_baselines(X_tr, y_tr, X_te, y_te, seed: int = 42) -> dict:
    """Train all five baseline models and return results dict."""
    trainers = {
        "linear_regression": train_linear_regression,
        "svr":               train_svr,
        "random_forest":     train_random_forest,
        "xgboost":           train_xgboost,
        "lightgbm":          train_lightgbm,
    }
    return {
        name: {"model": m, "eval": e}
        for name, fn in trainers.items()
        for m, e in [fn(X_tr, y_tr, X_te, y_te, seed) if name not in ("linear_regression", "svr")
                     else fn(X_tr, y_tr, X_te, y_te)]
    }
