"""Bayesian hyperparameter optimisation via Optuna."""

from __future__ import annotations

import time

import numpy as np
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, X_train, y_train) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
    }
    model = xgb.XGBRegressor(**params, n_estimators=300, random_state=42, verbosity=0)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()


def tune_xgboost(X_train, y_train, n_trials: int = 60, seed: int = 42):
    """Run Optuna study and return best XGBoost parameters."""
    study = optuna.create_study(sampler=TPESampler(seed=seed), direction="maximize")
    study.optimize(
        lambda t: _objective(t, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def retrain_best_xgboost(study, X_train, y_train, X_test, y_test) -> tuple:
    """Retrain XGBoost on full training set using best Optuna params."""
    t0    = time.time()
    model = xgb.XGBRegressor(
        **study.best_params, n_estimators=300, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, {
        "r2":   float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        "time": round(time.time() - t0, 3),
    }
