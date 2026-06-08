"""Bayesian hyperparameter optimisation of XGBoost via Optuna.

Mirrors notebook section 9: a TPE-sampled search over XGBoost's main
regularisation/capacity knobs, scored by 5-fold CV R², followed by a retrain
of the winning configuration on the full training split.
"""

from __future__ import annotations

import time

import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from .config import SEED
from .models import EvalResult, evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial, X_train, y_train, seed: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    return float(scores.mean())


def tune_xgboost(X_train, y_train, n_trials: int = 60, seed: int = SEED) -> optuna.Study:
    """Run the Optuna search and return the completed study (best_params, trials)."""
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: _objective(trial, X_train, y_train, seed),
                   n_trials=n_trials, show_progress_bar=False)
    return study


def retrain_best_xgboost(study: optuna.Study, X_train, y_train, X_test, y_test,
                         seed: int = SEED) -> EvalResult:
    """Retrain XGBoost with the study's best hyperparameters on the full train split."""
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    })

    t0 = time.time()
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate("XGBoost (Optuna)", y_test, y_pred, time.time() - t0, model)
