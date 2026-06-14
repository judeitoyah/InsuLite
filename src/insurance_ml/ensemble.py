"""Out-of-fold stacked ensemble."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def train_stacked_ensemble(X_train, y_train, X_test, y_test,
                            n_folds: int = 5, seed: int = 42) -> dict:
    """Out-of-fold stacking: LR + RF + XGB + LGB → Ridge meta-learner."""
    base_models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=200, random_state=seed),
        xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=seed, verbosity=0),
        lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=seed, verbosity=-1),
    ]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof  = np.zeros((len(X_train), len(base_models)))
    test_preds = np.zeros((len(X_test), len(base_models)))

    for i, model in enumerate(base_models):
        fold_test = np.zeros(len(X_test))
        for tr_idx, va_idx in kf.split(X_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr       = y_train.iloc[tr_idx]
            model.fit(Xtr, ytr)
            oof[va_idx, i] = model.predict(Xva)
            fold_test     += model.predict(X_test) / n_folds
        test_preds[:, i] = fold_test

    meta = Ridge()
    meta.fit(oof, y_train)
    y_pred = meta.predict(test_preds)

    return {
        "meta_learner":  meta,
        "r2":            float(r2_score(y_test, y_pred)),
        "rmse":          float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":           float(mean_absolute_error(y_test, y_pred)),
        "mape":          float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        "oof_predictions": oof,
        "test_predictions": y_pred,
    }
