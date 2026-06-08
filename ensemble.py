"""Out-of-fold stacked ensemble with a Ridge meta-learner.

Mirrors notebook section 8 (Novel Contribution — Stacked Ensemble): base
learners are trained on K-1 folds and predict on the held-out fold (OOF) so
the meta-learner never sees a base prediction made on data that base model
was trained on — the key anti-leakage property of stacking.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .config import SEED
from .models import EvalResult, evaluate

logger = logging.getLogger(__name__)


def _base_models(seed: int) -> dict:
    return {
        "LR": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                     random_state=seed, n_jobs=-1),
        "XGB": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                objective="reg:squarederror", random_state=seed,
                                n_jobs=-1, verbosity=0),
        "LGB": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 num_leaves=31, subsample=0.8,
                                 random_state=seed, n_jobs=-1, verbose=-1),
    }


def train_stacked_ensemble(X_train, y_train, X_test, y_test,
                           n_folds: int = 5, seed: int = SEED) -> tuple[EvalResult, dict]:
    """Train the OOF-stacked ensemble and return its evaluation plus diagnostics
    (per-base-model OOF R² and the fitted meta-learner) for reporting.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_models_cfg = _base_models(seed)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_preds = np.zeros((n_train, len(base_models_cfg)))
    test_preds = np.zeros((n_test, len(base_models_cfg)))
    oof_r2 = {}

    t0 = time.time()
    for j, (name, base_model) in enumerate(base_models_cfg.items()):
        test_fold_preds = np.zeros((n_test, n_folds))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            Xtr_f, Xval_f = X_train[tr_idx], X_train[val_idx]
            ytr_f = y_train[tr_idx]

            clone = type(base_model)(**base_model.get_params())
            clone.fit(Xtr_f, ytr_f)
            oof_preds[val_idx, j] = clone.predict(Xval_f)
            test_fold_preds[:, fold] = clone.predict(X_test)

        test_preds[:, j] = test_fold_preds.mean(axis=1)
        oof_r2[name] = float(r2_score(y_train, oof_preds[:, j]))
        logger.info("Stacked ensemble base model %s — OOF R2 = %.4f", name, oof_r2[name])

    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds, y_train)
    y_pred_stack = meta.predict(test_preds)
    elapsed = time.time() - t0

    result = evaluate("Stacked Ensemble", y_test, y_pred_stack, elapsed, meta)
    diagnostics = {
        "base_oof_r2": oof_r2,
        "meta_learner": meta,
        "base_models": base_models_cfg,
    }
    return result, diagnostics
