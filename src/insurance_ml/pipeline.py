"""End-to-end ML pipeline orchestration."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PipelineConfig
from .data import load_raw_data, dataset_overview
from .ensemble import train_stacked_ensemble
from .explainability import explain_risk_profiles
from .features import build_feature_matrix
from .models import train_baselines
from .preprocessing import apply_scaler, fit_scaler, preprocess
from .tuning import retrain_best_xgboost, tune_xgboost
from .uncertainty import fit_quantile_models, predict_quantiles, quantile_diagnostics
from .validation import cv_model_zoo, cv_summary_table, run_kfold_cv

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):          return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):return int(obj)
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return super().default(obj)


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the complete 6-stage pipeline and return results."""

    # ── Stage 1: Data ────────────────────────────────────────────────────────
    logger.info("[1/6] Loading and preprocessing data")
    df_raw  = load_raw_data(config.data_path)
    overview = dataset_overview(df_raw)
    df      = preprocess(df_raw)
    scaler  = fit_scaler(df)
    df      = apply_scaler(df, scaler)

    feat_out    = build_feature_matrix(df)
    final_feats = feat_out["final_features"]
    X = df[final_feats]
    y = df["expenses"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.train_test_split, random_state=config.seed
    )

    # ── Stage 2: Baseline models ─────────────────────────────────────────────
    logger.info("[2/6] Training baseline models")
    baseline = train_baselines(X_tr, y_tr, X_te, y_te, seed=config.seed)

    # ── Stage 3: Stacked ensemble ────────────────────────────────────────────
    logger.info("[3/6] Training stacked ensemble")
    ensemble = train_stacked_ensemble(X_tr, y_tr, X_te, y_te,
                                      n_folds=config.stack_folds, seed=config.seed)

    # ── Stage 4: Bayesian tuning ─────────────────────────────────────────────
    logger.info("[4/6] Bayesian hyperparameter tuning (XGBoost)")
    study             = tune_xgboost(X_tr, y_tr, n_trials=config.optuna_trials, seed=config.seed)
    tuned_model, tuned_eval = retrain_best_xgboost(study, X_tr, y_tr, X_te, y_te)

    # ── Stage 5: Validation ──────────────────────────────────────────────────
    logger.info("[5/6] Cross-validation")
    cv_res  = run_kfold_cv(X, y, cv_model_zoo(), cv_folds=config.cv_folds, seed=config.seed)
    cv_table = cv_summary_table(cv_res)

    # ── Stage 6: Explainability & uncertainty ────────────────────────────────
    logger.info("[6/6] Explainability + uncertainty quantification")
    y_pred_te     = tuned_model.predict(X_te)
    risk_profiles = explain_risk_profiles(tuned_model, X_te, y_te, y_pred_te)
    q_models      = fit_quantile_models(X_tr, y_tr, quantiles=config.quantiles)
    q_preds       = predict_quantiles(q_models, X_te)
    q_diag        = quantile_diagnostics(y_te, q_preds)

    # ── Persist ──────────────────────────────────────────────────────────────
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_model.pkl", "wb") as f:
        pickle.dump(tuned_model, f)
    with open(out / "results.json", "w") as f:
        json.dump({"tuned_eval": tuned_eval, "q_diagnostics": q_diag},
                  f, cls=_NumpyEncoder, indent=2)

    logger.info("Pipeline complete.")
    return {
        "overview":    overview,
        "feat_out":    feat_out,
        "baseline":    baseline,
        "ensemble":    ensemble,
        "tuned_model": tuned_model,
        "tuned_eval":  tuned_eval,
        "cv_table":    cv_table,
        "risk_profiles": risk_profiles,
        "q_preds":     q_preds,
        "q_diag":      q_diag,
        "X_te": X_te, "y_te": y_te, "y_pred_te": y_pred_te,
        "X_tr": X_tr, "y_tr": y_tr,
        "scaler": scaler, "final_features": final_feats,
    }
