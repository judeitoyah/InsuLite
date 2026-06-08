"""End-to-end orchestration of the insurance premium pipeline.

Each stage below corresponds to a notebook section (data -> preprocessing ->
features -> baselines -> stacking -> tuning -> validation -> explainability ->
uncertainty -> results). `run_pipeline` wires them together and persists the
artefacts an application or report would need.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from . import explainability, features, models, preprocessing, uncertainty, validation
from .config import PipelineConfig, TARGET
from .data import dataset_overview, load_raw_data
from .ensemble import train_stacked_ensemble
from .tuning import retrain_best_xgboost, tune_xgboost

logger = logging.getLogger(__name__)


def _prepare_data(config: PipelineConfig):
    """Stages 1-6: load, preprocess, engineer features, select, and split."""
    df_raw = load_raw_data(config.data_path)
    overview = dataset_overview(df_raw, TARGET)
    logger.info("Dataset overview: %s", overview)

    df_encoded, region_cols = preprocessing.preprocess(df_raw)
    df_feat = features.add_interaction_features(df_encoded, df_raw)

    feature_selection = features.build_feature_matrix(df_feat, region_cols, config.seed)
    final_feats = feature_selection["final_feats"]

    X = df_feat[final_feats].values
    y = df_feat[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed,
    )

    return {
        "df_raw": df_raw,
        "overview": overview,
        "region_cols": region_cols,
        "feature_selection": feature_selection,
        "final_feats": final_feats,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def _train_models(data: dict, config: PipelineConfig):
    """Stages 7-9: baselines, stacked ensemble, and Optuna-tuned XGBoost."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    baseline_results = models.train_baselines(X_train, y_train, X_test, y_test, seed=config.seed)

    stack_result, stack_diag = train_stacked_ensemble(
        X_train, y_train, X_test, y_test, n_folds=config.n_stack_folds, seed=config.seed,
    )

    study = tune_xgboost(X_train, y_train, n_trials=config.n_optuna_trials, seed=config.seed)
    optuna_result = retrain_best_xgboost(study, X_train, y_train, X_test, y_test, seed=config.seed)

    all_results = baseline_results + [stack_result, optuna_result]
    all_results.sort(key=lambda r: r.r2, reverse=True)

    return {
        "baseline_results": baseline_results,
        "stack_result": stack_result,
        "stack_diagnostics": stack_diag,
        "study": study,
        "optuna_result": optuna_result,
        "all_results": all_results,
        "best": all_results[0],
    }


def _validate(data: dict, training: dict, config: PipelineConfig):
    """Stage 10: K-fold CV across the model zoo plus Wilcoxon significance tests."""
    zoo = validation.cv_model_zoo(config.seed, optuna_best_params=training["study"].best_params)
    cv_r2, cv_rmse = validation.run_kfold_cv(data["X"], data["y"], zoo,
                                             n_folds=config.n_cv_folds, seed=config.seed)

    best_cv_model = max(cv_r2, key=lambda name: cv_r2[name].mean())
    wilcoxon_rows = validation.wilcoxon_vs_best(cv_r2, best_cv_model)
    summary_rows = validation.cv_summary_table(cv_r2, cv_rmse, n_folds=config.n_cv_folds)

    return {
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "best_cv_model": best_cv_model,
        "wilcoxon": wilcoxon_rows,
        "summary": summary_rows,
    }


def _explain(data: dict, training: dict):
    """Stage 11: SHAP global importance and per-profile explanations for the tuned XGBoost."""
    xgb_opt = training["optuna_result"].fitted
    final_feats = data["final_feats"]
    X_test, y_test = data["X_test"], data["y_test"]

    _, shap_values = explainability.compute_shap_values(xgb_opt, X_test, final_feats)
    importance = explainability.global_importance(shap_values, final_feats)

    y_pred = xgb_opt.predict(X_test)
    profiles = explainability.explain_risk_profiles(shap_values, X_test, y_test, y_pred, final_feats)

    return {
        "shap_values": shap_values,
        "global_importance": importance,
        "risk_profiles": profiles,
    }


def _quantify_uncertainty(data: dict, config: PipelineConfig):
    """Stage 12: quantile regression intervals and their empirical coverage."""
    qmodels = uncertainty.fit_quantile_models(data["X_train"], data["y_train"],
                                               quantiles=config.quantiles, seed=config.seed)
    qpreds = uncertainty.predict_quantiles(qmodels, data["X_test"])
    diagnostics = uncertainty.quantile_diagnostics(data["y_test"], qpreds,
                                                    lower=min(config.quantiles),
                                                    upper=max(config.quantiles))
    return {"models": qmodels, "predictions": qpreds, "diagnostics": diagnostics}


def _persist(data: dict, training: dict, validation_out: dict, explain_out: dict,
             uncertainty_out: dict, config: PipelineConfig) -> Path:
    """Save the best model, the tuned XGBoost, and a JSON results summary to `output_dir`."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(training["best"].fitted, out_dir / "best_model.joblib")
    joblib.dump(training["optuna_result"].fitted, out_dir / "xgboost_optuna.joblib")
    joblib.dump({"final_features": data["final_feats"], "region_cols": data["region_cols"]},
                out_dir / "feature_metadata.joblib")

    summary = {
        "dataset_overview": data["overview"],
        "final_features": data["final_feats"],
        "test_results": [r.as_dict() for r in training["all_results"]],
        "best_model": training["best"].model,
        "stacked_ensemble_oof_r2": training["stack_diagnostics"]["base_oof_r2"],
        "optuna_best_params": training["study"].best_params,
        "optuna_best_cv_r2": training["study"].best_value,
        "cv_summary": validation_out["summary"],
        "cv_best_model": validation_out["best_cv_model"],
        "wilcoxon_vs_best": validation_out["wilcoxon"],
        "shap_global_importance": explain_out["global_importance"].round(4).to_dict(),
        "risk_profile_explanations": explain_out["risk_profiles"],
        "uncertainty": uncertainty_out["diagnostics"],
    }

    with open(out_dir / "results_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    logger.info("Artefacts written to %s", out_dir.resolve())
    return out_dir


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def run_pipeline(config: PipelineConfig | None = None) -> dict:
    """Run the full pipeline and return every stage's outputs plus the artefact path."""
    config = config or PipelineConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Stage 1-6: data, preprocessing, feature engineering & selection")
    data = _prepare_data(config)
    logger.info("Final feature set (%d): %s", len(data["final_feats"]), data["final_feats"])

    logger.info("Stage 7-9: baselines, stacked ensemble, Optuna-tuned XGBoost")
    training = _train_models(data, config)
    logger.info("Best model on hold-out: %s (R2=%.4f)", training["best"].model, training["best"].r2)

    logger.info("Stage 10: %d-fold cross-validation & significance testing", config.n_cv_folds)
    validation_out = _validate(data, training, config)

    logger.info("Stage 11: SHAP explainability")
    explain_out = _explain(data, training)

    logger.info("Stage 12: quantile regression & uncertainty quantification")
    uncertainty_out = _quantify_uncertainty(data, config)

    out_dir = _persist(data, training, validation_out, explain_out, uncertainty_out, config)

    return {
        "config": config,
        "data": data,
        "training": training,
        "validation": validation_out,
        "explainability": explain_out,
        "uncertainty": uncertainty_out,
        "output_dir": out_dir,
    }
