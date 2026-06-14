"""Configuration and constants for the InsuLite pipeline."""

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED       = 42
TARGET     = "expenses"
LOG_TARGET = "log_expenses"

NUMERIC_COLS     = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]

FEATURE_GROUPS = {
    "core_numeric":    ["age", "bmi", "children"],
    "core_binary":     ["sex_enc", "smoker_enc"],
    "core_categorical":["region_northeast", "region_northwest", "region_southeast"],
    "interactions":    [
        "smoker_bmi", "smoker_age", "age_bmi",
        "age_sq", "bmi_sq", "bmi_obese", "age_children",
    ],
}

PALETTE = [
    "#4F8EF7", "#F76E6E", "#50C878", "#FFB347",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#95A5A6",
]

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "svr":               "Support Vector Regression",
    "random_forest":     "Random Forest",
    "xgboost":           "XGBoost",
    "lightgbm":          "LightGBM",
    "tuned_xgboost":     "XGBoost (Tuned)",
    "stacked_ensemble":  "Stacked Ensemble",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration container for the pipeline."""
    data_path:        str   = "data/insuranceFINAL.csv"
    output_dir:       str   = "outputs"
    train_test_split: float = 0.2
    cv_folds:         int   = 10
    stack_folds:      int   = 5
    optuna_trials:    int   = 60
    rfe_n_features:   int   = 10
    quantiles:        tuple = (0.10, 0.50, 0.90)
    plot:             bool  = True
    seed:             int   = SEED
