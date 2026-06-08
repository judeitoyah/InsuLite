"""Shared constants for the insurance premium pipeline.

Centralising these means every stage (preprocessing, features, models,
explainability) agrees on the target column, random seed and column groups
without re-deriving them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


SEED = 42
TARGET = "expenses"
LOG_TARGET = "log_expenses"

NUMERIC_COLS = ["age", "bmi", "children"]

INTERACTION_COLS = [
    "smoker_bmi",
    "smoker_age",
    "age_bmi",
    "age_sq",
    "bmi_sq",
    "bmi_obese",
    "smoker_obese",
    "age_children",
]

PALETTE = [
    "#2E86AB", "#E84855", "#3BB273", "#F4A261",
    "#9B59B6", "#1ABC9C", "#E67E22", "#2C3E50",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Run-level configuration threaded through the pipeline stages."""

    data_path: str = "data/insuranceFINAL.csv"
    output_dir: str = "outputs"
    test_size: float = 0.20
    n_cv_folds: int = 10
    n_stack_folds: int = 5
    n_optuna_trials: int = 60
    rfe_n_features: int = 10
    seed: int = SEED
    make_plots: bool = False
    quantiles: tuple[float, ...] = field(default=(0.10, 0.50, 0.90))
