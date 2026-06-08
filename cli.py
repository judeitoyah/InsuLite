"""Command-line entry point: `insurance-ml` (or `python -m insurance_ml.cli`)."""

from __future__ import annotations

import argparse

from .config import PipelineConfig
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the insurance premium prediction ML pipeline end-to-end.",
    )
    parser.add_argument("--data", default="data/insuranceFINAL.csv",
                        help="Path to insuranceFINAL.csv (default: data/insuranceFINAL.csv)")
    parser.add_argument("--output", default="outputs",
                        help="Directory to write models & results summary (default: outputs)")
    parser.add_argument("--trials", type=int, default=60,
                        help="Number of Optuna trials for XGBoost tuning (default: 60)")
    parser.add_argument("--cv-folds", type=int, default=10,
                        help="Number of folds for the statistical validation stage (default: 10)")
    parser.add_argument("--stack-folds", type=int, default=5,
                        help="Number of OOF folds for the stacked ensemble (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        data_path=args.data,
        output_dir=args.output,
        n_optuna_trials=args.trials,
        n_cv_folds=args.cv_folds,
        n_stack_folds=args.stack_folds,
        seed=args.seed,
    )

    result = run_pipeline(config)

    best = result["training"]["best"]
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print(f" Best model      : {best.model}")
    print(f" Test R^2        : {best.r2:.4f}")
    print(f" Test RMSE       : {best.rmse:.2f}")
    print(f" Test MAE        : {best.mae:.2f}")
    print(f" Artefacts       : {result['output_dir'].resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
