# InsuLite

**Medical Insurance Premium Prediction & Explainability Platform**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainable-purple?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

InsuLite is a modular, end-to-end machine learning pipeline for predicting medical insurance premiums, served through a professional Streamlit web interface. It covers every stage from raw data ingestion through feature engineering, model comparison, Bayesian hyperparameter tuning, SHAP explainability and quantile uncertainty quantification.

| Item | Detail |
|---|---|
| **Dataset** | 1,338 patient records — age, BMI, sex, smoker status, region, children |
| **Target** | Annual medical insurance premium (USD) |
| **Models** | Linear Regression · SVR · Random Forest · XGBoost · LightGBM · Stacked Ensemble |
| **Best R²** | > 0.89 (Tuned XGBoost) |
| **Interface** | 6-page Streamlit app with dark monospace theme |

---

## Repository Structure

```
InsuLite/
├── app.py                          ← Streamlit home page (logo, badges, pipeline overview)
├── pages/
│   ├── 1_Data_Explorer.py          ← Dataset distributions, correlations, breakdowns
│   ├── 2_Predict.py                ← Single-patient premium estimator with gauge
│   ├── 3_Model_Comparison.py       ← Leaderboard, bar charts, radar chart
│   ├── 4_Explainability.py         ← SHAP global importance, beeswarm, risk profiles
│   ├── 5_Uncertainty.py            ← 90% quantile intervals, coverage by risk band
│   └── 6_Run_Pipeline.py           ← Live end-to-end training with progress + logs
├── src/
│   └── insurance_ml/               ← Importable Python package
│       ├── __init__.py
│       ├── config.py               ← Constants, PipelineConfig dataclass
│       ├── data.py                 ← CSV loading and validation
│       ├── preprocessing.py        ← Encoding, scaling, log-transform
│       ├── features.py             ← Feature engineering and selection
│       ├── models.py               ← Five baseline regression models
│       ├── ensemble.py             ← OOF stacked ensemble
│       ├── tuning.py               ← Optuna Bayesian XGBoost tuning
│       ├── uncertainty.py          ← Quantile regression intervals
│       ├── validation.py           ← K-fold CV and Wilcoxon tests
│       ├── explainability.py       ← SHAP TreeExplainer
│       └── pipeline.py             ← 6-stage orchestrator
├── data/
│   └── insuranceFINAL.csv          ← Source dataset (1,338 rows)
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   └── test_features.py
├── .streamlit/
│   └── config.toml                 ← Dark monospace theme
├── pyproject.toml
├── requirements.txt
└── cli.py                          ← Command-line entry point
```

---

## Pipeline Stages

The pipeline runs 6 sequential stages:

| Stage | Module | Description |
|---|---|---|
| 1 | `data.py` + `preprocessing.py` | Load CSV, encode categoricals, scale numerics, log-transform target |
| 2 | `features.py` | Add 7 interaction features, rank by mutual information, apply RFE |
| 3 | `models.py` | Train 5 baseline models with full evaluation metrics |
| 4 | `ensemble.py` | Out-of-fold stacking — LR + RF + XGB + LGB → Ridge meta-learner |
| 5 | `tuning.py` | Optuna TPE Bayesian search, 60 trials, 5-fold CV objective |
| 6 | `explainability.py` + `uncertainty.py` | SHAP values + 90% quantile prediction intervals |

---

## Feature Engineering

| Category | Features |
|---|---|
| Autoregressive | `age`, `bmi`, `children` |
| Encoded | `sex_enc`, `smoker_enc`, `region_*` (one-hot) |
| Interactions | `smoker_bmi`, `smoker_age`, `age_bmi` |
| Polynomial | `age_sq`, `bmi_sq` |
| Binary flags | `bmi_obese` (BMI ≥ 30) |
| Cross terms | `age_children` |

Selection: union of top-10 by LassoCV, top-10 by Random Forest RFE, and 5 core domain features.

---

## Models

| Model | Implementation | Notes |
|---|---|---|
| Linear Regression | scikit-learn | Baseline |
| SVR (RBF) | scikit-learn | Scaled input |
| Random Forest | scikit-learn | 300 trees, n_jobs=-1 |
| XGBoost | xgboost | 500 rounds, L1+L2 reg |
| LightGBM | lightgbm | 500 rounds, 31 leaves |
| Stacked Ensemble | Ridge meta-learner | OOF predictions from above 4 |
| XGBoost (Tuned) | Optuna TPE | 60-trial Bayesian search |

---

## Streamlit App Pages

| Page | What you get |
|---|---|
| **Home** | Branded landing with logo, tech badges, pipeline overview, navigation guide |
| **Data Explorer** | Premium distribution, BMI vs premium scatter, seasonal boxplot, correlation heatmap, categorical pie charts |
| **Predict** | Patient profile sliders → instant premium estimate with risk gauge and risk-factor breakdown |
| **Model Comparison** | Full leaderboard table, R² and RMSE bar charts, multi-metric radar chart for top-3 models |
| **Explainability** | Global SHAP bar chart, beeswarm value distribution, individual SHAP waterfall for low/medium/high risk profiles |
| **Uncertainty** | Quantile fan chart (10th–90th percentile), empirical coverage diagnostics, coverage by risk band |
| **Run Pipeline** | Configure hyperparameters, launch training, watch live stage progress and log stream, view results summary |

---

## Installation

```bash
# Clone
git clone https://github.com/judeitoyah/InsuLite.git
cd InsuLite

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

---

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Command-line pipeline

```bash
python cli.py --data data/insuranceFINAL.csv --output outputs --trials 60
```

### Python API

```python
from insurance_ml.config import PipelineConfig
from insurance_ml.pipeline import run_pipeline

config  = PipelineConfig(data_path="data/insuranceFINAL.csv", optuna_trials=60)
results = run_pipeline(config)

print(f"Best R²  : {results['tuned_eval']['r2']:.4f}")
print(f"Best RMSE: ${results['tuned_eval']['rmse']:,.0f}")
```

---

## Results

Evaluated on a 20% held-out test set (268 patients).

| Model | R² | RMSE ($) | MAE ($) | MAPE% |
|---|---|---|---|---|
| XGBoost (Tuned) | **0.891** | **4,210** | **2,340** | **18.2** |
| Stacked Ensemble | 0.881 | 4,480 | 2,510 | 19.4 |
| LightGBM | 0.876 | 4,540 | 2,560 | 19.8 |
| XGBoost | 0.869 | 4,670 | 2,620 | 20.5 |
| Random Forest | 0.858 | 4,850 | 2,740 | 21.3 |
| SVR | 0.714 | 6,890 | 4,210 | 38.7 |
| Linear Regression | 0.691 | 7,220 | 4,610 | 42.1 |

**Key driver:** `smoker_enc` and `smoker_bmi` dominate SHAP importance — smoking status alone accounts for the largest share of premium variance.

---

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for full dependency list

---

## Author

**Jude Isememe Itoyah**  
Data Scientist & Analyst  
[github.com/judeitoyah](https://github.com/judeitoyah)

---

## Licence

MIT — see [LICENSE](LICENSE) for details.
