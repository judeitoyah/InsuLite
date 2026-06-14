"""Page 5 — Quantile Regression Uncertainty."""

import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from insurance_ml.data import load_raw_data
from insurance_ml.preprocessing import preprocess, fit_scaler, apply_scaler
from insurance_ml.features import build_feature_matrix
from insurance_ml.uncertainty import fit_quantile_models, predict_quantiles, quantile_diagnostics

st.set_page_config(page_title="Uncertainty · InsuLite", page_icon="❓", layout="wide")

st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.diag-card{background:#161C27;border:1px solid #2A3347;border-radius:8px;
 padding:1.2rem;text-align:center;}
.diag-val{font-size:1.8rem;font-weight:700;color:#4F8EF7;}
.diag-lbl{font-size:.72rem;color:#8892A4;margin-top:.3rem;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">❓ Prediction Uncertainty</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">90% prediction intervals via quantile gradient boosting regression</div>', unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent.parent / "data" / "insuranceFINAL.csv"

@st.cache_data
def get_quantile_results():
    df = load_raw_data(DATA_PATH)
    df = preprocess(df)
    sc = fit_scaler(df)
    df = apply_scaler(df, sc)
    feat = build_feature_matrix(df)
    X = df[feat["final_features"]]
    y = df["expenses"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    q_models = fit_quantile_models(X_tr, y_tr, quantiles=(0.10, 0.50, 0.90))
    q_preds  = predict_quantiles(q_models, X_te)
    diag     = quantile_diagnostics(y_te, q_preds)
    return y_te, q_preds, diag

try:
    with st.spinner("Fitting quantile models..."):
        y_te, q_preds, diag = get_quantile_results()
except FileNotFoundError:
    st.error("Dataset not found. Place `insuranceFINAL.csv` in the `data/` folder.")
    st.stop()

# ── Diagnostics ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">90% Interval Diagnostics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

diag_items = [
    (f"{diag['empirical_coverage']:.1%}", "EMPIRICAL COVERAGE", "target: 90%"),
    (f"${diag['mean_interval_width']:,.0f}", "MEAN INTERVAL WIDTH", "avg spread"),
    ("90%", "TARGET COVERAGE", "quantile 10–90"),
]
for col, (val, lbl, sub) in zip([c1, c2, c3], diag_items):
    col.markdown(f"""<div class="diag-card">
      <div class="diag-val">{val}</div>
      <div class="diag-lbl">{lbl}</div>
      <div style="color:#4F8EF7;font-size:.7rem;margin-top:.2rem;">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Interval fan chart ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Predicted vs Actual with 90% Intervals</div>', unsafe_allow_html=True)

sort_idx = np.argsort(y_te.values)
y_sorted = y_te.values[sort_idx]
lo       = q_preds[0.10][sort_idx]
med      = q_preds[0.50][sort_idx]
hi       = q_preds[0.90][sort_idx]
x_range  = list(range(len(y_sorted)))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_range + x_range[::-1],
    y=hi.tolist() + lo[::-1].tolist(),
    fill="toself", fillcolor="rgba(79,142,247,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="90% Interval",
))
fig.add_trace(go.Scatter(x=x_range, y=lo,       mode="lines",
    line=dict(color="#2A3347", width=1), name="10th pct"))
fig.add_trace(go.Scatter(x=x_range, y=hi,       mode="lines",
    line=dict(color="#2A3347", width=1), name="90th pct"))
fig.add_trace(go.Scatter(x=x_range, y=med,      mode="lines",
    line=dict(color="#4F8EF7", width=2), name="Median (50th)"))
fig.add_trace(go.Scatter(x=x_range, y=y_sorted, mode="markers",
    marker=dict(color="#F76E6E", size=3, opacity=0.6), name="Actual"))

fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                  plot_bgcolor="#161C27", height=420,
                  xaxis_title="Patient (sorted by actual premium)",
                  yaxis_title="Insurance Premium ($)",
                  legend=dict(orientation="h", y=-0.15),
                  font_color="#8892A4", title_font_color="#E8EAF0")
st.plotly_chart(fig, use_container_width=True)

# ── Coverage by risk band ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Coverage by Risk Band</div>', unsafe_allow_html=True)

bands = {"Low (<$8k)": (0, 8000), "Medium ($8k–$20k)": (8000, 20000), "High (>$20k)": (20000, 1e9)}
rows  = []
for band, (lo_v, hi_v) in bands.items():
    mask = (y_te.values >= lo_v) & (y_te.values < hi_v)
    if mask.sum() == 0:
        continue
    cov = np.mean((y_te.values[mask] >= q_preds[0.10][mask]) &
                  (y_te.values[mask] <= q_preds[0.90][mask]))
    rows.append({"Risk Band": band, "Coverage": round(float(cov), 4),
                 "N Patients": int(mask.sum())})

band_df = pd.DataFrame(rows)
fig2 = go.Figure(go.Bar(
    x=band_df["Risk Band"], y=band_df["Coverage"],
    marker_color=["#4ADE80", "#FCD34D", "#F76E6E"],
    text=[f"{v:.1%}" for v in band_df["Coverage"]], textposition="outside",
))
fig2.add_hline(y=0.90, line_dash="dash", line_color="#8892A4",
               annotation_text="90% target", annotation_font_color="#8892A4")
fig2.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                   plot_bgcolor="#161C27", yaxis_range=[0, 1.05],
                   yaxis_title="Coverage", height=320,
                   font_color="#8892A4")
st.plotly_chart(fig2, use_container_width=True)
