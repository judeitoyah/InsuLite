"""Page 4 — SHAP Explainability."""

import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from insurance_ml.data import load_raw_data
from insurance_ml.preprocessing import preprocess, fit_scaler, apply_scaler
from insurance_ml.features import build_feature_matrix

st.set_page_config(page_title="Explainability · InsuLite", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.risk-card{background:#161C27;border:1px solid #2A3347;border-radius:8px;padding:1rem;}
.risk-title{font-weight:700;margin-bottom:.5rem;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🧠 SHAP Explainability</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">Feature importance and individual prediction drivers via SHAP</div>', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent.parent / "outputs" / "best_model.pkl"
DATA_PATH  = Path(__file__).parent.parent / "data"  / "insuranceFINAL.csv"

if not MODEL_PATH.exists():
    st.warning("No trained model found. Go to **Run Pipeline** first.")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def prepare_data():
    df = load_raw_data(DATA_PATH)
    df = preprocess(df)
    scaler = fit_scaler(df)
    df = apply_scaler(df, scaler)
    feat = build_feature_matrix(df)
    X = df[feat["final_features"]]
    y = df["expenses"]
    return X, y, feat["final_features"]

try:
    model      = load_model()
    X, y, feats = prepare_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

import shap

@st.cache_data
def get_shap(_model, _X):
    explainer  = shap.TreeExplainer(_model)
    shap_vals  = explainer.shap_values(_X)
    return shap_vals

st.markdown('<div class="section-title">Computing SHAP values...</div>', unsafe_allow_html=True)
with st.spinner("Running TreeExplainer..."):
    shap_values = get_shap(model, X)

# ── Global importance bar ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Global Feature Importance (mean |SHAP|)</div>', unsafe_allow_html=True)

mean_abs = np.mean(np.abs(shap_values), axis=0)
imp_df   = pd.DataFrame({"feature": feats, "importance": mean_abs}).sort_values("importance")

fig = go.Figure(go.Bar(
    x=imp_df["importance"], y=imp_df["feature"],
    orientation="h",
    marker_color=["#4F8EF7" if v > imp_df["importance"].median() else "#2A3347"
                  for v in imp_df["importance"]],
))
fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                  plot_bgcolor="#161C27", height=max(300, len(feats)*22),
                  xaxis_title="Mean |SHAP value|",
                  font_color="#8892A4", title_font_color="#E8EAF0")
st.plotly_chart(fig, use_container_width=True)

# ── SHAP beeswarm (summary scatter) ──────────────────────────────────────────
st.markdown('<div class="section-title">SHAP Value Distribution (top 10 features)</div>', unsafe_allow_html=True)

top10   = imp_df.tail(10)["feature"].tolist()
top_idx = [feats.index(f) for f in top10]

fig2 = go.Figure()
for i, (feat_name, idx) in enumerate(zip(top10, top_idx)):
    sv   = shap_values[:, idx]
    fval = X.iloc[:, idx].values
    fig2.add_trace(go.Scatter(
        x=sv, y=[feat_name] * len(sv),
        mode="markers",
        marker=dict(color=fval, colorscale="RdBu_r",
                    size=4, opacity=0.6,
                    colorbar=dict(title="Feature Value") if i == 0 else None),
        name=feat_name, showlegend=False,
    ))

fig2.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                   plot_bgcolor="#161C27", height=420,
                   xaxis_title="SHAP Value (impact on prediction)",
                   font_color="#8892A4")
st.plotly_chart(fig2, use_container_width=True)

# ── Risk profile explanations ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Individual Risk Profiles</div>', unsafe_allow_html=True)

y_arr  = y.values
sorted_idx = np.argsort(y_arr)
n = len(sorted_idx)
profiles = {
    "Low Risk":    sorted_idx[n // 4],
    "Medium Risk": sorted_idx[n // 2],
    "High Risk":   sorted_idx[3 * n // 4],
}
colours = {"Low Risk": "#4ADE80", "Medium Risk": "#FCD34D", "High Risk": "#F76E6E"}

cols = st.columns(3)
for col, (label, idx) in zip(cols, profiles.items()):
    sv       = shap_values[idx]
    top_k    = np.argsort(np.abs(sv))[-5:]
    top_feat = [feats[i] for i in top_k]
    top_shap = [sv[i] for i in top_k]
    actual   = float(y_arr[idx])
    predicted= float(model.predict(X.iloc[[idx]])[0])

    with col:
        colour = colours[label]
        st.markdown(f'<div class="risk-card"><div class="risk-title" style="color:{colour};">{label}</div>', unsafe_allow_html=True)
        st.markdown(f"**Actual:** `${actual:,.0f}`  \n**Predicted:** `${predicted:,.0f}`")

        fig3 = go.Figure(go.Bar(
            x=top_shap, y=top_feat, orientation="h",
            marker_color=["#4F8EF7" if v > 0 else "#F76E6E" for v in top_shap],
        ))
        fig3.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                           plot_bgcolor="#161C27", height=220,
                           margin=dict(l=0, r=0, t=10, b=0),
                           font_color="#8892A4", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
