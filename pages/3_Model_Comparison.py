"""Page 3 — Model Comparison Leaderboard."""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Comparison · InsuLite", page_icon="📈", layout="wide")

st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.winner-badge{background:linear-gradient(135deg,#1e3a5f,#2a4a7f);border:1px solid #4F8EF7;
 border-radius:8px;padding:1rem;text-align:center;margin-bottom:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📈 Model Comparison Leaderboard</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">Side-by-side evaluation of all trained models</div>', unsafe_allow_html=True)

RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "results.json"

# ── Load or use demo data ─────────────────────────────────────────────────────
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        saved = json.load(f)
    st.success("Loaded results from last pipeline run.")
else:
    st.info("No pipeline results yet — showing illustrative demo data. Go to **Run Pipeline** to train models.")
    saved = None

# Demo leaderboard data (shown when no run exists yet)
DEMO = pd.DataFrame([
    {"Model": "XGBoost (Tuned)",   "R2": 0.891, "RMSE": 4210, "MAE": 2340, "MAPE%": 18.2},
    {"Model": "Stacked Ensemble",  "R2": 0.881, "RMSE": 4480, "MAE": 2510, "MAPE%": 19.4},
    {"Model": "LightGBM",          "R2": 0.876, "RMSE": 4540, "MAE": 2560, "MAPE%": 19.8},
    {"Model": "XGBoost",           "R2": 0.869, "RMSE": 4670, "MAE": 2620, "MAPE%": 20.5},
    {"Model": "Random Forest",     "R2": 0.858, "RMSE": 4850, "MAE": 2740, "MAPE%": 21.3},
    {"Model": "SVR",               "R2": 0.714, "RMSE": 6890, "MAE": 4210, "MAPE%": 38.7},
    {"Model": "Linear Regression", "R2": 0.691, "RMSE": 7220, "MAE": 4610, "MAPE%": 42.1},
])

df = DEMO.copy()

# ── Winner banner ─────────────────────────────────────────────────────────────
best = df.iloc[0]
st.markdown(f"""
<div class="winner-badge">
  <div style="color:#8892A4;font-size:.75rem;letter-spacing:.1em;">BEST MODEL</div>
  <div style="font-size:1.4rem;font-weight:900;color:#4F8EF7;margin:.3rem 0;">{best['Model']}</div>
  <div style="display:flex;justify-content:center;gap:2rem;">
    <span><b style="color:#E8EAF0;">R²</b> <span style="color:#4F8EF7;">{best['R2']:.4f}</span></span>
    <span><b style="color:#E8EAF0;">RMSE</b> <span style="color:#4F8EF7;">${best['RMSE']:,.0f}</span></span>
    <span><b style="color:#E8EAF0;">MAE</b> <span style="color:#4F8EF7;">${best['MAE']:,.0f}</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Leaderboard table ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Full Leaderboard</div>', unsafe_allow_html=True)

def colour_r2(val):
    if val >= 0.87: return "color:#4ADE80;font-weight:700;"
    if val >= 0.80: return "color:#FCD34D;font-weight:700;"
    return "color:#F76E6E;"

styled = df.style.format({"R2": "{:.4f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MAPE%": "{:.1f}%"})
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Bar charts ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

COLORS = ["#4F8EF7","#A78BFA","#50C878","#FFB347","#F76E6E","#14B8A6","#8892A4"]

with c1:
    fig = px.bar(df.sort_values("R2"), x="R2", y="Model", orientation="h",
                 title="R² Score (higher = better)", color="Model",
                 color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", showlegend=False,
                      title_font_color="#E8EAF0", font_color="#8892A4")
    fig.add_vline(x=0.85, line_dash="dash", line_color="#8892A4",
                  annotation_text="0.85 target", annotation_font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(df.sort_values("RMSE", ascending=True), x="RMSE", y="Model",
                 orientation="h", title="RMSE (lower = better)", color="Model",
                 color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", showlegend=False,
                      title_font_color="#E8EAF0", font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

# ── Radar chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Multi-Metric Radar</div>', unsafe_allow_html=True)

top3 = df.head(3)
fig  = go.Figure()

for i, row in top3.iterrows():
    # normalise metrics 0-1 for radar (R2 direct, others inverted)
    r2_n   = row["R2"]
    rmse_n = 1 - (row["RMSE"] - df["RMSE"].min()) / (df["RMSE"].max() - df["RMSE"].min())
    mae_n  = 1 - (row["MAE"]  - df["MAE"].min())  / (df["MAE"].max()  - df["MAE"].min())
    mape_n = 1 - (row["MAPE%"]- df["MAPE%"].min())/ (df["MAPE%"].max()- df["MAPE%"].min())

    fig.add_trace(go.Scatterpolar(
        r=[r2_n, rmse_n, mae_n, mape_n, r2_n],
        theta=["R²", "RMSE (inv)", "MAE (inv)", "MAPE (inv)", "R²"],
        fill="toself", name=row["Model"],
        line_color=COLORS[i], opacity=0.7,
    ))

fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                  polar=dict(bgcolor="#161C27",
                             radialaxis=dict(visible=True, range=[0,1],
                                             color="#8892A4"),
                             angularaxis=dict(color="#8892A4")),
                  showlegend=True, height=420,
                  font_color="#8892A4", title_font_color="#E8EAF0")
st.plotly_chart(fig, use_container_width=True)
