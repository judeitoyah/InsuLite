"""InsuLite — Streamlit home page."""

import streamlit as st

st.set_page_config(
    page_title="InsuLite",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace; }

/* ── Header logo ── */
.logo-box {
    background: linear-gradient(135deg, #0E1117 0%, #161C27 100%);
    border: 1px solid #2A3347;
    border-radius: 8px;
    padding: 2rem 2rem 1.2rem 2rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.logo-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: clamp(2.8rem, 6vw, 4.8rem);
    font-weight: 900;
    letter-spacing: 0.18em;
    background: linear-gradient(90deg, #4F8EF7 0%, #A78BFA 50%, #4F8EF7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    text-shadow: none;
}
.logo-sub {
    color: #8892A4;
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* ── Badges ── */
.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center; margin:1rem 0; }
.badge {
    display:inline-flex; align-items:center; gap:0.35rem;
    padding:0.28rem 0.75rem; border-radius:4px;
    font-size:0.78rem; font-weight:700; letter-spacing:0.06em;
}
.badge-label { background:#2A3347; color:#8892A4; }
.badge-val-blue   { background:#4F8EF7; color:#fff; }
.badge-val-green  { background:#22C55E; color:#fff; }
.badge-val-orange { background:#F59E0B; color:#fff; }
.badge-val-purple { background:#A78BFA; color:#fff; }
.badge-val-red    { background:#EF4444; color:#fff; }
.badge-val-teal   { background:#14B8A6; color:#fff; }

/* ── Metric cards ── */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; margin:1.5rem 0; }
.metric-card {
    background:#161C27; border:1px solid #2A3347; border-radius:8px;
    padding:1.1rem 1rem; text-align:center;
}
.metric-card .val { font-size:1.8rem; font-weight:700; color:#4F8EF7; }
.metric-card .lbl { font-size:0.75rem; color:#8892A4; margin-top:0.25rem; letter-spacing:0.05em; }

/* ── Section headers ── */
.section-title {
    color:#4F8EF7; font-size:0.8rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase;
    border-bottom:1px solid #2A3347; padding-bottom:0.4rem;
    margin:1.5rem 0 0.8rem 0;
}

/* ── Pipeline stages ── */
.stage-row { display:flex; flex-direction:column; gap:0.5rem; }
.stage {
    display:flex; align-items:center; gap:0.75rem;
    background:#161C27; border:1px solid #2A3347;
    border-radius:6px; padding:0.65rem 1rem;
}
.stage-num { color:#4F8EF7; font-weight:700; min-width:1.5rem; }
.stage-name { color:#E8EAF0; font-weight:600; }
.stage-desc { color:#8892A4; font-size:0.82rem; margin-left:auto; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#0E1117; border-right:1px solid #2A3347; }
</style>
""", unsafe_allow_html=True)

# ── Logo ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-box">
  <div class="logo-text">INSULITE</div>
  <div class="logo-sub">Medical Insurance Premium Prediction &amp; Explainability Platform</div>
  <div class="badge-row" style="margin-top:1rem;">
    <span class="badge"><span class="badge-label">PYTHON</span><span class="badge-val-blue">3.10+</span></span>
    <span class="badge"><span class="badge-label">STREAMLIT</span><span class="badge-val-red">1.33+</span></span>
    <span class="badge"><span class="badge-label">XGBOOST</span><span class="badge-val-orange">2.0+</span></span>
    <span class="badge"><span class="badge-label">SHAP</span><span class="badge-val-purple">EXPLAINABLE</span></span>
    <span class="badge"><span class="badge-label">OPTUNA</span><span class="badge-val-teal">BAYESIAN</span></span>
    <span class="badge"><span class="badge-label">LICENSE</span><span class="badge-val-green">MIT</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
for val, lbl in [
    ("1,338", "TRAINING RECORDS"),
    ("7",     "INPUT FEATURES"),
    ("5+",    "ML MODELS"),
    ("SHAP",  "EXPLAINABILITY"),
    ("90%",   "QUANTILE CI"),
    ("R²>0.86","BEST MODEL"),
]:
    st.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Pipeline stages ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">7-Stage ML Pipeline</div>', unsafe_allow_html=True)

stages = [
    ("01", "Data Ingestion",             "Load & validate NHS insurance CSV"),
    ("02", "Preprocessing",              "Encode categoricals · Log-transform · Scale"),
    ("03", "Feature Engineering",        "Interactions · Polynomial · MI ranking · RFE"),
    ("04", "Model Training",             "LR · SVR · RF · XGBoost · LightGBM baselines"),
    ("05", "Stacked Ensemble",           "OOF stacking with Ridge meta-learner"),
    ("06", "Bayesian Tuning",            "Optuna TPE · 60-trial XGBoost optimisation"),
    ("07", "Explainability & Uncertainty","SHAP values · Quantile regression intervals"),
]

st.markdown('<div class="stage-row">', unsafe_allow_html=True)
for num, name, desc in stages:
    st.markdown(f"""
    <div class="stage">
      <span class="stage-num">{num}</span>
      <span class="stage-name">{name}</span>
      <span class="stage-desc">{desc}</span>
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Navigation guide ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Navigate</div>', unsafe_allow_html=True)

cols = st.columns(3)
nav = [
    ("📊", "Data Explorer",    "Dataset distributions, correlations and summary stats"),
    ("🔮", "Predict",          "Instant premium estimate for a single patient profile"),
    ("📈", "Model Comparison", "Side-by-side RMSE · MAE · R² · MAPE leaderboard"),
    ("🧠", "Explainability",   "SHAP feature importance and individual risk drivers"),
    ("❓", "Uncertainty",      "90% prediction intervals via quantile regression"),
    ("⚙️", "Run Pipeline",     "Train all models end-to-end on your own dataset"),
]
for i, (icon, title, desc) in enumerate(nav):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
          <div style="font-size:1.4rem;">{icon}</div>
          <div style="color:#E8EAF0;font-weight:700;margin-top:0.3rem;">{title}</div>
          <div class="lbl" style="text-align:left;margin-top:0.3rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
      <div style="font-size:1.6rem;font-weight:900;
        background:linear-gradient(90deg,#4F8EF7,#A78BFA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        INSULITE
      </div>
      <div style="color:#8892A4;font-size:0.72rem;letter-spacing:0.1em;margin-top:0.2rem;">
        v1.0.0 · MIT
      </div>
    </div>
    <hr style="border-color:#2A3347;margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Pages**")
    st.page_link("app.py",                        label="🏠  Home")
    st.page_link("pages/1_Data_Explorer.py",       label="📊  Data Explorer")
    st.page_link("pages/2_Predict.py",             label="🔮  Predict Premium")
    st.page_link("pages/3_Model_Comparison.py",    label="📈  Model Comparison")
    st.page_link("pages/4_Explainability.py",      label="🧠  Explainability")
    st.page_link("pages/5_Uncertainty.py",         label="❓  Uncertainty")
    st.page_link("pages/6_Run_Pipeline.py",        label="⚙️  Run Pipeline")

    st.markdown("""
    <hr style="border-color:#2A3347;margin:1rem 0 0.5rem 0;">
    <div style="color:#8892A4;font-size:0.72rem;text-align:center;">
      Built by Jude Isememe Itoyah<br>github.com/judeitoyah
    </div>""", unsafe_allow_html=True)
