"""Page 1 — Data Explorer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from insurance_ml.data import load_raw_data, dataset_overview

st.set_page_config(page_title="Data Explorer · InsuLite", page_icon="📊", layout="wide")

# ── shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.stat-card{background:#161C27;border:1px solid #2A3347;border-radius:8px;padding:.9rem 1rem;text-align:center;}
.stat-val{font-size:1.6rem;font-weight:700;color:#4F8EF7;}
.stat-lbl{font-size:.72rem;color:#8892A4;margin-top:.2rem;letter-spacing:.05em;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Data Explorer</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">NHS Medical Insurance Dataset — 1,338 patient records</div>', unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "insuranceFINAL.csv"

@st.cache_data
def get_data():
    return load_raw_data(DATA_PATH)

try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `insuranceFINAL.csv` in the `data/` folder.")
    st.stop()

ov = dataset_overview(df)

# ── summary stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
cols = st.columns(6)
stats = [
    (ov["rows"],          "RECORDS"),
    (ov["columns"],       "FEATURES"),
    (ov["total_missing"], "MISSING"),
    (f"${ov['target_mean']:,.0f}", "MEAN PREMIUM"),
    (f"${ov['target_max']:,.0f}", "MAX PREMIUM"),
    (round(ov["target_skewness"], 2), "SKEWNESS"),
]
for col, (val, lbl) in zip(cols, stats):
    col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── raw table ─────────────────────────────────────────────────────────────────
with st.expander("Raw Data Sample (first 20 rows)", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ── distributions ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(df, x="expenses", nbins=50, title="Insurance Premium Distribution",
                       color_discrete_sequence=["#4F8EF7"])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", showlegend=False,
                      title_font_color="#E8EAF0", font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(df, x="smoker", y="expenses", color="smoker", title="Premium by Smoking Status",
                 color_discrete_map={"yes": "#F76E6E", "no": "#4F8EF7"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", showlegend=False,
                      title_font_color="#E8EAF0", font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    fig = px.scatter(df, x="bmi", y="expenses", color="smoker", title="BMI vs Premium (by Smoker)",
                     color_discrete_map={"yes": "#F76E6E", "no": "#4F8EF7"}, opacity=0.7)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", title_font_color="#E8EAF0", font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.scatter(df, x="age", y="expenses", color="smoker", title="Age vs Premium (by Smoker)",
                     color_discrete_map={"yes": "#F76E6E", "no": "#4F8EF7"}, opacity=0.7)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", title_font_color="#E8EAF0", font_color="#8892A4")
    st.plotly_chart(fig, use_container_width=True)

# ── correlation heatmap ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Correlation Matrix (Numeric Features)</div>', unsafe_allow_html=True)

df_num = df[["age", "bmi", "children", "expenses"]].copy()
corr   = df_num.corr().round(3)

fig = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale="Blues", zmin=-1, zmax=1,
    text=corr.values.round(2), texttemplate="%{text}",
))
fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                  plot_bgcolor="#161C27", height=350,
                  font_color="#8892A4", title_font_color="#E8EAF0")
st.plotly_chart(fig, use_container_width=True)

# ── categorical breakdowns ────────────────────────────────────────────────────
st.markdown('<div class="section-title">Categorical Breakdowns</div>', unsafe_allow_html=True)

c5, c6, c7 = st.columns(3)

for col_name, label, c in [("smoker", "Smoker Status", c5),
                             ("sex",    "Gender",        c6),
                             ("region", "Region",        c7)]:
    counts = df[col_name].value_counts().reset_index()
    counts.columns = [col_name, "count"]
    fig = px.pie(counts, names=col_name, values="count", title=label,
                 color_discrete_sequence=["#4F8EF7","#A78BFA","#F76E6E","#50C878"])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                      plot_bgcolor="#161C27", title_font_color="#E8EAF0",
                      font_color="#8892A4", showlegend=True)
    c.plotly_chart(fig, use_container_width=True)
