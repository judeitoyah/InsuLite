"""Page 2 — Single Patient Premium Prediction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Predict · InsuLite", page_icon="🔮", layout="wide")

st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.pred-box{background:#161C27;border:2px solid #4F8EF7;border-radius:10px;
 padding:2rem;text-align:center;margin:1rem 0;}
.pred-val{font-size:3rem;font-weight:900;
 background:linear-gradient(90deg,#4F8EF7,#A78BFA);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.pred-lbl{color:#8892A4;font-size:.82rem;margin-top:.5rem;letter-spacing:.08em;}
.risk-badge{display:inline-block;padding:.4rem 1.2rem;border-radius:20px;
 font-weight:700;font-size:.9rem;margin-top:.7rem;}
.risk-low   {background:#166534;color:#4ADE80;}
.risk-medium{background:#854D0E;color:#FCD34D;}
.risk-high  {background:#7F1D1D;color:#FCA5A5;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔮 Predict Insurance Premium</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">Enter patient details to get an instant premium estimate</div>', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent.parent / "outputs" / "best_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    age      = st.slider("Age", 18, 64, 35)
    sex      = st.selectbox("Sex", ["male", "female"])
with c2:
    bmi      = st.slider("BMI", 15.0, 54.0, 27.0, 0.1)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
with c3:
    smoker   = st.selectbox("Smoker", ["no", "yes"])
    region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

def build_features(age, sex, bmi, children, smoker, region):
    """Build the exact feature vector the trained model expects."""
    sex_enc    = 1 if sex == "male" else 0
    smoker_enc = 1 if smoker == "yes" else 0
    r_ne = 1 if region == "northeast" else 0
    r_nw = 1 if region == "northwest" else 0
    r_se = 1 if region == "southeast" else 0
    r_sw = 1 if region == "southwest" else 0

    smoker_bmi   = smoker_enc * bmi
    smoker_age   = smoker_enc * age
    age_bmi      = age * bmi
    age_sq       = age ** 2
    bmi_sq       = bmi ** 2
    bmi_obese    = 1 if bmi >= 30 else 0
    age_children = age * children

    return pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex_enc": sex_enc, "smoker_enc": smoker_enc,
        "region_northeast": r_ne, "region_northwest": r_nw,
        "region_southeast": r_se, "region_southwest": r_sw,
        "smoker_bmi": smoker_bmi, "smoker_age": smoker_age,
        "age_bmi": age_bmi, "age_sq": age_sq, "bmi_sq": bmi_sq,
        "bmi_obese": bmi_obese, "age_children": age_children,
    }])

def risk_label(pred):
    if pred < 8000:   return "LOW RISK",    "risk-low"
    if pred < 20000:  return "MEDIUM RISK", "risk-medium"
    return             "HIGH RISK",         "risk-high"

predict_btn = st.button("⚡ Calculate Premium", type="primary", use_container_width=True)

if predict_btn:
    if model is None:
        st.warning("No trained model found. Go to **Run Pipeline** page to train the models first.")
    else:
        X = build_features(age, sex, bmi, children, smoker, region)
        # Keep only columns the model was trained on
        try:
            cols = model.get_booster().feature_names
            X    = X[cols]
        except Exception:
            pass

        pred = float(model.predict(X)[0])
        label, css = risk_label(pred)

        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns([2, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-val">${pred:,.2f}</div>
              <div class="pred-lbl">ESTIMATED ANNUAL PREMIUM</div>
              <div class="risk-badge {css}">{label}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)
            for k, v in [("Age", age), ("BMI", f"{bmi:.1f}"),
                         ("Children", children), ("Smoker", smoker.upper()),
                         ("Region", region.title())]:
                st.markdown(f"**{k}:** `{v}`")

        with r3:
            st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
            risks = []
            if smoker == "yes":  risks.append(("🚬 Smoker", "HIGH"))
            if bmi >= 30:        risks.append(("⚖️ Obese BMI", "MEDIUM"))
            if age >= 50:        risks.append(("📅 Age 50+", "MEDIUM"))
            if children >= 3:    risks.append(("👨‍👩‍👧 3+ Children", "LOW"))
            if not risks:        risks.append(("✅ No Major Flags", "LOW"))
            for factor, lvl in risks:
                colour = {"HIGH":"#F76E6E","MEDIUM":"#FCD34D","LOW":"#4ADE80"}[lvl]
                st.markdown(f'<span style="color:{colour};font-weight:700;">{factor}</span>', unsafe_allow_html=True)

        # ── Gauge ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Premium Gauge</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis":  {"range": [0, 65000], "tickcolor": "#8892A4"},
                "bar":   {"color": "#4F8EF7"},
                "steps": [
                    {"range": [0, 8000],  "color": "#166534"},
                    {"range": [8000, 20000], "color": "#854D0E"},
                    {"range": [20000, 65000], "color": "#7F1D1D"},
                ],
                "threshold": {"line": {"color": "#A78BFA", "width": 3}, "value": pred},
            },
            title={"text": "Annual Premium Estimate", "font": {"color": "#E8EAF0"}},
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161C27",
                          font={"color": "#E8EAF0"}, height=300)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Fill in the patient profile above and click **Calculate Premium** to get a prediction.")
