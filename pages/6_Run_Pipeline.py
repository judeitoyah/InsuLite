"""Page 6 — Run the full ML pipeline."""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

from insurance_ml.config import PipelineConfig

st.set_page_config(page_title="Run Pipeline · InsuLite", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.page-title{font-size:1.5rem;font-weight:900;color:#4F8EF7;letter-spacing:.08em;}
.section-title{color:#4F8EF7;font-size:.78rem;font-weight:700;letter-spacing:.12em;
 text-transform:uppercase;border-bottom:1px solid #2A3347;padding-bottom:.35rem;margin:1.2rem 0 .7rem 0;}
.stage-done{color:#4ADE80;} .stage-run{color:#FCD34D;} .stage-wait{color:#8892A4;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">⚙️ Run Full Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8892A4;font-size:.85rem;margin-bottom:1rem;">Train all 7 models end-to-end and save outputs to disk</div>', unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent.parent / "data" / "insuranceFINAL.csv"

if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Please place `insuranceFINAL.csv` in the `data/` folder.")
    st.stop()

# ── Configuration ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    optuna_trials = st.slider("Optuna trials", 10, 100, 60)
    cv_folds      = st.slider("CV folds", 5, 15, 10)
with c2:
    stack_folds   = st.slider("Stack folds", 3, 10, 5)
    test_split    = st.slider("Test split", 0.10, 0.30, 0.20, 0.05)
with c3:
    seed          = st.number_input("Random seed", value=42, step=1)
    output_dir    = st.text_input("Output directory", value="outputs")

st.markdown('<div class="section-title">Pipeline Stages</div>', unsafe_allow_html=True)

stages = [
    "Data loading & preprocessing",
    "Feature engineering & selection",
    "Baseline model training (5 models)",
    "Stacked ensemble (OOF)",
    "Bayesian XGBoost tuning (Optuna)",
    "K-fold cross-validation",
    "SHAP explainability + quantile regression",
]
stage_holder = st.empty()

def render_stages(done: int = -1, running: int = -1):
    html = '<div style="display:flex;flex-direction:column;gap:.4rem;">'
    for i, s in enumerate(stages):
        if i < done:
            icon, cls = "✓", "stage-done"
        elif i == running:
            icon, cls = "▶", "stage-run"
        else:
            icon, cls = "○", "stage-wait"
        html += f'<div class="{cls}">[{icon}] Stage {i+1:02d} — {s}</div>'
    html += "</div>"
    stage_holder.markdown(html, unsafe_allow_html=True)

render_stages()

run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

if run_btn:
    from insurance_ml.pipeline import run_pipeline

    config = PipelineConfig(
        data_path=str(DATA_PATH),
        output_dir=output_dir,
        train_test_split=test_split,
        cv_folds=int(cv_folds),
        stack_folds=int(stack_folds),
        optuna_trials=int(optuna_trials),
        seed=int(seed),
    )

    progress = st.progress(0)
    log_box  = st.empty()
    logs     = []

    def log(msg):
        logs.append(msg)
        log_box.code("\n".join(logs), language="")

    try:
        log("Starting InsuLite pipeline...")
        render_stages(running=0)
        progress.progress(5)

        # Run pipeline (blocking — show live stage updates via monkey-patched logger)
        import logging
        class StreamlitHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                logs.append(msg)
                log_box.code("\n".join(logs), language="")
                # update stage indicator based on message
                m = msg.lower()
                if "[1/6]" in m: render_stages(running=0); progress.progress(10)
                elif "[2/6]" in m: render_stages(done=1, running=1); progress.progress(25)
                elif "[3/6]" in m: render_stages(done=2, running=2); progress.progress(40)
                elif "[4/6]" in m: render_stages(done=3, running=3); progress.progress(55)
                elif "[5/6]" in m: render_stages(done=4, running=4); progress.progress(70)
                elif "[6/6]" in m: render_stages(done=5, running=5); progress.progress(85)

        root_logger = logging.getLogger("insurance_ml")
        handler = StreamlitHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        results = run_pipeline(config)

        render_stages(done=len(stages))
        progress.progress(100)

        root_logger.removeHandler(handler)

        st.success("Pipeline completed successfully!")

        st.markdown('<div class="section-title">Results Summary</div>', unsafe_allow_html=True)
        ev = results["tuned_eval"]
        m1, m2, m3, m4 = st.columns(4)
        for col, (val, lbl) in zip([m1, m2, m3, m4], [
            (f"{ev['r2']:.4f}", "R²"),
            (f"${ev['rmse']:,.0f}", "RMSE"),
            (f"${ev['mae']:,.0f}", "MAE"),
            (f"{ev['mape']:.2f}%", "MAPE"),
        ]):
            col.metric(lbl, val)

        st.info(f"Model saved to `{output_dir}/best_model.pkl` — reload other pages to see updated results.")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        raise
