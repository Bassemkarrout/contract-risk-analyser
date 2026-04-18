"""
app.py — Entry point for the Contract Risk Analyser Streamlit app.

INTERVIEW TALKING POINT:
    Streamlit was chosen over Flask/FastAPI because it lets you build
    data-heavy UIs in pure Python with zero front-end code. For a CV
    project targeting a consulting firm like PwC, the ability to ship
    a polished demo quickly matters more than full web-framework control.
    The trade-off is limited customisation and single-threaded execution,
    which would matter at production scale but is fine for a prototype.
"""

import streamlit as st
from pipeline.pdf_extractor import extract_clauses
from pipeline.classifier import classify_clauses
from pipeline.risk_aggregator import aggregate_risk
from pipeline.explainer import explain_flagged_clauses
from ui.dashboard import render_dashboard

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NDA Risk Analyser",
    page_icon="⚖️",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚖️ NDA Contract Risk Analyser")
st.caption(
    "Upload an NDA PDF. The pipeline extracts clauses, classifies risk categories "
    "with a HuggingFace model, aggregates an overall risk score, and uses "
    "Gemini (via LangChain) to explain each flagged clause in plain English."
)

st.divider()

# ── Sidebar — configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    risk_threshold = st.slider(
        label="Flag clauses above confidence",
        min_value=0.50,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help=(
            "Clauses whose risk-category confidence score exceeds this threshold "
            "are sent to Gemini for plain-English explanation. "
            "Lower → more clauses explained (slower, more API cost). "
            "Higher → only the most certain risks explained."
        ),
    )

    st.markdown("---")
    st.markdown(
        "**Risk scoring weights**\n\n"
        "Each category carries a different weight when computing the overall score. "
        "Liability and indemnification are weighted highest because they carry the "
        "greatest legal and financial exposure in a typical NDA."
    )
    # Weights are defined here so a PwC interviewer can see the business logic
    # surfaced at the UI level — easy to change without touching core code.
    CATEGORY_WEIGHTS = {
        "liability":        0.25,
        "indemnification":  0.25,
        "termination":      0.15,
        "penalty":          0.15,
        "exclusivity":      0.10,
        "confidentiality":  0.10,
    }
    for cat, w in CATEGORY_WEIGHTS.items():
        st.write(f"• **{cat.capitalize()}**: {int(w*100)}%")

# ── Main upload area ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload NDA (.pdf)",
    type=["pdf"],
    help="Only PDF files are accepted. Scanned PDFs without an OCR layer may "
         "produce poor text extraction — see README for details.",
)

if uploaded_file is None:
    st.info("👆 Upload an NDA PDF to begin analysis.")
    st.stop()

# ── Pipeline ────────────────────────────────────────────────────────────────────
# Each step is a separate module so you can unit-test, swap, or explain
# each component independently during an interview.

with st.spinner("Step 1 / 4 — Extracting clauses from PDF…"):
    clauses = extract_clauses(uploaded_file)

if not clauses:
    st.error(
        "No text could be extracted from this PDF. "
        "If it is a scanned document, please run it through an OCR tool first."
    )
    st.stop()

with st.spinner("Step 2 / 4 — Classifying clauses with HuggingFace model…"):
    classified = classify_clauses(clauses)

with st.spinner("Step 3 / 4 — Aggregating overall risk score…"):
    risk_summary = aggregate_risk(classified, weights=CATEGORY_WEIGHTS)

with st.spinner("Step 4 / 4 — Generating plain-English explanations via Gemini…"):
    explained = explain_flagged_clauses(classified, threshold=risk_threshold)

# ── Render dashboard ────────────────────────────────────────────────────────────
render_dashboard(
    clauses=classified,
    risk_summary=risk_summary,
    explanations=explained,
    threshold=risk_threshold,
)
