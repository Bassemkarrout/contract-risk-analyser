"""
ui/dashboard.py - renders the Streamlit dashboard.

kept separate from app.py so the pipeline modules can be tested
independently without needing a running Streamlit instance.

risk colors depends
green = low, amber = medium, red = high.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st
import pandas as pd

from pipeline.classifier import ClassifiedClause
from pipeline.risk_aggregator import RiskSummary


def _score_to_bg(confidence: float) -> str:
    if confidence >= 0.70:
        return "#FFCCCC"
    elif confidence >= 0.50:
        return "#FFF3CC"
    else:
        return "#CCFFCC"


def _score_to_border(confidence: float) -> str:
    if confidence >= 0.70:
        return "#E63946"
    elif confidence >= 0.50:
        return "#F4A261"
    else:
        return "#2A9D8F"


def _rating_colour(rating: str) -> str:
    return {"Low": "#00cc66", "Medium": "orange", "High": "red"}.get(rating, "grey")


def render_dashboard(
    clauses: List[ClassifiedClause],
    risk_summary: RiskSummary,
    explanations: Dict[int, dict],
    threshold: float,
) -> None:
    _render_overall_score(risk_summary, threshold)
    st.divider()
    _render_category_chart(risk_summary)
    st.divider()
    _render_clause_list(clauses, explanations, threshold)


def _render_overall_score(risk_summary: RiskSummary, threshold: float) -> None:
    st.subheader("Overall Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    bg = {"High": "#1a1a1a", "Medium": "#1a1a1a", "Low": "#1a1a1a"}.get(risk_summary.rating, "#1a1a1a")
    colour = _rating_colour(risk_summary.rating)

    with col1:
        st.markdown(
            f"""<div style="background:{bg};border-radius:12px;padding:20px;text-align:center;">
                <h1 style="margin:0;color:{colour};">{risk_summary.rating}</h1>
                <p style="margin:0;font-size:0.85rem;">Overall Risk Rating</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Risk Score", f"{risk_summary.score:.1f} / 100")
    with col3:
        st.metric("Clauses Analysed", risk_summary.clause_count)
    with col4:
        st.metric(
            "Flagged Clauses",
            risk_summary.flagged_count,
            help="Clauses above the confidence threshold that received LLM explanations.",
        )

    if risk_summary.rating == "High":
        st.error(
            "This NDA contains multiple high-risk clauses. Legal review is strongly "
            "recommended before signing."
        )
    elif risk_summary.rating == "Medium":
        st.warning(
            "Some clauses warrant attention. Review the flagged items below and "
            "consider negotiating the highlighted terms."
        )
    else:
        st.success("No major risk clauses detected. Standard due-diligence review recommended.")


def _render_category_chart(risk_summary: RiskSummary) -> None:
    st.subheader("Risk by Category")

    df = pd.DataFrame({
        "Category": [k.capitalize() for k in risk_summary.category_scores],
        "Risk Score (0-100)": list(risk_summary.category_scores.values()),
    }).sort_values("Risk Score (0-100)", ascending=False)

    st.bar_chart(df.set_index("Category")["Risk Score (0-100)"], height=250)
    st.caption("Scores are weighted by category importance (see sidebar).")


def _render_clause_list(
    clauses: List[ClassifiedClause],
    explanations: Dict[int, dict],
    threshold: float,
) -> None:
    st.subheader(f"Clause Analysis ({len(clauses)} clauses)")

    flagged_count = sum(1 for c in clauses if c.confidence >= threshold)
    tab_all, tab_flagged = st.tabs([
        f"All Clauses ({len(clauses)})",
        f"Flagged Only ({flagged_count})",
    ])

    with tab_all:
        for idx, clause in enumerate(clauses):
            _render_clause_card(idx, clause, explanations, threshold)

    with tab_flagged:
        any_flagged = False
        for idx, clause in enumerate(clauses):
            if clause.confidence >= threshold:
                _render_clause_card(idx, clause, explanations, threshold)
                any_flagged = True
        if not any_flagged:
            st.info("No clauses met the threshold. Try lowering the slider in the sidebar.")


def _render_clause_card(
    idx: int,
    clause: ClassifiedClause,
    explanations: Dict[int, dict],
    threshold: float,
) -> None:
    is_flagged = clause.confidence >= threshold
    flag = "[FLAGGED]" if is_flagged else ""
    label = (
        f"{flag} Clause {idx + 1} - "
        f"{clause.category.capitalize()} "
        f"({clause.confidence:.0%} confidence)"
    )

    with st.expander(label, expanded=is_flagged):
        st.markdown(
            f"""<div style="
                border-left:4px solid {_score_to_border(clause.confidence)};
                padding:12px;
                font-size:0.9rem;line-height:1.6;
            ">{clause.text}</div>""",
            unsafe_allow_html=True,
        )

        with st.expander("All category scores", expanded=False):
            scores_df = pd.DataFrame({
                "Category": [k.capitalize() for k in clause.all_scores],
                "Confidence": [f"{v:.1%}" for v in clause.all_scores.values()],
            })
            st.dataframe(scores_df, hide_index=True, use_container_width=True)

        if is_flagged and idx in explanations:
            exp = explanations[idx]
            st.markdown("---")
            st.markdown("**Risk Explanation (Llama 3.3 via Groq)**")

            risk_level = exp.get("risk_level", "Unknown")
            badge_colour = _rating_colour(risk_level)
            st.markdown(
                f'<span style="background:{badge_colour};color:white;'
                f'padding:2px 8px;border-radius:8px;font-size:0.8rem;">'
                f'{risk_level} Risk</span>',
                unsafe_allow_html=True,
            )
            st.write("")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Plain English**")
                st.write(exp.get("plain_english_summary", "-"))
                st.markdown("**Why it's risky**")
                st.write(exp.get("why_risky", "-"))
            with col_b:
                st.markdown("**What to negotiate**")
                st.write(exp.get("what_to_negotiate", "-"))

        elif is_flagged:
            st.info("Explanation could not be generated for this clause.")
