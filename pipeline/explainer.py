"""
pipeline/explainer.py - handles the LLM explanation step for flagged clauses.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pipeline.classifier import ClassifiedClause


_SYSTEM_PROMPT = """You are a senior legal-risk analyst specialising in NDAs.

Respond with ONLY a valid JSON object. Each field must be exactly 2 sentences - no more, no less.

{{
  "risk_level": "Low" | "Medium" | "High",
  "plain_english_summary": "Sentence 1 explaining what the clause says. Sentence 2 explaining who it benefits.",
  "why_risky": "Sentence 1 on the main legal risk. Sentence 2 on the worst-case financial or legal consequence.",
  "what_to_negotiate": "Sentence 1 on what specific change to demand. Sentence 2 on what fair alternative wording looks like."
}}

Rules:
- Exactly 2 sentences per field
- Be specific and detailed within those 2 sentences
- Always complete and close the JSON
- No text outside the JSON
- No markdown fences"""

_HUMAN_PROMPT = """Clause category: {category}
Confidence score: {confidence}

Clause text:
{clause_text}

Respond with the JSON only."""


@st.cache_resource(show_spinner=False, max_entries=1)
def _build_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Run: export GROQ_API_KEY=your-key-here"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])

    return prompt | llm | StrOutputParser()


def explain_flagged_clauses(
    clauses: List[ClassifiedClause],
    threshold: float = 0.65,
) -> Dict[int, dict]:
    chain = _build_chain()
    explanations: Dict[int, dict] = {}

    for idx, clause in enumerate(clauses):
        if clause.confidence < threshold:
            continue

        try:
            raw = chain.invoke({
                "category": clause.category,
                "confidence": f"{clause.confidence:.0%}",
                "clause_text": clause.text[:800],
            })
            explanations[idx] = _parse_response(raw)

        except Exception as exc:
            explanations[idx] = {
                "risk_level": "Unknown",
                "plain_english_summary": "Explanation unavailable.",
                "why_risky": f"API error: {exc}",
                "what_to_negotiate": "-",
            }

    return explanations


def _parse_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "risk_level": "Unknown",
            "plain_english_summary": cleaned[:500],
            "why_risky": "Could not parse structured response from model.",
            "what_to_negotiate": "-",
        }
