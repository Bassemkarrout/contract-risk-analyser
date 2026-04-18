"""
pipeline/classifier.py — Step 2: Classify each clause into a risk category.

WHAT IT DOES:
    Uses a zero-shot classification pipeline from HuggingFace Transformers
    to label each extracted clause with one of six NDA risk categories and
    a confidence score.

INTERVIEW TALKING POINTS:

WHY zero-shot classification?
    Zero-shot classification (ZSC) lets us classify text into arbitrary
    label sets without any labelled training data. The model — typically
    a Natural Language Inference (NLI) model fine-tuned on MNLI — is
    prompted: "Does this text entail the hypothesis '<label>'?"
    For a CV project with no annotated NDA clause dataset, ZSC is the
    pragmatic choice. It sacrifices accuracy for speed-to-deploy.

WHICH MODEL and why?
    "facebook/bart-large-mnli" is the canonical ZSC model on the HF Hub.
    • It's well-documented and widely benchmarked.
    • ~400 MB — manageable on HF Spaces free tier.
    • BART's bidirectional encoder gives it good long-text understanding.

    ALTERNATIVES you should be able to name:
    • "cross-encoder/nli-deberta-v3-small" — smaller (180 MB), often
      higher accuracy on NLI tasks, good if memory is tight.
    • A supervised classifier fine-tuned on CUAD (Contract Understanding
      Atticus Dataset — 13,000 labelled contract clauses, freely available)
      would give higher F1 on real contracts. That would be the production
      upgrade path.

LIMITATIONS:
    • ZSC confidence scores are calibrated to the NLI task, not directly
      to "probability of being risky". A score of 0.72 does not mean 72%
      chance the clause is risky — it means the NLI model is 72% confident
      the text entails the label description. Use as a relative signal.
    • The model can confuse structurally similar clauses (e.g., a
      confidentiality clause that also mentions penalties may score high
      on both categories).
    • Speed: BART-large does ~1–3 clauses/second on CPU. With 30–50 clauses
      in a typical NDA, expect 20–60 seconds. GPU or a smaller model
      (deberta-v3-small) cuts this dramatically.
    • The model is English-only. Multi-lingual NDAs would need
      "joeddav/xlm-roberta-large-xnli" or translation pre-processing.

CACHING:
    The @st.cache_resource decorator (used in app.py via get_classifier())
    ensures the ~400 MB model is downloaded once per Spaces instance and
    kept in memory across Streamlit reruns. Without this, every file upload
    would re-download the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import streamlit as st
from transformers import pipeline


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ClassifiedClause:
    """
    Represents one extracted clause with its classification result.

    Attributes
    ----------
    text : str
        The raw clause text.
    category : str
        Predicted risk category (top-scoring label).
    confidence : float
        ZSC confidence score for the top category (0–1).
    all_scores : Dict[str, float]
        Scores for all six categories, useful for secondary analysis.
    """
    text: str
    category: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


# ── Risk category label descriptions ───────────────────────────────────────────
# These are the "hypotheses" fed to the NLI model.
# Writing them as short, specific sentences (rather than single words)
# significantly improves ZSC accuracy because the model was trained on
# full-sentence entailment pairs.

RISK_LABELS: Dict[str, str] = {
    "liability":        "This clause limits or caps the liability of one party",
    "indemnification":  "This clause requires one party to indemnify or hold harmless the other",
    "termination":      "This clause defines conditions under which the agreement can be terminated",
    "penalty":          "This clause imposes financial penalties or liquidated damages",
    "exclusivity":      "This clause grants exclusive rights or restricts dealings with third parties",
    "confidentiality":  "This clause imposes confidentiality obligations or defines confidential information",
}

_LABEL_DESCRIPTIONS = list(RISK_LABELS.values())
_LABEL_KEYS = list(RISK_LABELS.keys())


# ── Model loader (cached) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_classifier():
    """
    Download and cache the ZSC pipeline.

    INTERVIEW NOTE:
        We use @st.cache_resource (not @st.cache_data) because the pipeline
        object holds large model weights and is not serialisable by value.
        cache_resource stores the object in memory and shares it across all
        Streamlit sessions on the same server instance.
    """
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        # device=0 would use GPU if available; defaults to CPU on HF Spaces free tier
    )


# ── Public API ──────────────────────────────────────────────────────────────────

def classify_clauses(clauses: List[str]) -> List[ClassifiedClause]:
    """
    Classify each clause and return a list of ClassifiedClause objects.

    Parameters
    ----------
    clauses : List[str]
        Plain text clauses from pdf_extractor.

    Returns
    -------
    List[ClassifiedClause]
        Same order as input. Each has .category, .confidence, .all_scores.
    """
    classifier = _load_classifier()
    results: List[ClassifiedClause] = []

    for clause_text in clauses:
        output = classifier(
            sequences=clause_text,
            candidate_labels=_LABEL_DESCRIPTIONS,
            multi_label=False,   # softmax — scores sum to 1
        )

        # Map description strings back to short category keys
        top_idx = 0  # pipeline returns sorted descending by score
        top_label_desc = output["labels"][top_idx]
        top_score = output["scores"][top_idx]
        top_key = _description_to_key(top_label_desc)

        all_scores = {
            _description_to_key(desc): score
            for desc, score in zip(output["labels"], output["scores"])
        }

        results.append(
            ClassifiedClause(
                text=clause_text,
                category=top_key,
                confidence=round(top_score, 4),
                all_scores=all_scores,
            )
        )

    return results


# ── Internal helpers ────────────────────────────────────────────────────────────

def _description_to_key(description: str) -> str:
    """Reverse-lookup: description string → short category key."""
    for key, desc in RISK_LABELS.items():
        if desc == description:
            return key
    # Fallback — should never happen if RISK_LABELS is consistent
    return "unknown"
