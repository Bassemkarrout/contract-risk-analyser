"""
pipeline/risk_aggregator.py - aggregates clause scores into an overall risk rating.

Scoring approach: worst-clause aggregation.
For each risk category, we take the single highest confidence score found
across all clauses in that category (not an average). This means one
genuinely dangerous clause always surfaces in the overall score regardless
of how many benign clauses surround it - which is how a lawyer would
actually assess a contract in real life.

Example: a contract with 6 standard clauses and one liability clause
capped at $100 should score High on liability, not Low just because the
other 6 clauses are fine. The old averaging approach was masking real risk.

Thresholds: Low <40, Medium 40-70, High >=70.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pipeline.classifier import ClassifiedClause


@dataclass
class RiskSummary:
    score: float
    rating: str
    category_scores: Dict[str, float]
    clause_count: int
    flagged_count: int


_LOW_THRESHOLD = 40
_HIGH_THRESHOLD = 70


def aggregate_risk(
    clauses: List[ClassifiedClause],
    weights: Dict[str, float],
    flag_threshold: float = 0.65,
) -> RiskSummary:
    if not clauses:
        return RiskSummary(
            score=0.0,
            rating="Low",
            category_scores={k: 0.0 for k in weights},
            clause_count=0,
            flagged_count=0,
        )

    # For each category, find the single highest confidence score (worst-clause)
    # instead of averaging all clauses together. This ensures one dangerous
    # clause always pushes the score up, matching how lawyers assess contracts.
    category_peaks: Dict[str, float] = {k: 0.0 for k in weights}
    for clause in clauses:
        cat = clause.category
        if cat in category_peaks:
            if clause.confidence > category_peaks[cat]:
                category_peaks[cat] = clause.confidence

    # Convert peak confidence (0-1) to a 0-100 score per category
    category_scores: Dict[str, float] = {
        cat: round(peak * 100, 1)
        for cat, peak in category_peaks.items()
    }

    # Overall score: weighted sum of category peak scores
    overall = sum(
        category_scores[cat] * weights.get(cat, 0)
        for cat in category_scores
    )
    overall = round(min(overall, 100.0), 1)

    if overall < _LOW_THRESHOLD:
        rating = "Low"
    elif overall < _HIGH_THRESHOLD:
        rating = "Medium"
    else:
        rating = "High"

    flagged_count = sum(1 for c in clauses if c.confidence >= flag_threshold)

    return RiskSummary(
        score=overall,
        rating=rating,
        category_scores=category_scores,
        clause_count=len(clauses),
        flagged_count=flagged_count,
    )
