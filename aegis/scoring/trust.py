"""
Trust scoring engine (final layer of AegisAI core logic).

This module converts all ML reliability signals into:
    - trust_score
    - risk_level
    - recommendation
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.scoring.aggregator import ScoreAggregator, ScoreInputs


@dataclass
class TrustResult:
    trust_score: float
    risk_level: str
    recommendation: str


class TrustScorer:
    """
    Converts aggregated score into business-level decisions.
    """

    def __init__(self, aggregator: ScoreAggregator):
        self.aggregator = aggregator

    def evaluate(self, inputs: ScoreInputs) -> TrustResult:
        score = self.aggregator.aggregate(inputs)

        # Risk classification
        if score >= 0.85:
            risk = "LOW"
            recommendation = "Auto Approve"
        elif score >= 0.65:
            risk = "MEDIUM"
            recommendation = "Human Review Recommended"
        else:
            risk = "HIGH"
            recommendation = "Reject / Escalate"

        return TrustResult(
            trust_score=score,
            risk_level=risk,
            recommendation=recommendation,
        )