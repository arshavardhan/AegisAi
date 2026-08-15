"""
Trust scoring engine for AegisAI.

The trust scorer is intentionally small and deterministic: it accepts
already-normalized reliability signals, delegates aggregation to
``ScoreAggregator``, and converts the resulting score into a risk level
and recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.core.enums import Recommendation, RiskLevel
from aegis.scoring.aggregator import ScoreAggregator, ScoreInputs


@dataclass(frozen=True)
class TrustResult:
    """Final trust decision produced by AegisAI."""

    trust_score: float
    risk_level: RiskLevel
    recommendation: Recommendation


class TrustScorer:
    """Convert reliability signals into a trust decision."""

    def __init__(self, aggregator: ScoreAggregator | None = None) -> None:
        self.aggregator = aggregator or ScoreAggregator()

    def evaluate(self, inputs: ScoreInputs) -> TrustResult:
        """Aggregate reliability signals and classify the result."""
        score = self.aggregator.aggregate(inputs)

        if score >= 0.85:
            risk = RiskLevel.LOW
            recommendation = Recommendation.AUTO_APPROVE
        elif score >= 0.60:
            risk = RiskLevel.MEDIUM
            recommendation = Recommendation.HUMAN_REVIEW
        else:
            risk = RiskLevel.HIGH
            recommendation = Recommendation.REJECT

        return TrustResult(
            trust_score=score,
            risk_level=risk,
            recommendation=recommendation,
        )

    def compute(
        self,
        *,
        confidence: float,
        entropy: float,
        ood_score: float,
        is_ood: bool,
        drift_score: float = 0.0,
    ) -> TrustResult:
        """Convenience API for pipeline callers.

        ``ood_score`` must already be normalized to ``[0, 1]``. The OOD
        boolean is treated as a hard risk signal when it is true.
        """
        normalized_ood = 1.0 if is_ood else float(ood_score)
        inputs = ScoreInputs(
            confidence=float(confidence),
            uncertainty=float(entropy),
            ood=normalized_ood,
            drift=float(drift_score),
        )
        return self.evaluate(inputs)

    def __call__(self, inputs: ScoreInputs) -> TrustResult:
        return self.evaluate(inputs)
