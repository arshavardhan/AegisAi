"""
Trust scoring engine for AegisAI.

The trust scorer combines normalized reliability signals into a trust
score and delegates the final policy decision to RecommendationEngine.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.core.enums import Recommendation, RiskLevel
from aegis.core.exceptions import ValidationError
from aegis.recommendation.engine import RecommendationEngine
from aegis.scoring.aggregator import ScoreAggregator, ScoreInputs


@dataclass(frozen=True)
class TrustResult:
    """Final trust decision produced by AegisAI."""

    trust_score: float
    risk_level: RiskLevel
    recommendation: Recommendation


class TrustScorer:
    """Convert reliability signals into a trust decision."""

    def __init__(
        self,
        aggregator: ScoreAggregator | None = None,
        recommendation_engine: RecommendationEngine | None = None,
    ) -> None:
        """Initialize the trust scorer.

        Args:
            aggregator:
                Score aggregation engine.

            recommendation_engine:
                Policy engine responsible for converting trust scores
                into risk levels and recommendations.
        """
        self.aggregator = (
            aggregator
            if aggregator is not None
            else ScoreAggregator()
        )

        self.recommendation_engine = (
            recommendation_engine
            if recommendation_engine is not None
            else RecommendationEngine()
        )

    def evaluate(
        self,
        inputs: ScoreInputs,
    ) -> TrustResult:
        """Aggregate reliability signals and classify the result.

        Args:
            inputs:
                Normalized reliability signals.

        Returns:
            Final trust result.
        """
        score = self.aggregator.aggregate(inputs)

        risk_level, recommendation = (
            self.recommendation_engine.evaluate(score)
        )

        return TrustResult(
            trust_score=score,
            risk_level=risk_level,
            recommendation=recommendation,
        )

    def compute(
        self,
        *,
        confidence: float,
        entropy: float,
        ood_score: float,
        is_ood: bool,
        drift_score: float | None = None,
    ) -> TrustResult:
        """Compute a trust decision from reliability signals.

        Args:
            confidence:
                Maximum predicted probability in [0, 1].

            entropy:
                Normalized predictive entropy in [0, 1].

            ood_score:
                Normalized OOD risk in [0, 1].

            is_ood:
                Whether the OOD detector classified the sample
                as out-of-distribution.

            drift_score:
                Optional normalized drift risk in [0, 1].
                ``None`` means drift has not been measured.

        Returns:
            Final trust result.

        Raises:
            ValidationError:
                If an input cannot be converted to a numeric value.
        """
        try:
            normalized_confidence = float(confidence)
            normalized_entropy = float(entropy)
            normalized_ood = float(ood_score)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Trust-score inputs must be numeric."
            ) from exc

        # OOD classification is a hard risk signal.
        if is_ood:
            normalized_ood = 1.0

        normalized_drift = (
            None
            if drift_score is None
            else float(drift_score)
        )

        inputs = ScoreInputs(
            confidence=normalized_confidence,
            uncertainty=normalized_entropy,
            ood_risk=normalized_ood,
            drift=(
                0.0
                if normalized_drift is None
                else normalized_drift
            ),
        )

        return self.evaluate(inputs)

    def __call__(
        self,
        inputs: ScoreInputs,
    ) -> TrustResult:
        """Evaluate normalized reliability signals."""
        return self.evaluate(inputs)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"aggregator={self.aggregator!r}, "
            f"recommendation_engine="
            f"{self.recommendation_engine!r})"
        )