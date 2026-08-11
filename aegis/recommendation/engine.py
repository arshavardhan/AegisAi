"""
Recommendation engine for AegisAI.

This module converts trust scores into actionable recommendations.

The recommendation policy is intentionally separated from trust score
calculation so organizations can customize decision logic without
modifying the scoring algorithm.
"""

from __future__ import annotations

from aegis.config import settings
from aegis.core.enums import Recommendation, RiskLevel


class RecommendationEngine:
    """
    Generate recommendations from trust scores.

    Future versions may incorporate:
        - Business rules
        - Human approval workflows
        - Regulatory compliance
        - Cost-sensitive decisions
        - Model-specific policies
    """

    def recommend(
        self,
        trust_score: float,
    ) -> Recommendation:
        """
        Generate a recommendation.

        Args:
            trust_score:
                Trust score in the range [0, 1].

        Returns:
            Recommendation enum.
        """
        trust_score = self._clip(trust_score)

        if trust_score >= settings.auto_approve_threshold:
            return Recommendation.AUTO_APPROVE

        if trust_score >= settings.review_threshold:
            return Recommendation.HUMAN_REVIEW

        return Recommendation.REJECT

    def risk_level(
        self,
        trust_score: float,
    ) -> RiskLevel:
        """
        Determine the risk level associated with a trust score.

        Args:
            trust_score:
                Trust score in the range [0, 1].

        Returns:
            RiskLevel enum.
        """
        trust_score = self._clip(trust_score)

        if trust_score >= settings.auto_approve_threshold:
            return RiskLevel.LOW

        if trust_score >= settings.review_threshold:
            return RiskLevel.MEDIUM

        return RiskLevel.HIGH

    def evaluate(
        self,
        trust_score: float,
    ) -> tuple[RiskLevel, Recommendation]:
        """
        Evaluate both risk level and recommendation.

        Args:
            trust_score:
                Trust score.

        Returns:
            Tuple of (RiskLevel, Recommendation).
        """
        return (
            self.risk_level(trust_score),
            self.recommend(trust_score),
        )

    @staticmethod
    def _clip(value: float) -> float:
        """
        Clamp a value to the range [0, 1].
        """
        return max(0.0, min(1.0, float(value)))

    def __call__(
        self,
        trust_score: float,
    ) -> tuple[RiskLevel, Recommendation]:
        """
        Callable interface.
        """
        return self.evaluate(trust_score)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"approve={settings.auto_approve_threshold}, "
            f"review={settings.review_threshold})"
        )