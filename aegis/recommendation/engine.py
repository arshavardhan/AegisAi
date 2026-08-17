"""
Recommendation engine for AegisAI.

This module converts validated trust scores into actionable
risk levels and recommendations.

Recommendation policy is intentionally separated from trust
score calculation so organizations can customize decision logic
without modifying the scoring algorithm.
"""

from __future__ import annotations

import math

from aegis.config import settings
from aegis.core.enums import Recommendation, RiskLevel
from aegis.core.exceptions import ValidationError


class RecommendationEngine:
    """
    Generate risk levels and recommendations from trust scores.

    The configured thresholds define the decision policy:

        trust >= auto_approve_threshold
            -> LOW risk
            -> AUTO_APPROVE

        trust >= review_threshold
            -> MEDIUM risk
            -> HUMAN_REVIEW

        trust < review_threshold
            -> HIGH risk
            -> REJECT
    """

    def __init__(
        self,
        auto_approve_threshold: float | None = None,
        review_threshold: float | None = None,
    ) -> None:
        """Initialize the recommendation policy.

        Args:
            auto_approve_threshold:
                Minimum trust score for automatic approval.
                Defaults to the configured value.

            review_threshold:
                Minimum trust score for human review.
                Defaults to the configured value.

        Raises:
            ValidationError:
                If thresholds are invalid or inconsistent.
        """
        self.auto_approve_threshold = self._validate_threshold(
            (
                settings.auto_approve_threshold
                if auto_approve_threshold is None
                else auto_approve_threshold
            ),
            "auto_approve_threshold",
        )

        self.review_threshold = self._validate_threshold(
            (
                settings.review_threshold
                if review_threshold is None
                else review_threshold
            ),
            "review_threshold",
        )

        if self.review_threshold >= self.auto_approve_threshold:
            raise ValidationError(
                "review_threshold must be lower than "
                "auto_approve_threshold."
            )

    def recommend(
        self,
        trust_score: float,
    ) -> Recommendation:
        """
        Generate a recommendation from a trust score.

        Args:
            trust_score:
                Trust score in the range [0, 1].

        Returns:
            Recommendation enum.
        """
        score = self._validate_trust_score(trust_score)

        if score >= self.auto_approve_threshold:
            return Recommendation.AUTO_APPROVE

        if score >= self.review_threshold:
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
        score = self._validate_trust_score(trust_score)

        if score >= self.auto_approve_threshold:
            return RiskLevel.LOW

        if score >= self.review_threshold:
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
                Trust score in the range [0, 1].

        Returns:
            Tuple containing:
                - RiskLevel
                - Recommendation
        """
        score = self._validate_trust_score(trust_score)

        if score >= self.auto_approve_threshold:
            return (
                RiskLevel.LOW,
                Recommendation.AUTO_APPROVE,
            )

        if score >= self.review_threshold:
            return (
                RiskLevel.MEDIUM,
                Recommendation.HUMAN_REVIEW,
            )

        return (
            RiskLevel.HIGH,
            Recommendation.REJECT,
        )

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        """Validate a recommendation threshold."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(numeric_value):
            raise ValidationError(
                f"{name} must be finite."
            )

        if not 0.0 <= numeric_value <= 1.0:
            raise ValidationError(
                f"{name} must be within [0, 1]."
            )

        return numeric_value

    @staticmethod
    def _validate_trust_score(
        value: float,
    ) -> float:
        """Validate a trust score."""
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Trust score must be numeric."
            ) from exc

        if not math.isfinite(score):
            raise ValidationError(
                "Trust score must be finite."
            )

        if not 0.0 <= score <= 1.0:
            raise ValidationError(
                "Trust score must be within [0, 1]."
            )

        return score

    def __call__(
        self,
        trust_score: float,
    ) -> tuple[RiskLevel, Recommendation]:
        """Evaluate a trust score."""
        return self.evaluate(trust_score)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"approve={self.auto_approve_threshold}, "
            f"review={self.review_threshold})"
        )