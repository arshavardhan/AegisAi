"""
Score aggregation engine.

Combines normalized reliability signals into a unified trust score.

Signal semantics:

    confidence:
        Higher is better.

    uncertainty:
        Higher is worse.

    ood_risk:
        Higher is worse.

    drift:
        Higher is worse.

The aggregator itself is intentionally deterministic and does not
perform model-specific calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.core.exceptions import ValidationError
from aegis.scoring.weighting import ScoreWeights


@dataclass(frozen=True)
class ScoreInputs:
    """Normalized reliability signals used by the score aggregator.

    All values must be in the range [0, 1].

    Attributes:
        confidence:
            Model predictive confidence.
            Higher values indicate stronger confidence.

        uncertainty:
            Predictive uncertainty.
            Higher values indicate greater uncertainty.

        ood_risk:
            Normalized out-of-distribution risk.
            Higher values indicate greater OOD risk.

        drift:
            Normalized distribution drift.
            Higher values indicate greater drift.

    Note:
        Drift is supported by the scoring infrastructure but should
        only contribute when a real drift measurement is supplied.
    """

    confidence: float
    uncertainty: float
    ood_risk: float
    drift: float = 0.0


class ScoreAggregator:
    """Aggregate normalized reliability signals into a trust score."""

    def __init__(
        self,
        weights: ScoreWeights | None = None,
    ) -> None:
        """Initialize the aggregator.

        Args:
            weights:
                Optional score weights. If omitted, the configured
                defaults are used and normalized.
        """
        self.weights = (
            weights.normalize()
            if weights is not None
            else ScoreWeights().normalize()
        )

    def aggregate(
        self,
        inputs: ScoreInputs,
    ) -> float:
        """Compute a deterministic trust score.

        Confidence contributes positively.

        Uncertainty, OOD risk, and drift are converted into
        reliability terms using:

            reliability = 1 - risk

        Args:
            inputs:
                Normalized reliability signals.

        Returns:
            Trust score in the range [0, 1].

        Raises:
            ValidationError:
                If any input is outside [0, 1], NaN, or infinite.
        """
        self._validate_inputs(inputs)

        confidence = inputs.confidence
        uncertainty_reliability = 1.0 - inputs.uncertainty
        ood_reliability = 1.0 - inputs.ood_risk
        drift_reliability = 1.0 - inputs.drift

        score = (
            self.weights.confidence * confidence
            + self.weights.uncertainty * uncertainty_reliability
            + self.weights.ood * ood_reliability
            + self.weights.drift * drift_reliability
        )

        if not np.isfinite(score):
            raise ValidationError(
                "Trust-score aggregation produced a non-finite value."
            )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _validate_inputs(
        inputs: ScoreInputs,
    ) -> None:
        """Validate all score inputs."""
        values = {
            "confidence": inputs.confidence,
            "uncertainty": inputs.uncertainty,
            "ood_risk": inputs.ood_risk,
            "drift": inputs.drift,
        }

        for name, value in values.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{name} must be numeric."
                ) from exc

            if not np.isfinite(numeric_value):
                raise ValidationError(
                    f"{name} must be finite."
                )

            if not 0.0 <= numeric_value <= 1.0:
                raise ValidationError(
                    f"{name} must be within [0, 1]."
                )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"weights={self.weights})"
        )