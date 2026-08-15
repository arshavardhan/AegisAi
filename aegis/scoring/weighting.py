"""Configurable weights for AegisAI trust scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for normalized reliability signals.

    Drift is intentionally disabled in v0.1 because the prediction
    pipeline does not yet supply a live drift signal. It will be enabled
    when monitoring is integrated into the scoring pipeline.
    """

    confidence: float = 0.50
    uncertainty: float = 0.30
    ood: float = 0.20
    drift: float = 0.0

    def normalize(self) -> "ScoreWeights":
        """Return weights normalized to sum to one."""
        values = {
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "ood": float(self.ood),
            "drift": float(self.drift),
        }

        if any(value < 0.0 for value in values.values()):
            raise ValueError("Score weights cannot be negative.")

        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("At least one score weight must be greater than zero.")

        return ScoreWeights(
            confidence=values["confidence"] / total,
            uncertainty=values["uncertainty"] / total,
            ood=values["ood"] / total,
            drift=values["drift"] / total,
        )
