"""
Weighting system for AegisAI trust scoring.

Defines configurable importance weights for:
- confidence
- uncertainty
- OOD detection
- drift detection
"""

from dataclasses import dataclass


@dataclass
class ScoreWeights:
    """
    Configurable weights for trust score computation.
    """

    confidence: float = 0.35
    uncertainty: float = 0.25
    ood: float = 0.20
    drift: float = 0.20

    def normalize(self) -> "ScoreWeights":
        """
        Normalize weights so they sum to 1.
        """
        total = (
            self.confidence
            + self.uncertainty
            + self.ood
            + self.drift
        )

        return ScoreWeights(
            confidence=self.confidence / total,
            uncertainty=self.uncertainty / total,
            ood=self.ood / total,
            drift=self.drift / total,
        )