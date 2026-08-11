"""
Score aggregation engine.

Combines multiple reliability signals into a unified score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.scoring.weighting import ScoreWeights


@dataclass
class ScoreInputs:
    confidence: float
    uncertainty: float
    ood: float
    drift: float


class ScoreAggregator:
    """
    Aggregates ML reliability signals into a single score.
    """

    def __init__(self, weights: ScoreWeights | None = None):
        self.weights = weights.normalize() if weights else ScoreWeights().normalize()

    def aggregate(self, inputs: ScoreInputs) -> float:
        """
        Compute weighted trust score.
        """

        # Convert uncertainty, ood, drift into "risk terms"
        uncertainty = 1 - inputs.uncertainty
        ood = 1 - inputs.ood
        drift = 1 - inputs.drift

        score = (
            self.weights.confidence * inputs.confidence
            + self.weights.uncertainty * uncertainty
            + self.weights.ood * ood
            + self.weights.drift * drift
        )

        return float(np.clip(score, 0.0, 1.0))