"""
Confidence estimation for classification models.

The confidence score is defined as the maximum predicted probability
for each sample.

Example:
    Probabilities:
        [[0.10, 0.90],
         [0.60, 0.40]]

    Confidence:
        [0.90, 0.60]
"""

from __future__ import annotations

import numpy as np

from aegis.core.exceptions import ConfidenceError
from aegis.core.types import ProbabilityVector
from aegis.uncertainty.base import BaseUncertaintyEstimator


class ConfidenceEstimator(BaseUncertaintyEstimator):
    """
    Estimate prediction confidence using the maximum predicted
    probability.

    Higher confidence indicates the model is more certain about
    its prediction.
    """

    @property
    def name(self) -> str:
        """
        Name of the estimator.
        """
        return "Maximum Probability Confidence"

    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute confidence for each sample.

        Args:
            probabilities:
                Probability matrix of shape
                (n_samples, n_classes).

        Returns:
            NumPy array of confidence scores.
        """
        try:
            probabilities = np.asarray(probabilities, dtype=np.float64)

            self.validate(probabilities)

            confidence = np.max(probabilities, axis=1)

            return confidence

        except Exception as exc:
            raise ConfidenceError(
                f"Unable to compute confidence: {exc}"
            ) from exc

    def average(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the average confidence across all samples.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Mean confidence score.
        """
        confidence = self.compute(probabilities)
        return float(np.mean(confidence))

    def minimum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the minimum confidence.

        Returns:
            Lowest confidence score.
        """
        confidence = self.compute(probabilities)
        return float(np.min(confidence))

    def maximum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the maximum confidence.

        Returns:
            Highest confidence score.
        """
        confidence = self.compute(probabilities)
        return float(np.max(confidence))

    def statistics(
        self,
        probabilities: ProbabilityVector,
    ) -> dict[str, float]:
        """
        Compute summary statistics.

        Returns:
            Dictionary containing confidence statistics.
        """
        confidence = self.compute(probabilities)

        return {
            "mean": float(np.mean(confidence)),
            "std": float(np.std(confidence)),
            "min": float(np.min(confidence)),
            "max": float(np.max(confidence)),
        }