"""
Confidence estimation for classification models.

Confidence is defined as the maximum predicted probability
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

from aegis.core.exceptions import ConfidenceError, ValidationError
from aegis.core.types import ProbabilityVector
from aegis.uncertainty.base import BaseUncertaintyEstimator


class ConfidenceEstimator(BaseUncertaintyEstimator):
    """
    Estimate classification confidence using maximum probability.

    For each sample:

        confidence = max(P(class | x))

    Higher values indicate that the model assigns more probability
    mass to its most likely class.

    Note:
        This is predictive confidence, not calibrated confidence.
        Calibration is a separate statistical procedure.
    """

    @property
    def name(self) -> str:
        """Return the estimator name."""
        return "Maximum Probability Confidence"

    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute confidence for each sample.

        Args:
            probabilities:
                Probability matrix with shape
                ``(n_samples, n_classes)``.

        Returns:
            One confidence value in ``[0, 1]`` per sample.

        Raises:
            ValidationError:
                If the probability matrix is invalid.
            ConfidenceError:
                If confidence calculation fails unexpectedly.
        """
        try:
            values = self.validate(probabilities)

            confidence = np.max(values, axis=1)

            if not np.all(np.isfinite(confidence)):
                raise ConfidenceError(
                    "Confidence calculation produced "
                    "NaN or infinite values."
                )

            if np.any(confidence < 0.0) or np.any(confidence > 1.0):
                raise ConfidenceError(
                    "Confidence values must be within [0, 1]."
                )

            return confidence.astype(np.float64, copy=False)

        except ValidationError:
            raise
        except ConfidenceError:
            raise
        except Exception as exc:
            raise ConfidenceError(
                f"Unable to compute confidence: {exc}"
            ) from exc

    def average(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the mean confidence across all samples.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Mean confidence in ``[0, 1]``.
        """
        confidence = self.compute(probabilities)

        return float(np.mean(confidence))

    def minimum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the minimum confidence.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Lowest confidence value.
        """
        confidence = self.compute(probabilities)

        return float(np.min(confidence))

    def maximum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute the maximum confidence.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Highest confidence value.
        """
        confidence = self.compute(probabilities)

        return float(np.max(confidence))

    def statistics(
        self,
        probabilities: ProbabilityVector,
    ) -> dict[str, float]:
        """
        Compute confidence summary statistics.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Dictionary containing:

            - ``mean``
            - ``std``
            - ``min``
            - ``max``
        """
        confidence = self.compute(probabilities)

        return {
            "mean": float(np.mean(confidence)),
            "std": float(np.std(confidence)),
            "min": float(np.min(confidence)),
            "max": float(np.max(confidence)),
        }