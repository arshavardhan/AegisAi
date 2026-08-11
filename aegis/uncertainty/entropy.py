"""
Entropy-based uncertainty estimation.

Entropy measures the uncertainty of a probability distribution.

Lower entropy indicates higher certainty.
Higher entropy indicates greater uncertainty.

Mathematically:

    H(p) = -Σ p * log2(p)

The entropy is normalized to the range [0, 1], making it comparable
across classification problems with different numbers of classes.
"""

from __future__ import annotations

import numpy as np

from aegis.core.exceptions import EntropyError
from aegis.core.types import ProbabilityVector
from aegis.uncertainty.base import BaseUncertaintyEstimator


class EntropyEstimator(BaseUncertaintyEstimator):
    """
    Estimate predictive uncertainty using Shannon entropy.
    """

    @property
    def name(self) -> str:
        return "Shannon Entropy"

    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute normalized Shannon entropy for each prediction.

        Args:
            probabilities:
                Probability matrix of shape
                (n_samples, n_classes).

        Returns:
            Array containing one entropy value per sample.
            Values are normalized to the range [0, 1].
        """
        try:
            probabilities = np.asarray(
                probabilities,
                dtype=np.float64,
            )

            self.validate(probabilities)

            # Prevent log2(0)
            eps = np.finfo(np.float64).eps
            probs = np.clip(probabilities, eps, 1.0)

            entropy = -np.sum(
                probs * np.log2(probs),
                axis=1,
            )

            max_entropy = np.log2(probs.shape[1])

            normalized_entropy = entropy / max_entropy

            return normalized_entropy

        except Exception as exc:
            raise EntropyError(
                f"Unable to compute entropy: {exc}"
            ) from exc

    def average(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute average entropy.
        """
        return float(np.mean(self.compute(probabilities)))

    def minimum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute minimum entropy.
        """
        return float(np.min(self.compute(probabilities)))

    def maximum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute maximum entropy.
        """
        return float(np.max(self.compute(probabilities)))

    def statistics(
        self,
        probabilities: ProbabilityVector,
    ) -> dict[str, float]:
        """
        Return summary statistics for entropy.
        """
        entropy = self.compute(probabilities)

        return {
            "mean": float(np.mean(entropy)),
            "std": float(np.std(entropy)),
            "min": float(np.min(entropy)),
            "max": float(np.max(entropy)),
        }