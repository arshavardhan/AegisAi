"""
Entropy-based uncertainty estimation.

Entropy measures the uncertainty of a probability distribution.

Lower entropy indicates higher certainty.
Higher entropy indicates greater uncertainty.

For a probability distribution p:

    H(p) = -Σ p * log2(p)

The entropy is normalized by the maximum possible entropy for
the number of classes, producing values in the range [0, 1].
"""

from __future__ import annotations

import numpy as np

from aegis.core.exceptions import EntropyError, ValidationError
from aegis.core.types import ProbabilityVector
from aegis.uncertainty.base import BaseUncertaintyEstimator


class EntropyEstimator(BaseUncertaintyEstimator):
    """Estimate predictive uncertainty using normalized Shannon entropy."""

    @property
    def name(self) -> str:
        """Return the estimator name."""
        return "Normalized Shannon Entropy"

    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute normalized Shannon entropy for each prediction.

        Args:
            probabilities:
                Probability matrix with shape
                ``(n_samples, n_classes)``.

        Returns:
            One normalized entropy value in ``[0, 1]`` per sample.

        Raises:
            ValidationError:
                If the probability matrix is invalid.
            EntropyError:
                If entropy calculation fails unexpectedly.
        """
        try:
            values = self.validate(probabilities)

            # Shannon entropy:
            #
            # H(p) = -sum(p * log2(p))
            #
            # For p = 0, the mathematical limit is:
            #
            # 0 * log2(0) = 0
            #
            # Therefore zero probabilities are excluded from the
            # logarithm instead of being artificially clipped.

            positive = values > 0.0

            entropy = -np.sum(
                np.where(
                    positive,
                    values * np.log2(
                        np.where(
                            positive,
                            values,
                            1.0,
                        )
                    ),
                    0.0,
                ),
                axis=1,
            )

            n_classes = values.shape[1]
            max_entropy = np.log2(n_classes)

            if not np.isfinite(max_entropy) or max_entropy <= 0.0:
                raise EntropyError(
                    "Unable to determine maximum entropy."
                )

            normalized_entropy = entropy / max_entropy

            if not np.all(np.isfinite(normalized_entropy)):
                raise EntropyError(
                    "Entropy calculation produced "
                    "NaN or infinite values."
                )

            # Numerical floating-point operations can produce tiny
            # values outside the theoretical [0, 1] range.
            normalized_entropy = np.clip(
                normalized_entropy,
                0.0,
                1.0,
            )

            return normalized_entropy.astype(
                np.float64,
                copy=False,
            )

        except ValidationError:
            raise
        except EntropyError:
            raise
        except Exception as exc:
            raise EntropyError(
                f"Unable to compute entropy: {exc}"
            ) from exc

    def average(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute average normalized entropy.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Mean entropy in ``[0, 1]``.
        """
        entropy = self.compute(probabilities)

        return float(np.mean(entropy))

    def minimum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute minimum normalized entropy.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Lowest entropy value.
        """
        entropy = self.compute(probabilities)

        return float(np.min(entropy))

    def maximum(
        self,
        probabilities: ProbabilityVector,
    ) -> float:
        """
        Compute maximum normalized entropy.

        Args:
            probabilities:
                Probability matrix.

        Returns:
            Highest entropy value.
        """
        entropy = self.compute(probabilities)

        return float(np.max(entropy))

    def statistics(
        self,
        probabilities: ProbabilityVector,
    ) -> dict[str, float]:
        """
        Compute summary statistics for normalized entropy.

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
        entropy = self.compute(probabilities)

        return {
            "mean": float(np.mean(entropy)),
            "std": float(np.std(entropy)),
            "min": float(np.min(entropy)),
            "max": float(np.max(entropy)),
        }