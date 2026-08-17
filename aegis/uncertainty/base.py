"""
Base classes for uncertainty estimation.

All uncertainty estimation algorithms must inherit from
BaseUncertaintyEstimator.

Examples:
    - Confidence Estimation
    - Entropy
    - Monte Carlo Dropout
    - Deep Ensembles
    - Bayesian Neural Networks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aegis.core.exceptions import ValidationError
from aegis.core.types import ProbabilityVector


class BaseUncertaintyEstimator(ABC):
    """
    Abstract base class for uncertainty estimators.

    Every uncertainty estimation algorithm must implement
    the ``compute()`` method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable estimator name."""

    @abstractmethod
    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute one uncertainty value per prediction.

        Args:
            probabilities:
                Probability matrix with shape
                ``(n_samples, n_classes)``.

        Returns:
            One uncertainty value per sample.

        Raises:
            ValidationError:
                If the probability matrix is invalid.
        """

    def validate(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Validate and normalize a probability matrix.

        Args:
            probabilities:
                Model probability predictions.

        Returns:
            A validated NumPy floating-point array.

        Raises:
            ValidationError:
                If the probability matrix is invalid.
        """
        try:
            values = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Probabilities must be numeric."
            ) from exc

        if values.ndim != 2:
            raise ValidationError(
                "Probability matrix must have shape "
                "(n_samples, n_classes)."
            )

        n_samples, n_classes = values.shape

        if n_samples == 0:
            raise ValidationError(
                "Probability matrix cannot contain zero samples."
            )

        if n_classes < 2:
            raise ValidationError(
                "At least two classes are required."
            )

        if not np.all(np.isfinite(values)):
            raise ValidationError(
                "Probabilities cannot contain NaN or infinite values."
            )

        if np.any(values < 0.0):
            raise ValidationError(
                "Probabilities cannot be negative."
            )

        if np.any(values > 1.0):
            raise ValidationError(
                "Probabilities must be within [0, 1]."
            )

        row_sums = values.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValidationError(
                "Each probability row must sum to 1."
            )

        return values

    def __call__(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Allow estimators to be called like functions.

        Example:
            estimator(probabilities)
        """
        validated = self.validate(probabilities)
        return self.compute(validated)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}')"
        )