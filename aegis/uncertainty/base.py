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

import numpy as np

from aegis.core.types import ProbabilityVector


class BaseUncertaintyEstimator(ABC):
    """
    Abstract base class for uncertainty estimators.

    Every uncertainty estimation algorithm should implement the
    `compute()` method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable estimator name.
        """

    @abstractmethod
    def compute(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Compute uncertainty for each prediction.

        Args:
            probabilities:
                Probability matrix of shape
                (n_samples, n_classes).

        Returns:
            A NumPy array containing one uncertainty value per sample.
        """

    def validate(
        self,
        probabilities: ProbabilityVector,
    ) -> None:
        """
        Validate the probability matrix.

        Args:
            probabilities:
                Model probability predictions.

        Raises:
            ValueError:
                If the probability matrix is invalid.
        """
        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 2:
            raise ValueError(
                "Probability matrix must have shape "
                "(n_samples, n_classes)."
            )

        if probabilities.shape[1] < 2:
            raise ValueError(
                "At least two classes are required."
            )

        if np.any(probabilities < 0):
            raise ValueError(
                "Probabilities cannot be negative."
            )

        row_sums = probabilities.sum(axis=1)

        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                "Each probability row must sum to 1."
            )

    def __call__(
        self,
        probabilities: ProbabilityVector,
    ) -> np.ndarray:
        """
        Allow estimators to be called like functions.

        Example:
            estimator(probabilities)
        """
        self.validate(probabilities)
        return self.compute(probabilities)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"