"""
Base classes for Out-of-Distribution (OOD) detection.

OOD detection determines whether an input differs significantly from
the data distribution on which the model was trained.

All OOD detectors must inherit from BaseOODDetector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aegis.core.types import ModelInput


class BaseOODDetector(ABC):
    """
    Abstract base class for all Out-of-Distribution detectors.

    Each detector should be trained (fit) on reference data and then
    used to evaluate new samples.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable detector name.
        """

    @abstractmethod
    def fit(self, X: ModelInput) -> "BaseOODDetector":
        """
        Fit the detector using reference (training) data.

        Args:
            X:
                Reference feature matrix.

        Returns:
            The fitted detector.
        """

    @abstractmethod
    def score(self, X: ModelInput) -> np.ndarray:
        """
        Compute an OOD score for each sample.

        Higher scores indicate the sample is more likely to be
        out-of-distribution.

        Args:
            X:
                Input samples.

        Returns:
            One score per sample.
        """

    @abstractmethod
    def predict(self, X: ModelInput) -> np.ndarray:
        """
        Predict whether each sample is OOD.

        Returns:
            Boolean NumPy array.

            True  -> Out-of-Distribution

            False -> In-Distribution
        """

    def fit_predict(self, X: ModelInput) -> np.ndarray:
        """
        Convenience method.

        Fits the detector and predicts OOD labels for the same data.

        Args:
            X:
                Feature matrix.

        Returns:
            Boolean predictions.
        """
        self.fit(X)
        return self.predict(X)

    def validate_input(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Convert input into a NumPy array and validate shape.

        Args:
            X:
                Input samples.

        Returns:
            Two-dimensional NumPy array.

        Raises:
            ValueError:
                If the input has an invalid shape.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "Expected input of shape "
                "(n_samples, n_features)."
            )

        return X

    def __call__(self, X: ModelInput) -> np.ndarray:
        """
        Shortcut for predict().
        """
        return self.predict(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"