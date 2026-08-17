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

from aegis.core.exceptions import OODDetectionError
from aegis.core.types import ModelInput


class BaseOODDetector(ABC):
    """
    Abstract base class for Out-of-Distribution detectors.

    Each detector is fitted on reference data and then used to
    evaluate new samples against that reference distribution.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable detector name."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return whether the detector has been fitted."""

    @abstractmethod
    def fit(self, X: ModelInput) -> "BaseOODDetector":
        """
        Fit the detector using reference data.

        Args:
            X:
                Reference feature matrix.

        Returns:
            The fitted detector.

        Raises:
            OODDetectionError:
                If the reference data is invalid.
        """

    @abstractmethod
    def score(self, X: ModelInput) -> np.ndarray:
        """
        Compute a raw OOD score for each sample.

        Higher scores indicate greater distance from the
        reference distribution.

        Args:
            X:
                Input samples.

        Returns:
            One raw OOD score per sample.

        Raises:
            OODDetectionError:
                If the detector is not fitted or the input is invalid.
        """

    @abstractmethod
    def predict(self, X: ModelInput) -> np.ndarray:
        """
        Predict whether each sample is OOD.

        Returns:
            Boolean NumPy array.

            True:
                Out-of-Distribution.

            False:
                In-Distribution.
        """

    def fit_predict(self, X: ModelInput) -> np.ndarray:
        """
        Fit the detector and predict OOD labels for the same data.

        Args:
            X:
                Reference feature matrix.

        Returns:
            Boolean OOD predictions.
        """
        self.fit(X)
        return self.predict(X)

    def validate_input(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Convert and validate model input.

        The returned value is always a two-dimensional finite
        float64 NumPy array.

        Args:
            X:
                Input samples.

        Returns:
            Array with shape ``(n_samples, n_features)``.

        Raises:
            OODDetectionError:
                If the input cannot be converted to a valid
                feature matrix.
        """
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise OODDetectionError(
                "OOD input must contain numeric feature values."
            ) from exc

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise OODDetectionError(
                "Expected input of shape "
                "(n_samples, n_features)."
            )

        n_samples, n_features = values.shape

        if n_samples == 0:
            raise OODDetectionError(
                "OOD input cannot contain zero samples."
            )

        if n_features == 0:
            raise OODDetectionError(
                "OOD input must contain at least one feature."
            )

        if not np.all(np.isfinite(values)):
            raise OODDetectionError(
                "OOD input cannot contain NaN or infinite values."
            )

        return values

    def validate_feature_count(
        self,
        X: np.ndarray,
        expected_features: int,
    ) -> None:
        """
        Validate that input features match the reference feature count.

        Args:
            X:
                Validated two-dimensional input array.

            expected_features:
                Number of features used during fitting.

        Raises:
            OODDetectionError:
                If the feature count does not match.
        """
        if expected_features <= 0:
            raise OODDetectionError(
                "Expected feature count must be greater than zero."
            )

        if X.shape[1] != expected_features:
            raise OODDetectionError(
                "Feature count mismatch: expected "
                f"{expected_features} features, got "
                f"{X.shape[1]}."
            )

    def require_fitted(self) -> None:
        """
        Ensure the detector has been fitted.

        Raises:
            OODDetectionError:
                If ``fit()`` has not been called.
        """
        if not self.is_fitted:
            raise OODDetectionError(
                "OOD detector has not been fitted. "
                "Call fit(X_train) before score() or predict()."
            )

    def __call__(self, X: ModelInput) -> np.ndarray:
        """Shortcut for ``predict(X)``."""
        return self.predict(X)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"fitted={self.is_fitted})"
        )