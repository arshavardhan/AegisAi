"""
Z-Score based Out-of-Distribution (OOD) detector.

This detector learns the feature-wise mean and standard deviation from
reference data. New samples are evaluated using their absolute
Z-scores.

A sample is considered OOD if its maximum absolute feature Z-score
exceeds the configured threshold.
"""

from __future__ import annotations

import numpy as np

from aegis.config import settings
from aegis.core.types import ModelInput
from aegis.ood.base import BaseOODDetector


class ZScoreOODDetector(BaseOODDetector):
    """
    Feature-wise Z-score based OOD detector.
    """

    def __init__(
        self,
        threshold: float | None = None,
    ) -> None:
        """
        Initialize the detector.

        Args:
            threshold:
                Z-score threshold. If None, the global framework
                setting is used.
        """
        self.threshold = (
            threshold
            if threshold is not None
            else settings.ood_zscore_threshold
        )

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Z-Score OOD Detector"

    def fit(
        self,
        X: ModelInput,
    ) -> "ZScoreOODDetector":
        """
        Learn feature statistics from reference data.
        """
        X = self.validate_input(X)

        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        # Prevent division by zero
        self._std[self._std == 0.0] = 1e-12

        return self

    def score(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Compute OOD scores.

        The score is defined as the maximum absolute feature Z-score
        for each sample.

        Args:
            X:
                Samples to evaluate.

        Returns:
            One OOD score per sample.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Detector has not been fitted."
            )

        X = self.validate_input(X)

        z_scores = np.abs((X - self._mean) / self._std)

        return np.max(z_scores, axis=1)

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Predict whether samples are out-of-distribution.

        Returns:
            Boolean NumPy array where:

            True  -> OOD

            False -> In-distribution
        """
        scores = self.score(X)

        return scores > self.threshold

    def statistics(self) -> dict[str, np.ndarray]:
        """
        Return detector statistics.

        Returns:
            Dictionary containing learned feature statistics.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Detector has not been fitted."
            )

        return {
            "mean": self._mean.copy(),
            "std": self._std.copy(),
            "threshold": np.asarray(self.threshold),
        }

    @property
    def is_fitted(self) -> bool:
        """
        Whether the detector has been fitted.
        """
        return (
            self._mean is not None
            and self._std is not None
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"is_fitted={self.is_fitted})"
        )