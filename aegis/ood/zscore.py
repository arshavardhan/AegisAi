"""Z-score based Out-of-Distribution (OOD) detector."""

from __future__ import annotations

import numpy as np

from aegis.config import settings
from aegis.core.exceptions import OODDetectionError
from aegis.core.types import ModelInput
from aegis.ood.base import BaseOODDetector


class ZScoreOODDetector(BaseOODDetector):
    """Feature-wise Z-score based OOD detector.

    The detector estimates the reference distribution from training
    data and calculates the maximum absolute feature-wise Z-score
    for each new sample.

    The raw score is an unbounded distance:

        raw_score = max(abs((x - mean) / std))

    A sample is considered OOD when:

        raw_score > threshold

    The detector deliberately does not normalize the raw score.
    Normalization into an OOD risk value in [0, 1] belongs to the
    reliability pipeline.
    """

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize the detector.

        Args:
            threshold:
                Z-score threshold used for the OOD decision.
                If omitted, the configured default is used.

        Raises:
            OODDetectionError:
                If the threshold is invalid.
        """
        configured_threshold = (
            threshold
            if threshold is not None
            else settings.ood_zscore_threshold
        )

        try:
            self.threshold = float(configured_threshold)
        except (TypeError, ValueError) as exc:
            raise OODDetectionError(
                "OOD threshold must be numeric."
            ) from exc

        if not np.isfinite(self.threshold):
            raise OODDetectionError(
                "OOD threshold must be finite."
            )

        if self.threshold <= 0.0:
            raise OODDetectionError(
                "OOD threshold must be greater than zero."
            )

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features: int | None = None

    @property
    def name(self) -> str:
        """Return the detector name."""
        return "Z-Score OOD Detector"

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector has been fitted."""
        return (
            self._mean is not None
            and self._std is not None
            and self._n_features is not None
        )

    def fit(self, X: ModelInput) -> "ZScoreOODDetector":
        """Fit the detector using reference data.

        Args:
            X:
                Reference feature matrix.

        Returns:
            The fitted detector.

        Raises:
            OODDetectionError:
                If the reference data is invalid.
        """
        values = self.validate_input(X)

        if values.shape[0] < 2:
            raise OODDetectionError(
                "At least two reference samples are required "
                "to estimate feature standard deviations."
            )

        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        if not np.all(np.isfinite(mean)):
            raise OODDetectionError(
                "Reference data produced invalid feature means."
            )

        if not np.all(np.isfinite(std)):
            raise OODDetectionError(
                "Reference data produced invalid feature standard deviations."
            )

        # Constant features contain no useful information about
        # distance from the reference distribution. A scale of 1.0
        # prevents division by zero while keeping such features
        # neutral rather than creating artificially large scores.
        std = np.where(std == 0.0, 1.0, std)

        self._mean = mean
        self._std = std
        self._n_features = values.shape[1]

        return self

    def score(self, X: ModelInput) -> np.ndarray:
        """Compute the raw OOD distance for each sample.

        Args:
            X:
                Input samples.

        Returns:
            One unbounded raw OOD score per sample.

        Raises:
            OODDetectionError:
                If the detector is not fitted or the input is invalid.
        """
        self.require_fitted()

        values = self.validate_input(X)

        if self._n_features is None:
            raise OODDetectionError(
                "Detector feature configuration is unavailable."
            )

        self.validate_feature_count(
            values,
            self._n_features,
        )

        if self._mean is None or self._std is None:
            raise OODDetectionError(
                "Detector parameters are unavailable."
            )

        z_scores = np.abs(
            (values - self._mean) / self._std
        )

        raw_scores = np.max(
            z_scores,
            axis=1,
        )

        if not np.all(np.isfinite(raw_scores)):
            raise OODDetectionError(
                "OOD calculation produced NaN or infinite scores."
            )

        return raw_scores.astype(
            np.float64,
            copy=False,
        )

    def predict(self, X: ModelInput) -> np.ndarray:
        """Predict whether each sample is out-of-distribution.

        Args:
            X:
                Input samples.

        Returns:
            Boolean array where ``True`` indicates OOD.
        """
        raw_scores = self.score(X)

        return (
            raw_scores > self.threshold
        ).astype(bool)

    def statistics(self) -> dict[str, np.ndarray | float]:
        """Return fitted detector statistics.

        Returns:
            Dictionary containing the reference means, standard
            deviations, and configured threshold.

        Raises:
            OODDetectionError:
                If the detector has not been fitted.
        """
        self.require_fitted()

        if self._mean is None or self._std is None:
            raise OODDetectionError(
                "Detector parameters are unavailable."
            )

        return {
            "mean": self._mean.copy(),
            "std": self._std.copy(),
            "threshold": float(self.threshold),
        }

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"is_fitted={self.is_fitted})"
        )