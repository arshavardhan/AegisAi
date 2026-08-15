"""Z-score based Out-of-Distribution (OOD) detector."""

from __future__ import annotations

import numpy as np

from aegis.config import settings
from aegis.core.types import ModelInput
from aegis.ood.base import BaseOODDetector


class ZScoreOODDetector(BaseOODDetector):
    """Feature-wise Z-score OOD detector.

    The raw score is the maximum absolute feature Z-score. AegisAI's
    pipeline converts that unbounded distance into a normalized risk value.
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = float(
            threshold if threshold is not None else settings.ood_zscore_threshold
        )
        if self.threshold <= 0:
            raise ValueError("OOD threshold must be greater than zero.")

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features: int | None = None

    @property
    def name(self) -> str:
        return "Z-Score OOD Detector"

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._std is not None

    def fit(self, X: ModelInput) -> "ZScoreOODDetector":
        X = self.validate_input(X)
        if X.shape[0] == 0:
            raise ValueError("Reference data cannot be empty.")

        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0.0] = 1e-12
        self._n_features = X.shape[1]
        return self

    def _validate_fitted_shape(self, X: np.ndarray) -> None:
        if self._mean is None or self._std is None or self._n_features is None:
            raise RuntimeError("Detector has not been fitted.")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {X.shape[1]}."
            )

    def score(self, X: ModelInput) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Detector has not been fitted.")

        X = self.validate_input(X)
        self._validate_fitted_shape(X)
        z_scores = np.abs((X - self._mean) / self._std)
        return np.max(z_scores, axis=1)

    def predict(self, X: ModelInput) -> np.ndarray:
        return self.score(X) > self.threshold

    def statistics(self) -> dict[str, np.ndarray | float]:
        if not self.is_fitted:
            raise RuntimeError("Detector has not been fitted.")

        return {
            "mean": self._mean.copy(),
            "std": self._std.copy(),
            "threshold": self.threshold,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, is_fitted={self.is_fitted})"
        )
