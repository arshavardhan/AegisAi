"""
Base classes for drift detection.

All drift detection algorithms should inherit from
BaseDriftDetector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDriftDetector(ABC):
    """
    Abstract base class for drift detectors.
    """

    def __init__(
        self,
        threshold: float = 0.05,
    ) -> None:
        """
        Initialize the detector.

        Args:
            threshold:
                Drift threshold.
        """
        self.threshold = threshold
        self.reference_data: np.ndarray | None = None

    @abstractmethod
    def fit(
        self,
        reference_data: Any,
    ) -> "BaseDriftDetector":
        """
        Store reference data.

        Args:
            reference_data:
                Training/reference dataset.

        Returns:
            Self.
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        current_data: Any,
    ) -> float:
        """
        Compute drift score.

        Args:
            current_data:
                New inference data.

        Returns:
            Drift score.
        """
        raise NotImplementedError

    def detect(
        self,
        current_data: Any,
    ) -> bool:
        """
        Determine whether drift exists.

        Args:
            current_data:
                Current dataset.

        Returns:
            True if drift detected.
        """
        return self.score(current_data) > self.threshold

    def reset(self) -> None:
        """
        Clear stored reference data.
        """
        self.reference_data = None

    @property
    def is_fitted(self) -> bool:
        """
        Check if detector has been fitted.
        """
        return self.reference_data is not None

    def __call__(
        self,
        current_data: Any,
    ) -> bool:
        """
        Callable interface.
        """
        return self.detect(current_data)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )