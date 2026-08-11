"""
Statistical data drift detector.

Uses the Kolmogorov-Smirnov (KS) test to compare the
distribution of reference data against incoming data.

Reference:
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from aegis.drift.base import BaseDriftDetector


class DataDriftDetector(BaseDriftDetector):
    """
    Detect numerical data drift using the KS test.

    The detector compares each feature independently and returns
    the worst (minimum) p-value across all features.

    Lower p-values indicate stronger evidence of drift.
    """

    def fit(
        self,
        reference_data: Any,
    ) -> "DataDriftDetector":
        """
        Store the reference dataset.

        Args:
            reference_data:
                Training dataset.

        Returns:
            Self.
        """
        reference = np.asarray(reference_data)

        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)

        self.reference_data = reference

        return self

    def score(
        self,
        current_data: Any,
    ) -> float:
        """
        Compute the drift score.

        Returns:
            Drift score in [0,1].

            Higher score = more drift.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Detector must be fitted first."
            )

        current = np.asarray(current_data)

        if current.ndim == 1:
            current = current.reshape(-1, 1)

        p_values = []

        for feature in range(self.reference_data.shape[1]):
            _, p_value = ks_2samp(
                self.reference_data[:, feature],
                current[:, feature],
            )

            p_values.append(p_value)

        min_p = float(min(p_values))

        # Convert p-value into drift score.
        drift_score = 1.0 - min_p

        return float(
            np.clip(drift_score, 0.0, 1.0)
        )

    def feature_scores(
        self,
        current_data: Any,
    ) -> dict[int, float]:
        """
        Compute drift score for every feature individually.

        Returns:
            Mapping:
                feature_index -> drift_score
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Detector must be fitted first."
            )

        current = np.asarray(current_data)

        if current.ndim == 1:
            current = current.reshape(-1, 1)

        scores = {}

        for feature in range(self.reference_data.shape[1]):
            _, p = ks_2samp(
                self.reference_data[:, feature],
                current[:, feature],
            )

            scores[feature] = float(
                np.clip(1.0 - p, 0.0, 1.0)
            )

        return scores

    def summary(
        self,
        current_data: Any,
    ) -> dict[str, Any]:
        """
        Return a drift summary.
        """
        score = self.score(current_data)

        return {
            "drift_score": score,
            "drift_detected": score > self.threshold,
            "feature_scores": self.feature_scores(
                current_data
            ),
        }