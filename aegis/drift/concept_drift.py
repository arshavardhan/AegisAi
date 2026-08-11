"""
Concept drift detector.

Concept drift occurs when the relationship between inputs and outputs
changes over time, even if the input distribution itself remains stable.

This implementation monitors prediction distribution drift using the
Kolmogorov-Smirnov (KS) test.

Future versions may include:
    - DDM (Drift Detection Method)
    - EDDM
    - ADWIN
    - Page-Hinkley
    - Online accuracy monitoring
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from aegis.drift.base import BaseDriftDetector


class ConceptDriftDetector(BaseDriftDetector):
    """
    Detect concept drift using model prediction distributions.
    """

    def fit(
        self,
        reference_predictions: Any,
    ) -> "ConceptDriftDetector":
        """
        Store baseline model predictions.

        Args:
            reference_predictions:
                Predictions from the reference period.

        Returns:
            Self.
        """
        self.reference_data = np.asarray(
            reference_predictions,
            dtype=np.float64,
        ).flatten()

        return self

    def score(
        self,
        current_predictions: Any,
    ) -> float:
        """
        Compute concept drift score.

        Args:
            current_predictions:
                Predictions from the current period.

        Returns:
            Drift score in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Detector must be fitted first."
            )

        current = np.asarray(
            current_predictions,
            dtype=np.float64,
        ).flatten()

        _, p_value = ks_2samp(
            self.reference_data,
            current,
        )

        drift_score = 1.0 - float(p_value)

        return float(
            np.clip(drift_score, 0.0, 1.0)
        )

    def summary(
        self,
        current_predictions: Any,
    ) -> dict[str, Any]:
        """
        Return a summary of concept drift.
        """
        score = self.score(current_predictions)

        return {
            "drift_score": score,
            "drift_detected": score > self.threshold,
            "reference_samples": len(self.reference_data),
            "current_samples": len(
                np.asarray(current_predictions).flatten()
            ),
        }

    def update_reference(
        self,
        predictions: Any,
    ) -> None:
        """
        Replace the reference prediction distribution.

        Useful when retraining a model.
        """
        self.reference_data = np.asarray(
            predictions,
            dtype=np.float64,
        ).flatten()