"""
Scikit-learn model wrapper.

This module provides a unified interface for scikit-learn estimators,
allowing them to integrate seamlessly with the AegisAI reliability
pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class SklearnWrapper(BaseModelWrapper):
    """
    Wrapper for scikit-learn estimators.

    Supported estimators include classifiers and regressors that
    implement the standard scikit-learn API.
    """

    @property
    def framework(self) -> str:
        """
        Name of the wrapped framework.
        """
        return "scikit-learn"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate that the wrapped estimator exposes the required API.
        """
        super().validate()

        if not callable(getattr(self.model, "fit", None)):
            raise ModelCompatibilityError(
                "The supplied object does not appear to be a "
                "valid scikit-learn estimator."
            )

    def predict(self, X: ModelInput) -> np.ndarray:
        """
        Predict labels or values.

        Args:
            X:
                Input samples.

        Returns:
            NumPy array of predictions.
        """
        try:
            predictions = self.model.predict(X)
            return np.asarray(predictions)

        except Exception as exc:
            raise PredictionError(
                f"Prediction failed: {exc}"
            ) from exc

    def predict_proba(self, X: ModelInput) -> ProbabilityVector:
        """
        Predict class probabilities.

        Args:
            X:
                Input samples.

        Returns:
            Probability matrix of shape
            (n_samples, n_classes).

        Raises:
            PredictionError:
                If probabilities cannot be computed.
        """
        if not self.supports_predict_proba:
            raise PredictionError(
                f"{self.model.__class__.__name__} "
                "does not implement predict_proba()."
            )

        try:
            probabilities = self.model.predict_proba(X)
            return np.asarray(probabilities, dtype=np.float64)

        except Exception as exc:
            raise PredictionError(
                f"Probability prediction failed: {exc}"
            ) from exc

    def predict_confidence(self, X: ModelInput) -> np.ndarray:
        """
        Return the maximum predicted probability for each sample.

        This provides a simple confidence estimate for
        classification models.

        Args:
            X:
                Input samples.

        Returns:
            Confidence score for each prediction.

        Raises:
            PredictionError:
                If probability estimation is unavailable.
        """
        probabilities = self.predict_proba(X)
        return probabilities.max(axis=1)

    def classes(self) -> np.ndarray:
        """
        Return the estimator's class labels.

        Returns:
            Array of class labels.

        Raises:
            ModelCompatibilityError:
                If the estimator has no classes_ attribute.
        """
        if not hasattr(self.model, "classes_"):
            raise ModelCompatibilityError(
                "Model does not expose classes_."
            )

        return np.asarray(self.model.classes_)

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """
        return (
            f"SklearnWrapper("
            f"model={self.model.__class__.__name__}, "
            f"predict_proba={self.supports_predict_proba})"
        )