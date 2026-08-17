"""
Scikit-learn model wrapper.

Provides the AegisAI interface for scikit-learn classification
estimators.
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
    Wrapper for scikit-learn classification estimators.

    The current AegisAI reliability pipeline requires estimators
    implementing:

        predict()
        predict_proba()
        classes_
    """

    @property
    def framework(self) -> str:
        """Return the wrapped framework name."""
        return "scikit-learn"

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate the wrapped estimator.

        Raises:
            ModelCompatibilityError:
                If the object does not satisfy the AegisAI
                classification wrapper contract.
        """
        super().validate()

        if not callable(
            getattr(
                self.model,
                "fit",
                None,
            )
        ):
            raise ModelCompatibilityError(
                "The supplied object does not appear to be a "
                "valid scikit-learn estimator."
            )

        if not hasattr(self.model, "classes_"):
            raise ModelCompatibilityError(
                "The supplied scikit-learn estimator does not "
                "appear to be fitted. Missing classes_."
            )

        classes = np.asarray(
            self.model.classes_
        )

        if classes.ndim != 1:
            raise ModelCompatibilityError(
                "The estimator classes_ attribute must be "
                "one-dimensional."
            )

        if len(classes) < 2:
            raise ModelCompatibilityError(
                "AegisAI classification models must contain "
                "at least two classes."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Generate class predictions.

        Args:
            X:
                Input samples.

        Returns:
            One prediction per sample.

        Raises:
            PredictionError:
                If prediction fails.
        """
        try:
            predictions = np.asarray(
                self.model.predict(X)
            )

        except Exception as exc:
            raise PredictionError(
                f"scikit-learn prediction failed: {exc}"
            ) from exc

        if predictions.ndim == 0:
            predictions = predictions.reshape(1)

        if predictions.ndim != 1:
            raise PredictionError(
                "scikit-learn predictions must be "
                "one-dimensional."
            )

        if len(predictions) == 0:
            raise PredictionError(
                "scikit-learn returned no predictions."
            )

        return predictions

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Generate class probabilities.

        Args:
            X:
                Input samples.

        Returns:
            Probability matrix with shape:

                (n_samples, n_classes)

        Raises:
            PredictionError:
                If probabilities are unavailable or invalid.
        """
        if not self.supports_predict_proba:
            raise PredictionError(
                f"{self.model.__class__.__name__} "
                "does not implement predict_proba()."
            )

        try:
            probabilities = np.asarray(
                self.model.predict_proba(X),
                dtype=np.float64,
            )

        except Exception as exc:
            raise PredictionError(
                "scikit-learn probability prediction failed: "
                f"{exc}"
            ) from exc

        self._validate_probabilities(probabilities)

        return probabilities

    def predict_confidence(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Return maximum predicted probability for each sample.

        This is a convenience method. The main AegisAI pipeline
        calculates confidence through its uncertainty subsystem.
        """
        probabilities = self.predict_proba(X)

        return np.max(
            probabilities,
            axis=1,
        )

    def classes(self) -> np.ndarray:
        """
        Return the estimator's class labels.

        Returns:
            One class label per probability column.

        Raises:
            ModelCompatibilityError:
                If classes_ is unavailable or invalid.
        """
        if not hasattr(
            self.model,
            "classes_",
        ):
            raise ModelCompatibilityError(
                "Model does not expose classes_."
            )

        classes = np.asarray(
            self.model.classes_
        )

        if classes.ndim != 1:
            raise ModelCompatibilityError(
                "Model classes_ must be one-dimensional."
            )

        if len(classes) < 2:
            raise ModelCompatibilityError(
                "At least two classes are required."
            )

        return classes

    def _validate_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """Validate sklearn probability output."""
        if probabilities.ndim != 2:
            raise PredictionError(
                "scikit-learn predict_proba() must return a "
                "two-dimensional array."
            )

        if probabilities.shape[1] < 2:
            raise PredictionError(
                "Probability output must contain at least "
                "two classes."
            )

        if probabilities.shape[0] == 0:
            raise PredictionError(
                "Probability output contains no samples."
            )

        if not np.all(
            np.isfinite(probabilities)
        ):
            raise PredictionError(
                "Probability output contains NaN or "
                "infinite values."
            )

        if np.any(probabilities < 0.0) or np.any(
            probabilities > 1.0
        ):
            raise PredictionError(
                "Probability values must be within [0, 1]."
            )

        row_sums = probabilities.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            raise PredictionError(
                "Each probability row must sum to 1."
            )

        try:
            classes = self.classes()
        except ModelCompatibilityError as exc:
            raise PredictionError(
                str(exc)
            ) from exc

        if probabilities.shape[1] != len(classes):
            raise PredictionError(
                "Probability output class count does not "
                "match the estimator classes_."
            )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"predict={self.supports_predict}, "
            f"predict_proba="
            f"{self.supports_predict_proba})"
        )