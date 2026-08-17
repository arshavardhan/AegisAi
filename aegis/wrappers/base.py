"""
Base wrapper interface for AegisAI model integrations.

AegisAI uses wrappers to provide a consistent interface between
different machine-learning frameworks and the reliability pipeline.

For the current classification pipeline, a supported wrapper must
provide:

    predict()
    predict_proba()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aegis.core.exceptions import ModelCompatibilityError
from aegis.core.types import ModelInput, Prediction, ProbabilityVector


class BaseModelWrapper(ABC):
    """
    Abstract base class for AegisAI model wrappers.

    A wrapper adapts an underlying trained model to the interface
    required by the AegisAI reliability pipeline.

    Attributes:
        model:
            The underlying model instance.
    """

    def __init__(self, model: Any) -> None:
        if model is None:
            raise ModelCompatibilityError(
                "Model instance cannot be None."
            )

        self._model = model

    @property
    def model(self) -> Any:
        """Return the wrapped model instance."""
        return self._model

    @property
    @abstractmethod
    def framework(self) -> str:
        """
        Return the name of the underlying ML framework.

        Example:
            scikit-learn
        """

    @property
    def supports_predict(self) -> bool:
        """Return whether the model implements predict()."""
        return callable(
            getattr(
                self.model,
                "predict",
                None,
            )
        )

    @property
    def supports_predict_proba(self) -> bool:
        """Return whether the model implements predict_proba()."""
        return callable(
            getattr(
                self.model,
                "predict_proba",
                None,
            )
        )

    @property
    def supports_decision_function(self) -> bool:
        """Return whether the model implements decision_function()."""
        return callable(
            getattr(
                self.model,
                "decision_function",
                None,
            )
        )

    @abstractmethod
    def predict(
        self,
        X: ModelInput,
    ) -> Prediction | np.ndarray:
        """
        Generate model predictions.

        Args:
            X:
                Input samples.

        Returns:
            One prediction per input sample.
        """

    @abstractmethod
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
            ModelCompatibilityError:
                If probability prediction is unavailable.
        """

    def decision_function(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Generate decision scores when supported.

        Args:
            X:
                Input samples.

        Returns:
            Decision scores.

        Raises:
            ModelCompatibilityError:
                If decision_function() is unavailable.
        """
        if not self.supports_decision_function:
            raise ModelCompatibilityError(
                f"{self.framework} model does not support "
                "decision_function()."
            )

        try:
            return np.asarray(
                self.model.decision_function(X)
            )
        except Exception as exc:
            raise ModelCompatibilityError(
                f"{self.framework} decision_function() failed: "
                f"{exc}"
            ) from exc

    def validate(self) -> None:
        """
        Validate the minimum model API required by AegisAI.

        For the current classification reliability pipeline,
        both predict() and predict_proba() are required.

        Raises:
            ModelCompatibilityError:
                If the model does not satisfy the required API.
        """
        if not self.supports_predict:
            raise ModelCompatibilityError(
                f"{self.framework} model must implement predict()."
            )

        if not self.supports_predict_proba:
            raise ModelCompatibilityError(
                f"{self.framework} model must implement "
                "predict_proba() for the current "
                "classification reliability pipeline."
            )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"framework='{self.framework}', "
            f"model={self.model.__class__.__name__})"
        )