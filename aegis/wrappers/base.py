"""
Base wrapper interface for all AI models supported by AegisAI.

Every model integration (scikit-learn, PyTorch, TensorFlow, XGBoost,
LLMs, etc.) must inherit from BaseModelWrapper.

The wrapper provides a unified interface so the rest of the framework
never needs to know which ML library is being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aegis.core.exceptions import ModelCompatibilityError
from aegis.core.types import ModelInput, Prediction, ProbabilityVector


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    Every supported AI framework should implement this interface.

    Attributes:
        model:
            The underlying trained model instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

        if model is None:
            raise ModelCompatibilityError(
                "Model instance cannot be None."
            )

    @property
    def model(self) -> Any:
        """
        Return the wrapped model instance.
        """
        return self._model

    @property
    @abstractmethod
    def framework(self) -> str:
        """
        Name of the underlying ML framework.

        Example:
            scikit-learn
            pytorch
            tensorflow
        """

    @property
    def supports_predict_proba(self) -> bool:
        """
        Whether the wrapped model implements predict_proba().
        """
        return callable(getattr(self.model, "predict_proba", None))

    @property
    def supports_decision_function(self) -> bool:
        """
        Whether the wrapped model implements decision_function().
        """
        return callable(getattr(self.model, "decision_function", None))

    @property
    def supports_predict(self) -> bool:
        """
        Whether the wrapped model implements predict().
        """
        return callable(getattr(self.model, "predict", None))

    @abstractmethod
    def predict(self, X: ModelInput) -> Prediction | np.ndarray:
        """
        Predict labels.

        Args:
            X:
                Input samples.

        Returns:
            Model predictions.
        """

    @abstractmethod
    def predict_proba(self, X: ModelInput) -> ProbabilityVector:
        """
        Predict class probabilities.

        Args:
            X:
                Input samples.

        Returns:
            Probability vector.

        Raises:
            NotImplementedError:
                If probability prediction is unavailable.
        """

    def decision_function(self, X: ModelInput) -> np.ndarray:
        """
        Return decision scores if supported.

        Args:
            X:
                Input samples.

        Returns:
            Decision scores.

        Raises:
            NotImplementedError:
                If the wrapped model does not support
                decision_function().
        """
        if not self.supports_decision_function:
            raise NotImplementedError(
                f"{self.framework} does not support "
                "decision_function()."
            )

        return self.model.decision_function(X)

    def validate(self) -> None:
        """
        Validate that the wrapped model satisfies the minimum API
        required by AegisAI.

        Raises:
            ModelCompatibilityError:
                If the model is incompatible.
        """
        if not self.supports_predict:
            raise ModelCompatibilityError(
                "Wrapped model must implement predict()."
            )

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"framework='{self.framework}', "
            f"model={self.model.__class__.__name__})"
        )