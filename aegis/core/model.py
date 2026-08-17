"""
Public model interface for AegisAI.

AegisModel is the primary user-facing API for applying AegisAI
reliability analysis to a supported classification model.
"""

from __future__ import annotations

from typing import Any

from aegis.core.exceptions import ModelCompatibilityError
from aegis.core.pipeline import PredictionPipeline
from aegis.core.report import PredictionReport
from aegis.core.types import ModelInput
from aegis.wrappers.base import BaseModelWrapper
from aegis.wrappers.sklearn_wrapper import SklearnWrapper


class AegisModel:
    """Unified reliability interface for supported ML classifiers.

    AegisModel wraps an already-trained classification estimator and
    adds reliability analysis around its predictions.

    Important:
        ``AegisModel.fit(X_train)`` does NOT train the underlying
        machine-learning estimator.

        The estimator must already be fitted.

        ``fit(X_train)`` trains AegisAI's reference-dependent
        reliability components, currently including OOD detection.
    """

    def __init__(self, model: Any) -> None:
        """Initialize AegisAI around a supported model.

        Args:
            model:
                Already-trained classification estimator.

        Raises:
            ModelCompatibilityError:
                If the supplied model is missing, unsupported, or
                incompatible with the AegisAI classification API.
        """
        if model is None:
            raise ModelCompatibilityError(
                "Model instance cannot be None."
            )

        self._model = model
        self._wrapper = self._select_wrapper(model)

        self._pipeline = PredictionPipeline(
            self._wrapper
        )

    @property
    def model(self) -> Any:
        """Return the underlying machine-learning model."""
        return self._model

    @property
    def wrapper(self) -> BaseModelWrapper:
        """Return the model wrapper used by AegisAI."""
        return self._wrapper

    @property
    def pipeline(self) -> PredictionPipeline:
        """Return the underlying reliability pipeline."""
        return self._pipeline

    @property
    def is_fitted(self) -> bool:
        """Return whether AegisAI's reference components are fitted."""
        return self._pipeline.is_fitted

    def fit(
        self,
        X_train: ModelInput,
    ) -> "AegisModel":
        """Fit AegisAI's reference-dependent components.

        This method does not train the underlying estimator.

        Args:
            X_train:
                Training/reference features used by reliability
                components such as OOD detection.

        Returns:
            The current AegisModel instance.

        Example:
            >>> classifier.fit(X_train, y_train)
            >>> aegis = AegisModel(classifier)
            >>> aegis.fit(X_train)
        """
        self._pipeline.fit(X_train)
        return self

    def predict(
        self,
        X: ModelInput,
    ) -> PredictionReport:
        """Analyze the first input sample.

        Args:
            X:
                One or more input samples.

        Returns:
            Reliability report for the first sample.

        Raises:
            PredictionError:
                If AegisAI has not been fitted or prediction fails.
        """
        return self._pipeline.predict(X)

    def predict_batch(
        self,
        X: ModelInput,
    ) -> list[PredictionReport]:
        """Analyze every input sample.

        Args:
            X:
                Input samples.

        Returns:
            One reliability report per input sample.

        Raises:
            PredictionError:
                If AegisAI has not been fitted or prediction fails.
        """
        return self._pipeline.predict_batch(X)

    @staticmethod
    def _select_wrapper(
        model: Any,
    ) -> BaseModelWrapper:
        """Select the wrapper for a supported model.

        The current public framework supports scikit-learn
        classification estimators.

        Args:
            model:
                Model instance to wrap.

        Returns:
            Compatible model wrapper.

        Raises:
            ModelCompatibilityError:
                If the model belongs to an unsupported framework.
        """
        module_name = (
            model.__class__.__module__.lower()
        )

        if module_name.startswith("sklearn."):
            return SklearnWrapper(model)

        raise ModelCompatibilityError(
            "Unsupported model type: "
            f"{model.__class__.__module__}."
            f"{model.__class__.__name__}. "
            "Currently supported framework: "
            "scikit-learn."
        )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"wrapper={self.wrapper.framework}, "
            f"fitted={self.is_fitted})"
        )