"""
Public API for AegisAI.

AegisModel is the primary entry point for users. It automatically wraps
supported machine learning models and exposes a unified interface for
prediction with reliability analysis.
"""

from __future__ import annotations

from typing import Any

from aegis.core.pipeline import PredictionPipeline
from aegis.core.report import PredictionReport
from aegis.wrappers.base import BaseModelWrapper
from aegis.wrappers.sklearn_wrapper import SklearnWrapper


class AegisModel:
    """
    Public interface for AegisAI.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier().fit(X_train, y_train)
        >>> model = AegisModel(clf)
        >>> model.fit(X_train)
        >>> report = model.predict(X_test[:1])
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the Aegis model.

        Args:
            model:
                A trained machine learning model.
        """
        self._model = model
        self._wrapper = self._select_wrapper(model)
        self._pipeline = PredictionPipeline(self._wrapper)

    @property
    def model(self) -> Any:
        """
        Return the wrapped model.
        """
        return self._model

    @property
    def wrapper(self) -> BaseModelWrapper:
        """
        Return the selected model wrapper.
        """
        return self._wrapper

    def fit(self, X_train) -> "AegisModel":
        """
        Fit internal AegisAI components.

        Note:
            This does NOT train the ML model itself.
            It trains only AegisAI modules (e.g., OOD detector).

        Args:
            X_train:
                Training feature matrix.

        Returns:
            Self.
        """
        self._pipeline.fit(X_train)
        return self

    def predict(self, X) -> PredictionReport:
        """
        Run the complete AegisAI prediction pipeline.

        Args:
            X:
                Input samples.

        Returns:
            PredictionReport containing prediction and reliability
            metrics.
        """
        return self._pipeline.predict(X)

    def predict_batch(
        self,
        X,
    ) -> list[PredictionReport]:
        """
        Predict multiple samples.

        Args:
            X:
                Batch of samples.

        Returns:
            List of PredictionReport objects.
        """
        reports = []

        for sample in X:
            report = self.predict([sample])
            reports.append(report)

        return reports

    @staticmethod
    def _select_wrapper(model: Any) -> BaseModelWrapper:
        """
        Automatically select the appropriate wrapper.

        Current support:
            - Scikit-learn

        Future support:
            - PyTorch
            - TensorFlow
            - XGBoost
            - CatBoost
            - LightGBM
            - Hugging Face
            - Ollama
            - OpenAI
        """
        module_name = model.__class__.__module__.lower()

        if "sklearn" in module_name:
            return SklearnWrapper(model)

        raise ValueError(
            f"Unsupported model type: {model.__class__.__name__}. "
            "No compatible wrapper found."
        )

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"wrapper={self.wrapper.framework})"
        )