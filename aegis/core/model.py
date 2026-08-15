"""Public API for AegisAI."""

from __future__ import annotations

from typing import Any

from aegis.core.pipeline import PredictionPipeline
from aegis.core.report import PredictionReport
from aegis.wrappers.base import BaseModelWrapper
from aegis.wrappers.sklearn_wrapper import SklearnWrapper


class AegisModel:
    """Unified reliability interface for supported ML classifiers.

    The wrapped estimator must already be trained. ``fit`` trains AegisAI's
    reference components (currently the OOD detector); it does not train the
    underlying estimator.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._wrapper = self._select_wrapper(model)
        self._pipeline = PredictionPipeline(self._wrapper)

    @property
    def model(self) -> Any:
        """Return the wrapped model."""
        return self._model

    @property
    def wrapper(self) -> BaseModelWrapper:
        """Return the selected model wrapper."""
        return self._wrapper

    @property
    def is_fitted(self) -> bool:
        """Whether AegisAI's reliability components are fitted."""
        return self._pipeline.is_fitted

    def fit(self, X_train) -> "AegisModel":
        """Fit AegisAI reference components using training features."""
        self._pipeline.fit(X_train)
        return self

    def predict(self, X) -> PredictionReport:
        """Analyze the first sample in ``X`` and return its report.

        For multiple samples, use ``predict_batch`` to obtain one report
        per sample.
        """
        return self._pipeline.predict(X)

    def predict_batch(self, X) -> list[PredictionReport]:
        """Analyze every sample in ``X`` and return individual reports."""
        return self._pipeline.predict_batch(X)

    @staticmethod
    def _select_wrapper(model: Any) -> BaseModelWrapper:
        """Select a compatible wrapper for the supplied model."""
        if model is None:
            raise ValueError("Model cannot be None.")

        module_name = model.__class__.__module__.lower()
        if "sklearn" in module_name:
            return SklearnWrapper(model)

        raise ValueError(
            f"Unsupported model type: {model.__class__.__name__}. "
            "Currently supported framework: scikit-learn."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"wrapper={self.wrapper.framework}, "
            f"fitted={self.is_fitted})"
        )
