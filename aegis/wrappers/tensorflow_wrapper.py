"""
TensorFlow / Keras model wrapper.

This wrapper adapts TensorFlow Keras models to the AegisAI
BaseModelWrapper interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class TensorFlowWrapper(BaseModelWrapper):
    """
    Wrapper for TensorFlow / Keras models.
    """

    @property
    def framework(self) -> str:
        return "tensorflow"

    def __init__(self, model: Any) -> None:
        if tf is None:
            raise ImportError(
                "TensorFlow is not installed. "
                "Install it with: pip install tensorflow"
            )

        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate that the wrapped object is a TensorFlow model.
        """
        if not isinstance(self.model, tf.keras.Model):
            raise ModelCompatibilityError(
                "Expected a tf.keras.Model."
            )

    def _to_numpy(self, X: ModelInput) -> np.ndarray:
        """
        Convert input to a NumPy array.
        """
        return np.asarray(X, dtype=np.float32)

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Predict class labels.
        """
        try:
            x = self._to_numpy(X)

            outputs = self.model.predict(
                x,
                verbose=0,
            )

            outputs = np.asarray(outputs)

            if outputs.ndim == 1:
                return outputs

            return np.argmax(outputs, axis=1)

        except Exception as exc:
            raise PredictionError(
                f"Prediction failed: {exc}"
            ) from exc

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Predict class probabilities.
        """
        try:
            x = self._to_numpy(X)

            outputs = self.model.predict(
                x,
                verbose=0,
            )

            probabilities = np.asarray(
                outputs,
                dtype=np.float64,
            )

            # Binary sigmoid output
            if probabilities.ndim == 1:
                probabilities = np.column_stack(
                    (
                        1.0 - probabilities,
                        probabilities,
                    )
                )

            return probabilities

        except Exception as exc:
            raise PredictionError(
                f"Probability prediction failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"TensorFlowWrapper("
            f"model={self.model.__class__.__name__})"
        )