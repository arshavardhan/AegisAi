"""
PyTorch model wrapper.

This wrapper adapts PyTorch models to the AegisAI
BaseModelWrapper interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class PyTorchWrapper(BaseModelWrapper):
    """
    Wrapper for PyTorch models.
    """

    @property
    def framework(self) -> str:
        return "pytorch"

    def __init__(self, model: Any) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. "
                "Install it with: pip install torch"
            )

        super().__init__(model)
        self.validate()

        self.model.eval()

    def validate(self) -> None:
        """
        Validate the wrapped model.
        """
        if not isinstance(self.model, torch.nn.Module):
            raise ModelCompatibilityError(
                "Expected a torch.nn.Module."
            )

    def _to_tensor(self, X: ModelInput) -> torch.Tensor:
        """
        Convert input into a float32 tensor.
        """
        if isinstance(X, torch.Tensor):
            return X.float()

        return torch.tensor(
            np.asarray(X),
            dtype=torch.float32,
        )

    def predict(self, X: ModelInput) -> np.ndarray:
        """
        Predict class labels.
        """
        try:
            x = self._to_tensor(X)

            with torch.no_grad():
                outputs = self.model(x)

            if outputs.ndim == 1:
                return outputs.cpu().numpy()

            return torch.argmax(
                outputs,
                dim=1,
            ).cpu().numpy()

        except Exception as exc:
            raise PredictionError(
                f"Prediction failed: {exc}"
            ) from exc

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Predict class probabilities using softmax.
        """
        try:
            x = self._to_tensor(X)

            with torch.no_grad():
                logits = self.model(x)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            return probabilities.cpu().numpy()

        except Exception as exc:
            raise PredictionError(
                f"Probability prediction failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"PyTorchWrapper("
            f"model={self.model.__class__.__name__})"
        )