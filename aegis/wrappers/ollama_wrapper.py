"""
Ollama model wrapper.

This wrapper adapts locally running Ollama models to the
AegisAI BaseModelWrapper interface.

Supported:
    - llama3
    - mistral
    - gemma
    - qwen
    - deepseek
    - Any Ollama chat model

Requires:
    pip install ollama
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ollama import Client
except ImportError:
    Client = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class OllamaWrapper(BaseModelWrapper):
    """
    Wrapper for Ollama chat models.
    """

    @property
    def framework(self) -> str:
        return "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ) -> None:
        if Client is None:
            raise ImportError(
                "Ollama package is not installed.\n"
                "Install with: pip install ollama"
            )

        self.client = Client(host=host)

        super().__init__(model)

    def validate(self) -> None:
        """
        Validate the supplied model name.
        """
        if not isinstance(self.model, str):
            raise ModelCompatibilityError(
                "Ollama model must be specified as a string."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Generate text using the Ollama chat API.

        Args:
            X:
                Prompt string.

        Returns:
            NumPy array containing generated text.
        """
        try:
            prompt = str(X)

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            text = response["message"]["content"]

            return np.asarray([text])

        except Exception as exc:
            raise PredictionError(
                f"Ollama generation failed: {exc}"
            ) from exc

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Ollama currently does not expose token probabilities.

        Placeholder implementation.

        Future versions will support:

        - logprobs
        - uncertainty estimation
        - sampling variance
        """
        prediction = self.predict(X)

        return np.asarray([[1.0]])

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Convenience method returning plain text.
        """
        return self.predict(prompt)[0]

    def chat(
        self,
        messages: list[dict],
    ) -> str:
        """
        Multi-turn chat.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
            )

            return response["message"]["content"]

        except Exception as exc:
            raise PredictionError(
                f"Chat failed: {exc}"
            ) from exc

    def embeddings(
        self,
        text: str,
    ):
        """
        Placeholder for future embedding support.
        """
        raise NotImplementedError(
            "Embedding support will be added in v0.2."
        )

    def __repr__(self) -> str:
        return (
            f"OllamaWrapper("
            f"model='{self.model}')"
        )