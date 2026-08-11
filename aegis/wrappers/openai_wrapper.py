"""
OpenAI model wrapper.

This wrapper adapts OpenAI chat models to the
AegisAI BaseModelWrapper interface.

Supported:
    - gpt-4o
    - gpt-4.1
    - gpt-5
    - Future chat-based OpenAI models

Requires:
    pip install openai
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import (
    ModelInput,
    ProbabilityVector,
)
from aegis.wrappers.base import BaseModelWrapper


class OpenAIWrapper(BaseModelWrapper):
    """
    Wrapper for OpenAI chat models.
    """

    @property
    def framework(self) -> str:
        return "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "OpenAI package is not installed.\n"
                "Install with: pip install openai"
            )

        if not api_key:
            raise ModelCompatibilityError(
                "An OpenAI API key is required."
            )

        self.client = OpenAI(api_key=api_key)

        super().__init__(model)

        self.validate()

    def validate(self) -> None:
        """
        Validate the supplied model name.
        """
        if not isinstance(self.model, str):
            raise ModelCompatibilityError(
                "Model name must be a string."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Generate a response for the supplied prompt.

        Args:
            X:
                Prompt string.

        Returns:
            NumPy array containing generated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": str(X),
                    }
                ],
            )

            text = response.choices[0].message.content

            return np.asarray([text])

        except Exception as exc:
            raise PredictionError(
                f"OpenAI request failed: {exc}"
            ) from exc

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Placeholder probability output.

        Current OpenAI APIs do not consistently expose
        probability distributions across all models.

        Future versions will support:
            - logprobs
            - token confidence
            - uncertainty estimation
        """
        _ = X

        return np.asarray([[1.0]])

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text from a prompt.
        """
        return self.predict(prompt)[0]

    def chat(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        Multi-turn chat.

        Args:
            messages:
                OpenAI chat-format messages.

        Returns:
            Assistant response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as exc:
            raise PredictionError(
                f"Chat request failed: {exc}"
            ) from exc

    def embedding(
        self,
        text: str,
    ):
        """
        Placeholder for embedding support.

        Future versions will integrate the
        embeddings API.
        """
        raise NotImplementedError(
            "Embedding support will be added in v0.2."
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIWrapper("
            f"model='{self.model}')"
        )