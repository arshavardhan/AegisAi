"""
Hugging Face Transformers Pipeline wrapper.

Currently supported:
    - transformers.pipeline(task="text-classification")

Future versions will support:
    - AutoModelForSequenceClassification
    - SentenceTransformer
    - Token Classification
    - Zero-shot Classification
    - Question Answering
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from transformers import Pipeline
except ImportError:
    Pipeline = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class HuggingFaceWrapper(BaseModelWrapper):
    """
    Wrapper for Hugging Face text classification pipelines.
    """

    @property
    def framework(self) -> str:
        return "huggingface"

    def __init__(self, model: Any) -> None:
        if Pipeline is None:
            raise ImportError(
                "Transformers is not installed. "
                "Install it with: pip install transformers"
            )

        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate the wrapped object.
        """
        if not isinstance(self.model, Pipeline):
            raise ModelCompatibilityError(
                "Expected a transformers.Pipeline instance."
            )

        if self.model.task != "text-classification":
            raise ModelCompatibilityError(
                "Only text-classification pipelines are "
                "currently supported."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Predict labels.

        Args:
            X:
                A string or list of strings.

        Returns:
            NumPy array of predicted labels.
        """
        try:
            outputs = self.model(X)

            labels = [
                result["label"]
                for result in outputs
            ]

            return np.asarray(labels)

        except Exception as exc:
            raise PredictionError(
                f"Prediction failed: {exc}"
            ) from exc

    def predict_proba(
        self,
        X: ModelInput,
    ) -> ProbabilityVector:
        """
        Return confidence scores.

        Since the standard Hugging Face pipeline only returns
        the predicted label and score, we construct a simple
        probability vector.

        NOTE:
            Future versions will support return_all_scores=True
            for complete probability distributions.
        """
        try:
            outputs = self.model(X)

            probabilities = []

            for result in outputs:
                score = float(result["score"])

                probabilities.append(
                    [
                        1.0 - score,
                        score,
                    ]
                )

            return np.asarray(
                probabilities,
                dtype=np.float64,
            )

        except Exception as exc:
            raise PredictionError(
                f"Probability prediction failed: {exc}"
            ) from exc

    def labels(
        self,
        X: ModelInput,
    ) -> list[str]:
        """
        Return predicted labels.
        """
        return self.predict(X).tolist()

    def scores(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Return confidence scores only.
        """
        outputs = self.model(X)

        return np.asarray(
            [
                float(result["score"])
                for result in outputs
            ]
        )

    def __repr__(self) -> str:
        return (
            f"HuggingFaceWrapper("
            f"task={self.model.task})"
        )