"""
CatBoost model wrapper.

This wrapper adapts CatBoost models to the AegisAI
BaseModelWrapper interface.

Supported:
    - CatBoostClassifier
    - CatBoostRegressor
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class CatBoostWrapper(BaseModelWrapper):
    """
    Wrapper for CatBoost models.
    """

    @property
    def framework(self) -> str:
        return "catboost"

    def __init__(self, model: Any) -> None:
        if CatBoostClassifier is None:
            raise ImportError(
                "CatBoost is not installed. "
                "Install it with: pip install catboost"
            )

        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate the wrapped model.
        """
        valid_types = (
            CatBoostClassifier,
            CatBoostRegressor,
        )

        if not isinstance(self.model, valid_types):
            raise ModelCompatibilityError(
                "Expected a CatBoostClassifier or "
                "CatBoostRegressor."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Predict labels or regression values.
        """
        try:
            predictions = self.model.predict(X)
            return np.asarray(predictions)

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
        if not hasattr(self.model, "predict_proba"):
            raise PredictionError(
                "This CatBoost model does not support "
                "predict_proba()."
            )

        try:
            probabilities = self.model.predict_proba(X)

            return np.asarray(
                probabilities,
                dtype=np.float64,
            )

        except Exception as exc:
            raise PredictionError(
                f"Probability prediction failed: {exc}"
            ) from exc

    def feature_importance(self) -> np.ndarray:
        """
        Return feature importance scores.
        """
        try:
            importance = self.model.get_feature_importance()
            return np.asarray(
                importance,
                dtype=np.float64,
            )

        except Exception as exc:
            raise PredictionError(
                f"Unable to compute feature importance: {exc}"
            ) from exc

    def feature_names(self) -> list[str]:
        """
        Return feature names if available.
        """
        if hasattr(self.model, "feature_names_"):
            return list(self.model.feature_names_)

        return []

    @property
    def tree_count(self) -> int:
        """
        Number of trees in the trained model.
        """
        return int(self.model.tree_count_)

    def __repr__(self) -> str:
        return (
            f"CatBoostWrapper("
            f"model={self.model.__class__.__name__}, "
            f"trees={self.tree_count})"
        )