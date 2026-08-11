"""
LightGBM model wrapper.

This wrapper adapts LightGBM estimators to the AegisAI
BaseModelWrapper interface.

Supported:
    - LGBMClassifier
    - LGBMRegressor
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class LightGBMWrapper(BaseModelWrapper):
    """
    Wrapper for LightGBM estimators.
    """

    @property
    def framework(self) -> str:
        return "lightgbm"

    def __init__(self, model: Any) -> None:
        if LGBMClassifier is None:
            raise ImportError(
                "LightGBM is not installed. "
                "Install it with: pip install lightgbm"
            )

        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate the wrapped model.
        """
        valid_types = (
            LGBMClassifier,
            LGBMRegressor,
        )

        if not isinstance(self.model, valid_types):
            raise ModelCompatibilityError(
                "Expected an LGBMClassifier or "
                "LGBMRegressor."
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
                "This LightGBM model does not support "
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

    @property
    def booster(self):
        """
        Return the native LightGBM Booster object.
        """
        if hasattr(self.model, "booster_"):
            return self.model.booster_

        return None

    def feature_importance(
        self,
        importance_type: str = "split",
    ) -> np.ndarray:
        """
        Return feature importance.

        Args:
            importance_type:
                "split" or "gain"
        """
        booster = self.booster

        if booster is None:
            raise PredictionError(
                "Native booster is unavailable."
            )

        return np.asarray(
            booster.feature_importance(
                importance_type=importance_type
            ),
            dtype=np.float64,
        )

    def feature_names(self) -> list[str]:
        """
        Return feature names.
        """
        booster = self.booster

        if booster is None:
            return []

        return booster.feature_name()

    @property
    def num_trees(self) -> int:
        """
        Number of trees in the model.
        """
        booster = self.booster

        if booster is None:
            return 0

        return booster.num_trees()

    def __repr__(self) -> str:
        return (
            f"LightGBMWrapper("
            f"model={self.model.__class__.__name__}, "
            f"trees={self.num_trees})"
        )