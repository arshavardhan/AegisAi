"""
XGBoost model wrapper.

This wrapper adapts XGBoost estimators to the AegisAI
BaseModelWrapper interface.

Supported:
    - XGBClassifier
    - XGBRegressor
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from aegis.core.exceptions import (
    ModelCompatibilityError,
    PredictionError,
)
from aegis.core.types import ModelInput, ProbabilityVector
from aegis.wrappers.base import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    """
    Wrapper for XGBoost estimators.
    """

    @property
    def framework(self) -> str:
        return "xgboost"

    def __init__(self, model: Any) -> None:
        if xgb is None:
            raise ImportError(
                "XGBoost is not installed. "
                "Install it with: pip install xgboost"
            )

        super().__init__(model)
        self.validate()

    def validate(self) -> None:
        """
        Validate that the wrapped object is an XGBoost estimator.
        """
        valid_types = (
            xgb.XGBClassifier,
            xgb.XGBRegressor,
        )

        if not isinstance(self.model, valid_types):
            raise ModelCompatibilityError(
                "Expected an XGBClassifier or XGBRegressor."
            )

    def predict(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        Predict labels or regression values.
        """
        try:
            return np.asarray(self.model.predict(X))

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

        Raises:
            PredictionError:
                If the wrapped estimator does not support
                probability prediction.
        """
        if not hasattr(self.model, "predict_proba"):
            raise PredictionError(
                "This XGBoost model does not support "
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
        Return the underlying Booster object.
        """
        if hasattr(self.model, "get_booster"):
            return self.model.get_booster()

        return None

    def feature_importance(self) -> np.ndarray:
        """
        Return feature importance scores.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise PredictionError(
                "Feature importance is unavailable."
            )

        return np.asarray(
            self.model.feature_importances_,
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        return (
            f"XGBoostWrapper("
            f"model={self.model.__class__.__name__})"
        )