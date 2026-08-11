"""
Prediction pipeline for AegisAI.

This module orchestrates the complete prediction workflow,
combining model inference with reliability analysis.

Pipeline:

Input
    ↓
Model Prediction
    ↓
Probability Prediction
    ↓
Confidence
    ↓
Entropy
    ↓
OOD Detection
    ↓
Trust Score
    ↓
Recommendation
    ↓
PredictionReport
"""

from __future__ import annotations

import numpy as np

from aegis.core.exceptions import PredictionError
from aegis.core.report import PredictionReport
from aegis.ood.zscore import ZScoreOODDetector
from aegis.recommendation.engine import RecommendationEngine
from aegis.scoring.trust import TrustScorer
from aegis.uncertainty.confidence import ConfidenceEstimator
from aegis.uncertainty.entropy import EntropyEstimator
from aegis.wrappers.base import BaseModelWrapper


class PredictionPipeline:
    """
    Main prediction pipeline.
    """

    def __init__(
        self,
        wrapper: BaseModelWrapper,
        ood_detector: ZScoreOODDetector | None = None,
    ) -> None:

        self.wrapper = wrapper

        self.confidence = ConfidenceEstimator()
        self.entropy = EntropyEstimator()

        self.ood = ood_detector or ZScoreOODDetector()

        self.trust = TrustScorer()

        self.recommendation = RecommendationEngine()

    def fit(
        self,
        X_train,
    ) -> None:
        """
        Fit internal components.

        Currently only the OOD detector requires fitting.
        """
        self.ood.fit(X_train)

    def predict(
        self,
        X,
    ) -> PredictionReport:
        """
        Execute the complete reliability pipeline.
        """

        try:

            prediction = self.wrapper.predict(X)

            probabilities = self.wrapper.predict_proba(X)

            confidence = float(
                self.confidence.compute(probabilities)[0]
            )

            entropy = float(
                self.entropy.compute(probabilities)[0]
            )

            ood_score = float(
                self.ood.score(X)[0]
            )

            is_ood = bool(
                self.ood.predict(X)[0]
            )

            trust_result = self.trust.compute(
                confidence=confidence,
                entropy=entropy,
                ood_score=ood_score,
                is_ood=is_ood,
            )

            risk_level, recommendation = (
                self.recommendation.evaluate(
                    trust_result.trust_score
                )
            )

            return PredictionReport(
                prediction=prediction[0],
                confidence=confidence,
                calibrated_confidence=confidence,
                uncertainty=entropy,
                ood=is_ood,
                ood_score=ood_score,
                drift_score=0.0,
                trust_score=trust_result.trust_score,
                risk_level=risk_level,
                recommendation=recommendation,
            )

        except Exception as exc:
            raise PredictionError(
                f"Prediction pipeline failed: {exc}"
            ) from exc

    def __call__(self, X):
        """
        Shortcut for predict().
        """
        return self.predict(X)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"wrapper={self.wrapper})"
        )