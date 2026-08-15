"""Prediction pipeline for AegisAI.

The pipeline orchestrates model inference and reliability analysis for
classification models with probability support.
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
    """Main AegisAI prediction pipeline."""

    def __init__(
        self,
        wrapper: BaseModelWrapper,
        ood_detector: ZScoreOODDetector | None = None,
        trust_scorer: TrustScorer | None = None,
        recommendation: RecommendationEngine | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.confidence = ConfidenceEstimator()
        self.entropy = EntropyEstimator()
        self.ood = ood_detector or ZScoreOODDetector()
        self.trust = trust_scorer or TrustScorer()
        self.recommendation = recommendation or RecommendationEngine()

        self.wrapper.validate()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether AegisAI reference components have been fitted."""
        return self._fitted

    def fit(self, X_train) -> None:
        """Fit AegisAI reference components using training features."""
        self.ood.fit(X_train)
        self._fitted = True

    def predict(self, X) -> PredictionReport:
        """Execute reliability analysis for one or more input samples.

        The returned report currently represents the first sample. Use
        ``predict_batch`` on ``AegisModel`` when individual reports are
        required for every sample.
        """
        if not self._fitted:
            raise PredictionError(
                "AegisAI has not been fitted. Call fit(X_train) before predict()."
            )

        try:
            prediction = np.asarray(self.wrapper.predict(X))
            probabilities = np.asarray(self.wrapper.predict_proba(X), dtype=np.float64)

            if prediction.shape[0] != probabilities.shape[0]:
                raise PredictionError(
                    "Prediction and probability outputs contain different numbers of samples."
                )

            confidence_values = self.confidence.compute(probabilities)
            entropy_values = self.entropy.compute(probabilities)
            raw_ood_scores = self.ood.score(X)
            ood_flags = self.ood.predict(X)

            confidence = float(confidence_values[0])
            entropy = float(entropy_values[0])
            raw_ood_score = float(raw_ood_scores[0])
            is_ood = bool(ood_flags[0])

            # Convert an unbounded maximum Z-score into a bounded risk
            # value. A score at or above the configured threshold maps to
            # maximum OOD risk; values below it scale proportionally.
            threshold = float(self.ood.threshold)
            ood_score = min(raw_ood_score / threshold, 1.0) if threshold > 0 else 1.0

            trust_result = self.trust.compute(
                confidence=confidence,
                entropy=entropy,
                ood_score=ood_score,
                is_ood=is_ood,
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
                risk_level=trust_result.risk_level,
                recommendation=trust_result.recommendation,
                metadata={
                    "ood_raw_zscore": raw_ood_score,
                    "ood_threshold": threshold,
                },
            )

        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Prediction pipeline failed: {exc}") from exc

    def predict_batch(self, X) -> list[PredictionReport]:
        """Return one reliability report per input sample."""
        if not self._fitted:
            raise PredictionError(
                "AegisAI has not been fitted. Call fit(X_train) before predict_batch()."
            )

        try:
            predictions = np.asarray(self.wrapper.predict(X))
            probabilities = np.asarray(self.wrapper.predict_proba(X), dtype=np.float64)
            raw_ood_scores = self.ood.score(X)
            ood_flags = self.ood.predict(X)
            confidence_values = self.confidence.compute(probabilities)
            entropy_values = self.entropy.compute(probabilities)
            threshold = float(self.ood.threshold)

            reports: list[PredictionReport] = []
            for index, prediction in enumerate(predictions):
                raw_ood = float(raw_ood_scores[index])
                is_ood = bool(ood_flags[index])
                ood_score = min(raw_ood / threshold, 1.0) if threshold > 0 else 1.0
                trust_result = self.trust.compute(
                    confidence=float(confidence_values[index]),
                    entropy=float(entropy_values[index]),
                    ood_score=ood_score,
                    is_ood=is_ood,
                )
                reports.append(
                    PredictionReport(
                        prediction=prediction,
                        confidence=float(confidence_values[index]),
                        calibrated_confidence=float(confidence_values[index]),
                        uncertainty=float(entropy_values[index]),
                        ood=is_ood,
                        ood_score=ood_score,
                        drift_score=0.0,
                        trust_score=trust_result.trust_score,
                        risk_level=trust_result.risk_level,
                        recommendation=trust_result.recommendation,
                        metadata={
                            "ood_raw_zscore": raw_ood,
                            "ood_threshold": threshold,
                        },
                    )
                )
            return reports
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Batch prediction failed: {exc}") from exc

    def __call__(self, X) -> PredictionReport:
        return self.predict(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(wrapper={self.wrapper})"
