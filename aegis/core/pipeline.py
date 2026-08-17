"""
Prediction pipeline for AegisAI.

The prediction pipeline orchestrates model inference and reliability
analysis without owning business decision policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aegis.core.exceptions import PredictionError
from aegis.core.report import PredictionReport
from aegis.ood.zscore import ZScoreOODDetector
from aegis.scoring.trust import TrustScorer
from aegis.uncertainty.confidence import ConfidenceEstimator
from aegis.uncertainty.entropy import EntropyEstimator
from aegis.wrappers.base import BaseModelWrapper


class PredictionPipeline:
    """
    Orchestrate model prediction and reliability analysis.

    The pipeline is responsible for:

        model prediction
        probability validation
        confidence
        uncertainty
        OOD detection
        trust scoring
        report generation

    Recommendation policy is delegated to ``TrustScorer`` and its
    ``RecommendationEngine``.
    """

    def __init__(
        self,
        wrapper: BaseModelWrapper,
        ood_detector: ZScoreOODDetector | None = None,
        trust_scorer: TrustScorer | None = None,
    ) -> None:
        """Initialize the prediction pipeline.

        Args:
            wrapper:
                Model wrapper responsible for prediction.

            ood_detector:
                Optional OOD detector.

            trust_scorer:
                Optional trust scoring engine.
        """
        self.wrapper = wrapper

        self.confidence = ConfidenceEstimator()
        self.entropy = EntropyEstimator()

        self.ood = (
            ood_detector
            if ood_detector is not None
            else ZScoreOODDetector()
        )

        self.trust = (
            trust_scorer
            if trust_scorer is not None
            else TrustScorer()
        )

        self.wrapper.validate()

        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether AegisAI reference components are fitted."""
        return self._fitted

    def fit(self, X_train: Any) -> "PredictionPipeline":
        """
        Fit AegisAI reliability components using reference data.

        The underlying ML model is not trained here. The supplied
        model is expected to already be fitted.

        Args:
            X_train:
                Reference training features used by the OOD detector.

        Returns:
            The fitted prediction pipeline.
        """
        try:
            X = np.asarray(X_train)

            if X.size == 0:
                raise PredictionError(
                    "Reference training data cannot be empty."
                )

            self.ood.fit(X_train)

        except PredictionError:
            raise

        except Exception as exc:
            raise PredictionError(
                f"Failed to fit prediction pipeline: {exc}"
            ) from exc

        self._fitted = True

        return self

    def predict(
        self,
        X: Any,
    ) -> PredictionReport:
        """
        Return a reliability report for the first input sample.

        For multiple samples, use ``predict_batch``.
        """
        reports = self.predict_batch(X)

        if not reports:
            raise PredictionError(
                "Prediction input contains no samples."
            )

        return reports[0]

    def predict_batch(
        self,
        X: Any,
    ) -> list[PredictionReport]:
        """
        Return one reliability report for every input sample.
        """
        self._require_fitted()

        try:
            predictions = self._validate_predictions(
                self.wrapper.predict(X)
            )

            probabilities = self._validate_probabilities(
                self.wrapper.predict_proba(X),
                expected_samples=len(predictions),
            )

            confidence_values = np.asarray(
                self.confidence.compute(probabilities),
                dtype=np.float64,
            )

            uncertainty_values = np.asarray(
                self.entropy.compute(probabilities),
                dtype=np.float64,
            )

            raw_ood_scores = np.asarray(
                self.ood.score(X),
                dtype=np.float64,
            )

            ood_flags = np.asarray(
                self.ood.predict(X),
                dtype=bool,
            )

            self._validate_metric_lengths(
                expected_samples=len(predictions),
                confidence=confidence_values,
                uncertainty=uncertainty_values,
                ood_scores=raw_ood_scores,
                ood_flags=ood_flags,
            )

            threshold = self._get_ood_threshold()

            reports: list[PredictionReport] = []

            for index, prediction in enumerate(predictions):
                confidence = float(
                    confidence_values[index]
                )

                uncertainty = float(
                    uncertainty_values[index]
                )

                raw_ood_score = float(
                    raw_ood_scores[index]
                )

                is_ood = bool(
                    ood_flags[index]
                )

                normalized_ood_risk = (
                    self._normalize_ood_score(
                        raw_ood_score,
                        threshold,
                    )
                )

                trust_result = self.trust.compute(
                    confidence=confidence,
                    entropy=uncertainty,
                    ood_score=normalized_ood_risk,
                    is_ood=is_ood,
                )

                reports.append(
                    PredictionReport(
                        prediction=self._to_python_value(
                            prediction
                        ),
                        confidence=confidence,
                        uncertainty=uncertainty,
                        ood=is_ood,
                        ood_score=normalized_ood_risk,
                        trust_score=trust_result.trust_score,
                        risk_level=trust_result.risk_level,
                        recommendation=(
                            trust_result.recommendation
                        ),
                        metadata={
                            "ood_raw_score": raw_ood_score,
                            "ood_threshold": threshold,
                        },
                    )
                )

            return reports

        except PredictionError:
            raise

        except Exception as exc:
            raise PredictionError(
                f"Batch prediction failed: {exc}"
            ) from exc

    def __call__(
        self,
        X: Any,
    ) -> PredictionReport:
        """Run prediction analysis."""
        return self.predict(X)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"wrapper={self.wrapper!r}, "
            f"ood={self.ood!r}, "
            f"trust={self.trust!r}, "
            f"fitted={self.is_fitted})"
        )

    def _require_fitted(self) -> None:
        """Ensure the reliability pipeline has been fitted."""
        if not self._fitted:
            raise PredictionError(
                "AegisAI has not been fitted. "
                "Call fit(X_train) before predict()."
            )

    @staticmethod
    def _validate_predictions(
        predictions: Any,
    ) -> np.ndarray:
        """Validate and normalize model predictions."""
        values = np.asarray(predictions)

        if values.ndim == 0:
            values = values.reshape(1)

        if values.ndim != 1:
            raise PredictionError(
                "Model predictions must be a one-dimensional array."
            )

        if len(values) == 0:
            raise PredictionError(
                "Model returned no predictions."
            )

        return values

    @staticmethod
    def _validate_probabilities(
        probabilities: Any,
        expected_samples: int,
    ) -> np.ndarray:
        """Validate model probability output."""
        values = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise PredictionError(
                "Model probabilities must have shape "
                "(n_samples, n_classes)."
            )

        if values.shape[0] != expected_samples:
            raise PredictionError(
                "Prediction and probability outputs contain "
                "different numbers of samples."
            )

        if values.shape[1] < 2:
            raise PredictionError(
                "Classification probabilities must contain "
                "at least two classes."
            )

        if not np.all(np.isfinite(values)):
            raise PredictionError(
                "Model probabilities contain NaN or infinite values."
            )

        if np.any(values < 0.0) or np.any(values > 1.0):
            raise PredictionError(
                "Model probabilities must be within [0, 1]."
            )

        row_sums = values.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            raise PredictionError(
                "Each probability row must sum to 1."
            )

        return values

    @staticmethod
    def _validate_metric_lengths(
        expected_samples: int,
        confidence: np.ndarray,
        uncertainty: np.ndarray,
        ood_scores: np.ndarray,
        ood_flags: np.ndarray,
    ) -> None:
        """Ensure all reliability metrics contain one value per sample."""
        metrics = {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "OOD scores": ood_scores,
            "OOD flags": ood_flags,
        }

        for name, values in metrics.items():
            if values.ndim != 1:
                raise PredictionError(
                    f"{name} must contain one value per sample."
                )

            if len(values) != expected_samples:
                raise PredictionError(
                    f"{name} contains {len(values)} values, "
                    f"but {expected_samples} samples were predicted."
                )

    def _get_ood_threshold(self) -> float:
        """Return and validate the configured OOD threshold."""
        try:
            threshold = float(self.ood.threshold)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                "OOD detector threshold must be numeric."
            ) from exc

        if not np.isfinite(threshold) or threshold <= 0.0:
            raise PredictionError(
                "OOD detector threshold must be greater than zero."
            )

        return threshold

    @staticmethod
    def _normalize_ood_score(
        raw_score: float,
        threshold: float,
    ) -> float:
        """
        Convert an unbounded OOD distance into [0, 1] risk.

        Values at or above the configured threshold represent
        maximum normalized OOD risk.
        """
        if not np.isfinite(raw_score):
            raise PredictionError(
                "OOD detector returned a NaN or infinite score."
            )

        if raw_score < 0.0:
            raise PredictionError(
                "OOD detector returned a negative score."
            )

        return float(
            min(
                raw_score / threshold,
                1.0,
            )
        )

    @staticmethod
    def _to_python_value(
        value: Any,
    ) -> Any:
        """Convert NumPy scalar values into native Python values."""
        if isinstance(value, np.generic):
            return value.item()

        return value