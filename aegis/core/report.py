"""
Prediction reports for AegisAI.

A PredictionReport contains the observable reliability results for a
single model prediction.

The report intentionally contains only metrics that AegisAI currently
computes. Experimental or unavailable metrics must not be represented
as if they were measured.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aegis.core.enums import Recommendation, RiskLevel


class PredictionReport(BaseModel):
    """Structured reliability report for a model prediction.

    Attributes:
        prediction:
            The underlying model prediction.

        confidence:
            Maximum predicted class probability in [0, 1].

        uncertainty:
            Normalized predictive entropy in [0, 1].
            Higher values indicate greater uncertainty.

        ood:
            Whether the input was classified as out-of-distribution.

        ood_score:
            Normalized OOD risk in [0, 1].
            This is derived from the raw OOD detector distance.

        trust_score:
            Aggregated reliability score in [0, 1].
            Higher values indicate greater trust.

        risk_level:
            Policy-derived risk classification.

        recommendation:
            Policy-derived action recommendation.

        metadata:
            Additional diagnostic information that does not form part
            of the core reliability contract.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # ==========================================================
    # Prediction
    # ==========================================================

    prediction: Any

    # ==========================================================
    # Reliability Metrics
    # ==========================================================

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Maximum predicted class probability."
        ),
    )

    uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized predictive entropy."
        ),
    )

    # ==========================================================
    # OOD
    # ==========================================================

    ood: bool = Field(
        description=(
            "Whether the input was classified as "
            "out-of-distribution."
        ),
    )

    ood_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized out-of-distribution risk."
        ),
    )

    # ==========================================================
    # Trust Decision
    # ==========================================================

    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Aggregated reliability trust score."
        ),
    )

    risk_level: RiskLevel

    recommendation: Recommendation

    # ==========================================================
    # Metadata
    # ==========================================================

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional diagnostic information."
        ),
    )

    # ==========================================================
    # Serialization
    # ==========================================================

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a Python dictionary."""
        return self.model_dump()

    def to_json(
        self,
        indent: int = 4,
    ) -> str:
        """Return the report as formatted JSON.

        Args:
            indent:
                Number of spaces used for JSON indentation.

        Returns:
            JSON representation of the report.
        """
        if indent < 0:
            raise ValueError(
                "JSON indentation cannot be negative."
            )

        return self.model_dump_json(
            indent=indent,
        )

    # ==========================================================
    # Human-readable Output
    # ==========================================================

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        return (
            f"Prediction={self.prediction}, "
            f"Confidence={self.confidence:.3f}, "
            f"Uncertainty={self.uncertainty:.3f}, "
            f"OOD={self.ood}, "
            f"OOD Risk={self.ood_score:.3f}, "
            f"Trust={self.trust_score:.3f}, "
            f"Risk={self.risk_level.value}, "
            f"Recommendation={self.recommendation.value}"
        )

    def __str__(self) -> str:
        """Return the report as formatted JSON."""
        return self.to_json()