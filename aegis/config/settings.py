"""
Global configuration for AegisAI.

This module defines the configurable framework settings used across
AegisAI.

Configuration is centralized here so modules do not hardcode
thresholds or scoring weights.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameworkSettings(BaseModel):
    """
    Global configuration for the AegisAI framework.

    The settings object is validated at creation time and whenever
    an attribute is changed.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    # ==========================================================
    # Confidence
    # ==========================================================

    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum model confidence considered acceptable "
            "for reliability analysis."
        ),
    )

    # Calibration is not implemented yet.
    calibration_enabled: bool = False

    # ==========================================================
    # Uncertainty
    # ==========================================================

    entropy_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum normalized prediction entropy considered "
            "acceptable."
        ),
    )

    # ==========================================================
    # Out-of-Distribution Detection
    # ==========================================================

    ood_zscore_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Maximum absolute feature Z-score threshold used "
            "to classify a sample as OOD."
        ),
    )

    # ==========================================================
    # Trust Score Weights
    # ==========================================================

    confidence_weight: float = Field(
        default=0.50,
        ge=0.0,
    )

    uncertainty_weight: float = Field(
        default=0.30,
        ge=0.0,
    )

    ood_weight: float = Field(
        default=0.20,
        ge=0.0,
    )

    drift_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Drift contribution. Disabled by default because "
            "the current pipeline does not calculate live drift."
        ),
    )

    # ==========================================================
    # Recommendation Thresholds
    # ==========================================================

    auto_approve_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )

    review_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    # ==========================================================
    # Runtime
    # ==========================================================

    random_seed: int = 42

    enable_logging: bool = True

    enable_monitoring: bool = False

    debug: bool = False

    # ==========================================================
    # Cross-field Validation
    # ==========================================================

    @model_validator(mode="after")
    def validate_configuration(self) -> "FrameworkSettings":
        """Validate relationships between configuration values."""

        if self.review_threshold >= self.auto_approve_threshold:
            raise ValueError(
                "review_threshold must be lower than "
                "auto_approve_threshold."
            )

        weights = (
            self.confidence_weight,
            self.uncertainty_weight,
            self.ood_weight,
            self.drift_weight,
        )

        if sum(weights) <= 0.0:
            raise ValueError(
                "At least one trust-score weight must be greater "
                "than zero."
            )

        return self


settings = FrameworkSettings()
"""
Singleton configuration instance.

Import this object throughout AegisAI:

    from aegis.config import settings
"""