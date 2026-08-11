"""
Global configuration for AegisAI.

This module defines all configurable framework settings.

No module inside AegisAI should hardcode thresholds or magic numbers.
Instead, import the shared `settings` instance.

Example:
    from aegis.config import settings

    if confidence < settings.confidence_threshold:
        ...
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrameworkSettings(BaseModel):
    """
    Global configuration for the AegisAI framework.

    This object stores all configurable thresholds and runtime
    options used throughout the framework.
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
        description="Minimum confidence required for automatic approval.",
    )

    calibration_enabled: bool = True

    # ==========================================================
    # Uncertainty
    # ==========================================================

    entropy_threshold: float = Field(
        default=0.50,
        ge=0.0,
        description="Maximum acceptable prediction entropy.",
    )

    # ==========================================================
    # Out-of-Distribution Detection
    # ==========================================================

    ood_zscore_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Z-score threshold for OOD detection.",
    )

    # ==========================================================
    # Trust Score Weights
    # ==========================================================

    confidence_weight: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    entropy_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )

    ood_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
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


settings = FrameworkSettings()
"""
Singleton configuration instance.

Import this object throughout the framework:

    from aegis.config import settings
"""