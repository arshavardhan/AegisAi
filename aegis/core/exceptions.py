"""Exception hierarchy for the AegisAI framework.

All framework-specific exceptions inherit from ``AegisError``.
This allows users to catch either a specific failure or all
AegisAI failures through the common base exception.
"""

from __future__ import annotations


class AegisError(Exception):
    """Base exception for all AegisAI errors."""

    default_message = "An unknown AegisAI error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = (
            message
            if message is not None
            else self.default_message
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ConfigurationError(AegisError):
    """Raised when AegisAI configuration is invalid."""

    default_message = "Invalid framework configuration."


class ValidationError(AegisError):
    """Raised when input or component validation fails."""

    default_message = "Input validation failed."


class ModelCompatibilityError(AegisError):
    """Raised when a model is unsupported or incompatible."""

    default_message = (
        "The supplied model is not compatible with AegisAI."
    )


class PredictionError(AegisError):
    """Raised when model prediction or prediction analysis fails."""

    default_message = "The prediction process failed."


class WrapperError(AegisError):
    """Raised when a model wrapper encounters an error."""

    default_message = "A model wrapper error occurred."


class ConfidenceError(AegisError):
    """Raised when confidence estimation fails."""

    default_message = "Confidence estimation failed."


class EntropyError(AegisError):
    """Raised when entropy calculation fails."""

    default_message = "Entropy calculation failed."


class OODDetectionError(AegisError):
    """Raised when out-of-distribution detection fails."""

    default_message = "OOD detection failed."


class TrustScoreError(AegisError):
    """Raised when trust-score calculation fails."""

    default_message = "Trust score calculation failed."


class ReportGenerationError(AegisError):
    """Raised when a prediction report cannot be generated."""

    default_message = "Report generation failed."


class PluginError(AegisError):
    """Raised when a plugin fails to load or execute."""

    default_message = "Plugin execution failed."


__all__ = [
    "AegisError",
    "ConfigurationError",
    "ValidationError",
    "ModelCompatibilityError",
    "PredictionError",
    "WrapperError",
    "ConfidenceError",
    "EntropyError",
    "OODDetectionError",
    "TrustScoreError",
    "ReportGenerationError",
    "PluginError",
]