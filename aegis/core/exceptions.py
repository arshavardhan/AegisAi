"""
Custom exceptions for the AegisAI framework.

This module defines the complete exception hierarchy used throughout
the framework. All AegisAI-specific exceptions inherit from
`AegisError`, allowing users to catch framework-specific errors
without affecting unrelated exceptions.
"""

from __future__ import annotations


class AegisError(Exception):
    """
    Base exception for all AegisAI errors.

    Users can catch this exception to handle all framework-specific
    failures.

    Example:
        try:
            report = model.predict(sample)
        except AegisError as exc:
            print(exc)
    """

    default_message = "An unknown AegisAI error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ConfigurationError(AegisError):
    """
    Raised when the framework configuration is invalid.
    """

    default_message = "Invalid framework configuration."


class ValidationError(AegisError):
    """
    Raised when input validation fails.
    """

    default_message = "Input validation failed."


class ModelCompatibilityError(AegisError):
    """
    Raised when an unsupported or incompatible model is supplied.
    """

    default_message = (
        "The supplied model is not compatible with AegisAI."
    )


class PredictionError(AegisError):
    """
    Raised when model prediction fails.
    """

    default_message = (
        "The prediction process failed."
    )


class WrapperError(AegisError):
    """
    Raised when a model wrapper encounters an error.
    """

    default_message = (
        "A model wrapper error occurred."
    )


class ConfidenceError(AegisError):
    """
    Raised when confidence estimation cannot be computed.
    """

    default_message = (
        "Confidence estimation failed."
    )


class EntropyError(AegisError):
    """
    Raised when entropy computation fails.
    """

    default_message = (
        "Entropy calculation failed."
    )


class OODDetectionError(AegisError):
    """
    Raised when Out-of-Distribution detection fails.
    """

    default_message = (
        "OOD detection failed."
    )


class TrustScoreError(AegisError):
    """
    Raised when trust score computation fails.
    """

    default_message = (
        "Trust score calculation failed."
    )


class ReportGenerationError(AegisError):
    """
    Raised when a prediction report cannot be generated.
    """

    default_message = (
        "Report generation failed."
    )


class PluginError(AegisError):
    """
    Raised when a plugin fails to load or execute.
    """

    default_message = (
        "Plugin execution failed."
    )