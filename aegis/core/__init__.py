"""
Core components of the AegisAI framework.

This package contains the fundamental building blocks used throughout
AegisAI, including:

- AegisModel
- Prediction Pipeline
- Report Models
- Exceptions
- Enums
- Shared Types

Most internal modules should import shared classes from this package
rather than referencing individual implementation files directly.
"""

from .model import AegisModel
from .report import PredictionReport, SafetyReport
from .exceptions import (
    AegisError,
    ConfigurationError,
    ModelCompatibilityError,
    PredictionError,
    ValidationError,
)

__all__ = [
    "AegisModel",
    "PredictionReport",
    "SafetyReport",
    "AegisError",
    "ConfigurationError",
    "ModelCompatibilityError",
    "PredictionError",
    "ValidationError",
]