"""Enumerations used throughout the AegisAI framework."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Base class for string-valued enums."""

    def __str__(self) -> str:
        return self.value


class RiskLevel(StrEnum):
    """Risk level assigned to a prediction."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(StrEnum):
    """Action recommended for a prediction."""

    AUTO_APPROVE = "Auto Approve"
    HUMAN_REVIEW = "Requires Human Review"
    REJECT = "Reject Prediction"


class ModelType(StrEnum):
    """Model families supported by AegisAI."""

    SKLEARN = "scikit-learn"


class PredictionStatus(StrEnum):
    """Status of a prediction operation."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    WARNING = "WARNING"


class WrapperCapability(StrEnum):
    """Capabilities supported by a model wrapper."""

    PREDICT = "predict"
    PREDICT_PROBA = "predict_proba"
    DECISION_FUNCTION = "decision_function"