"""
Configurable weights for AegisAI trust scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.config import settings
from aegis.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for normalized reliability signals.

    Signal semantics:

        confidence:
            Higher confidence improves trust.

        uncertainty:
            Higher uncertainty reduces trust.

        ood:
            Higher OOD risk reduces trust.

        drift:
            Higher drift reduces trust.

    If a value is omitted, the centralized framework configuration
    is used.
    """

    confidence: float | None = None
    uncertainty: float | None = None
    ood: float | None = None
    drift: float | None = None

    def __post_init__(self) -> None:
        """Load default values from the global configuration."""

        if self.confidence is None:
            object.__setattr__(
                self,
                "confidence",
                settings.confidence_weight,
            )

        if self.uncertainty is None:
            object.__setattr__(
                self,
                "uncertainty",
                settings.uncertainty_weight,
            )

        if self.ood is None:
            object.__setattr__(
                self,
                "ood",
                settings.ood_weight,
            )

        if self.drift is None:
            object.__setattr__(
                self,
                "drift",
                settings.drift_weight,
            )

    def normalize(self) -> "ScoreWeights":
        """Return a new set of weights normalized to sum to one.

        Raises:
            ConfigurationError:
                If a weight is invalid or all weights are zero.
        """
        values = {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "ood": self.ood,
            "drift": self.drift,
        }

        normalized_values: dict[str, float] = {}

        for name, value in values.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Score weight '{name}' must be numeric."
                ) from exc

            if not np.isfinite(numeric_value):
                raise ConfigurationError(
                    f"Score weight '{name}' must be finite."
                )

            if numeric_value < 0.0:
                raise ConfigurationError(
                    f"Score weight '{name}' cannot be negative."
                )

            normalized_values[name] = numeric_value

        total = sum(normalized_values.values())

        if total <= 0.0:
            raise ConfigurationError(
                "At least one score weight must be greater than zero."
            )

        return ScoreWeights(
            confidence=(
                normalized_values["confidence"] / total
            ),
            uncertainty=(
                normalized_values["uncertainty"] / total
            ),
            ood=(
                normalized_values["ood"] / total
            ),
            drift=(
                normalized_values["drift"] / total
            ),
        )

    def is_drift_enabled(self) -> bool:
        """Return whether drift contributes to the score."""
        return float(self.drift) > 0.0

    def as_dict(self) -> dict[str, float]:
        """Return the configured weights as a dictionary."""
        return {
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "ood": float(self.ood),
            "drift": float(self.drift),
        }

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"confidence={self.confidence}, "
            f"uncertainty={self.uncertainty}, "
            f"ood={self.ood}, "
            f"drift={self.drift})"
        )