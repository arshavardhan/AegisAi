"""
Shared type definitions for the AegisAI framework.

This module centralizes commonly used type aliases to improve
readability, consistency, and maintainability across the codebase.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# =============================================================================
# NumPy Types
# =============================================================================

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

# =============================================================================
# Generic ML Data Types
# =============================================================================

FeatureVector: TypeAlias = FloatArray

ProbabilityVector: TypeAlias = FloatArray

Prediction: TypeAlias = int | float | str | bool

PredictionBatch: TypeAlias = Sequence[Prediction]

Label: TypeAlias = Prediction

# =============================================================================
# Supported Input Types
# =============================================================================

ModelInput: TypeAlias = (
    FloatArray
    | Sequence[float]
    | Sequence[Sequence[float]]
    | pd.DataFrame
    | pd.Series
)

# =============================================================================
# Generic Output Types
# =============================================================================

ModelOutput: TypeAlias = Any

JsonDict: TypeAlias = dict[str, Any]

JsonList: TypeAlias = list[Any]

Metadata: TypeAlias = Mapping[str, Any]

# =============================================================================
# Report Types
# =============================================================================

Score: TypeAlias = float

Confidence: TypeAlias = float

Entropy: TypeAlias = float

TrustScore: TypeAlias = float

OODScore: TypeAlias = float

DriftScore: TypeAlias = float

RiskValue: TypeAlias = str

RecommendationValue: TypeAlias = str