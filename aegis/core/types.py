"""Shared type definitions for the AegisAI framework."""

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
# Model Data Types
# =============================================================================

FeatureVector: TypeAlias = FloatArray

ProbabilityVector: TypeAlias = FloatArray

Prediction: TypeAlias = Any

PredictionBatch: TypeAlias = Sequence[Prediction]

Label: TypeAlias = Prediction


# =============================================================================
# Model Input
# =============================================================================

ModelInput: TypeAlias = (
    FloatArray
    | NDArray[Any]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | pd.DataFrame
    | pd.Series
)


# =============================================================================
# Model Output
# =============================================================================

ModelOutput: TypeAlias = Any


# =============================================================================
# Serialization Types
# =============================================================================

JsonDict: TypeAlias = dict[str, Any]

JsonList: TypeAlias = list[Any]

Metadata: TypeAlias = Mapping[str, Any]


# =============================================================================
# Reliability Scores
# =============================================================================

Score: TypeAlias = float

Confidence: TypeAlias = float

Entropy: TypeAlias = float

TrustScore: TypeAlias = float

OODScore: TypeAlias = float

DriftScore: TypeAlias = float


# =============================================================================
# Enum Value Types
# =============================================================================

RiskValue: TypeAlias = str

RecommendationValue: TypeAlias = str