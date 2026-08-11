"""
Framework-wide constants for AegisAI.

This module contains immutable values that define framework behavior,
metadata, report keys, and supported model families.

Unlike `settings.py`, these values are not intended to be modified
at runtime.
"""

from __future__ import annotations

from typing import Final

# ==========================================================
# Framework Metadata
# ==========================================================

FRAMEWORK_NAME: Final[str] = "AegisAI"

FRAMEWORK_DESCRIPTION: Final[str] = (
    "Production-grade AI Reliability & Trust Framework"
)

# ==========================================================
# Risk Levels
# ==========================================================

RISK_LOW: Final[str] = "LOW"
RISK_MEDIUM: Final[str] = "MEDIUM"
RISK_HIGH: Final[str] = "HIGH"

RISK_LEVELS: Final[tuple[str, ...]] = (
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
)

# ==========================================================
# Recommendation Labels
# ==========================================================

AUTO_APPROVE: Final[str] = "Auto Approve"

HUMAN_REVIEW: Final[str] = "Requires Human Review"

REJECT: Final[str] = "Reject Prediction"

# ==========================================================
# Supported Wrapper Types
# ==========================================================

SKLEARN: Final[str] = "scikit-learn"
PYTORCH: Final[str] = "pytorch"
TENSORFLOW: Final[str] = "tensorflow"
XGBOOST: Final[str] = "xgboost"
CATBOOST: Final[str] = "catboost"
LIGHTGBM: Final[str] = "lightgbm"
HUGGINGFACE: Final[str] = "huggingface"
OLLAMA: Final[str] = "ollama"
OPENAI: Final[str] = "openai"

SUPPORTED_WRAPPERS: Final[tuple[str, ...]] = (
    SKLEARN,
    PYTORCH,
    TENSORFLOW,
    XGBOOST,
    CATBOOST,
    LIGHTGBM,
    HUGGINGFACE,
    OLLAMA,
    OPENAI,
)

# ==========================================================
# Default Report Keys
# ==========================================================

PREDICTION_KEY: Final[str] = "prediction"
CONFIDENCE_KEY: Final[str] = "confidence"
CALIBRATED_CONFIDENCE_KEY: Final[str] = "calibrated_confidence"
UNCERTAINTY_KEY: Final[str] = "uncertainty"
OOD_KEY: Final[str] = "ood"
OOD_SCORE_KEY: Final[str] = "ood_score"
DRIFT_SCORE_KEY: Final[str] = "drift_score"
TRUST_SCORE_KEY: Final[str] = "trust_score"
RISK_LEVEL_KEY: Final[str] = "risk_level"
RECOMMENDATION_KEY: Final[str] = "recommendation"

# ==========================================================
# Safety Report Keys
# ==========================================================

SAFETY_KEY: Final[str] = "safety"
PASSED_KEY: Final[str] = "passed"
TOXICITY_KEY: Final[str] = "toxicity"
BIAS_KEY: Final[str] = "bias"
PII_KEY: Final[str] = "pii"
PROMPT_INJECTION_KEY: Final[str] = "prompt_injection"
HALLUCINATION_KEY: Final[str] = "hallucination"

# ==========================================================
# Numerical Defaults
# ==========================================================

EPSILON: Final[float] = 1e-12

DEFAULT_RANDOM_SEED: Final[int] = 42

DEFAULT_DECIMALS: Final[int] = 4