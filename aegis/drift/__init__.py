"""
Data drift detection package.

This package contains algorithms for detecting distribution shifts
between training and inference data.

Modules:
    - base
    - data_drift
    - concept_drift
    - metrics
"""

from aegis.drift.base import BaseDriftDetector

__all__ = [
    "BaseDriftDetector",
]