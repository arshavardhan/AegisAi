"""
AegisAI
=======

A production-grade AI Reliability & Trust Framework.

AegisAI provides a unified interface for evaluating the reliability,
confidence, uncertainty, and trustworthiness of predictions generated
by machine learning models and large language models.

Typical usage:

    from aegis import AegisModel

    model = AegisModel(trained_model)
    report = model.predict(sample)

Only the public API is exported from this module.
"""

from .version import __version__
from .core.model import AegisModel

__all__ = [
    "AegisModel",
    "__version__",
]