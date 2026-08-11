"""
Configuration package for AegisAI.

This package contains all configurable settings, constants,
and framework-wide defaults.

Nothing in the framework should hardcode thresholds or
configuration values. Instead, import them from this package.
"""

from .settings import FrameworkSettings, settings
from .constants import *

__all__ = [
    "FrameworkSettings",
    "settings",
]