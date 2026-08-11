"""
Version information for AegisAI.

This module defines the canonical version of the framework.
All internal modules should import the version from here rather
than hardcoding version strings.
"""

from __future__ import annotations

from typing import Final

#: Current package version following Semantic Versioning (SemVer).
__version__: Final[str] = "0.1.0"

#: Human-readable package name.
PACKAGE_NAME: Final[str] = "AegisAI"

#: Short framework description.
DESCRIPTION: Final[str] = (
    "Production-grade AI Reliability & Trust Framework."
)