"""
Statistical distance and divergence metrics.

These utilities are used across:
    - Drift detection
    - Trust scoring
    - Monitoring systems

All functions are designed to be:
    - Numerically stable
    - Dependency-minimal
    - Reusable across the framework
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy, wasserstein_distance


# -----------------------------
# Utility helpers
# -----------------------------

def _normalize(p: np.ndarray) -> np.ndarray:
    """
    Normalize array into probability distribution.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / np.sum(p)


# -----------------------------
# KL Divergence
# -----------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence: KL(p || q)

    Args:
        p: Reference distribution
        q: Current distribution

    Returns:
        KL divergence value
    """
    p = _normalize(p)
    q = _normalize(q)

    return float(entropy(p, q))


# -----------------------------
# Jensen-Shannon Divergence
# -----------------------------

def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL).

    Returns:
        Value in [0, 1] approximately
    """
    p = _normalize(p)
    q = _normalize(q)

    m = 0.5 * (p + q)

    return float(
        0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    )


# -----------------------------
# Wasserstein Distance
# -----------------------------

def wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Wasserstein (Earth Mover's) distance.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    return float(wasserstein_distance(p, q))


# -----------------------------
# Population Stability Index (PSI)
# -----------------------------

def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index.

    Commonly used in credit risk and production ML monitoring.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    breakpoints = np.linspace(0, 1, bins + 1)

    expected_counts, _ = np.histogram(
        expected, bins=breakpoints
    )
    actual_counts, _ = np.histogram(
        actual, bins=breakpoints
    )

    expected_dist = expected_counts / np.sum(expected_counts)
    actual_dist = actual_counts / np.sum(actual_counts)

    expected_dist = np.clip(expected_dist, 1e-6, None)
    actual_dist = np.clip(actual_dist, 1e-6, None)

    psi_value = np.sum(
        (actual_dist - expected_dist)
        * np.log(actual_dist / expected_dist)
    )

    return float(psi_value)


# -----------------------------
# Helper summary function
# -----------------------------

def drift_summary(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    """
    Compute multiple drift metrics at once.
    """
    return {
        "kl_divergence": kl_divergence(reference, current),
        "js_divergence": jensen_shannon_divergence(reference, current),
        "wasserstein": wasserstein(reference, current),
    }