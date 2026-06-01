# src/riskquant/config/namespace.py
"""
Global ID Namespace definitions and validation contracts.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from riskquant.exceptions import ConfigurationError

# Expand to 100 Million per block
_NODE_MIN_ID = 0
_NODE_MAX_ID = 99_999_999

_CURVE_MIN_ID = 100_000_000
_CURVE_MAX_ID = 199_999_999

_SURFACE_MIN_ID = 200_000_000
_SURFACE_MAX_ID = 299_999_999


class IdNamespaceBounds(NamedTuple):
    min_id: int
    max_id: int
    name: str


assert (
    _NODE_MIN_ID
    < _NODE_MAX_ID
    < _CURVE_MIN_ID
    < _CURVE_MAX_ID
    < _SURFACE_MIN_ID
    < _SURFACE_MAX_ID
), "ID namespace boundaries must be strictly increasing and non-overlapping."


class GlobalIdNamespace:
    """Global ID boundaries for RiskQuant execution mapping."""

    NODES = IdNamespaceBounds(
        min_id=_NODE_MIN_ID,
        max_id=_NODE_MAX_ID,
        name="Nodes (Risk Factors, dimension = 0)",
    )
    CURVES = IdNamespaceBounds(
        min_id=_CURVE_MIN_ID,
        max_id=_CURVE_MAX_ID,
        name="Curves (Containers, dimension = 1)",
    )
    SURFACES = IdNamespaceBounds(
        min_id=_SURFACE_MIN_ID,
        max_id=_SURFACE_MAX_ID,
        name="Surfaces (Containers, dimension = 2)",
    )


def enforce_id_namespace(
    *,
    ids: npt.NDArray[np.int32],
    bounds: IdNamespaceBounds,
) -> None:
    """
    Vectorized assertion that all IDs fall within their allowed global block.
    Raises ConfigurationError immediately on violation.
    """
    out_of_bounds = ids[(ids < bounds.min_id) | (ids > bounds.max_id)]

    if out_of_bounds.size > 0:
        bad_examples = np.unique(out_of_bounds)[:20]
        raise ConfigurationError(
            f"CRITICAL: {bounds.name} ID Namespace Violation.\n"
            f"Expected IDs between {bounds.min_id:1_.0f} and {bounds.max_id:1_.0f}.\n"
            f"Found {out_of_bounds.size:1,.0f} violations. Examples: {bad_examples}"
        )
