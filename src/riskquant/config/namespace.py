# src/riskquant/config/namespace.py
"""
Global ID Namespace definitions and validation contracts.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from riskquant.exceptions import ConfigurationError

_BLOCK_SIZE: int = 250_000_000  # one block

_NODE_MIN_ID = 0  # node id 0 stays valid
_NODE_MAX_ID = _BLOCK_SIZE - 1  # [0,           249_999_999]
_CURVE_MIN_ID = _BLOCK_SIZE  # [250_000_000, 499_999_999]
_CURVE_MAX_ID = 2 * _BLOCK_SIZE - 1
_SURFACE_MIN_ID = 2 * _BLOCK_SIZE  # [500_000_000, 749_999_999]
_SURFACE_MAX_ID = 3 * _BLOCK_SIZE - 1
_CUBE_MIN_ID = 3 * _BLOCK_SIZE  # [750_000_000, 999_999_999]
_CUBE_MAX_ID = 4 * _BLOCK_SIZE - 1


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
    < _CUBE_MIN_ID
    < _CUBE_MAX_ID
), "ID namespace boundaries must be strictly increasing and non-overlapping."


assert (
    max(_NODE_MAX_ID, _CURVE_MAX_ID, _SURFACE_MAX_ID, _CUBE_MAX_ID)
    <= np.iinfo(np.int32).max
)


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
    CUBES = IdNamespaceBounds(
        min_id=_CUBE_MIN_ID,
        max_id=_CUBE_MAX_ID,
        name="Cubes (Containers, dimension = 3)",
    )


def enforce_id_namespace(
    *,
    ids: npt.NDArray[np.integer],
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


if __name__ == "__main__":
    # Quick sanity check of namespace boundaries
    print("Global ID Namespace Boundaries:")
    for namespace in [
        GlobalIdNamespace.NODES,
        GlobalIdNamespace.CURVES,
        GlobalIdNamespace.SURFACES,
        GlobalIdNamespace.CUBES,
    ]:
        print(f"{namespace.name}: {namespace.min_id:1_.0f} to {namespace.max_id:1_.0f}")
