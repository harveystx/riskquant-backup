# src/riskquant/observations/orientation.py

"""
Explicit orientation declaration for empirical observation matrices.

The canonical internal layout is ``(K, S)``: variables on rows, observations on
columns. Callers declare which layout they hold. The package never infers the
orientation from the shape and never transposes silently.

Rationale for the canonical layout
----------------------------------
Every pairwise dependence estimator (Kendall, weighted Kendall, distance
correlation, mutual information) consumes one complete variable at a time. With
variables on rows of a C-contiguous matrix, each variable is a unit-stride
slice, so the sort / scan kernels read straight lines of memory. The Pearson
Gram product ``Z @ Z.T`` also wants row-major variables.

Standard ``Enum`` semantics apply. Equality is identity-based, members are
hashable, and comparison against a foreign type returns ``False`` instead of
raising. The prototype's custom ``__eq__`` is deliberately not reproduced: it
set ``__hash__`` to ``None``, which made the members unusable as dictionary
keys.
"""

from __future__ import annotations

from enum import Enum


class ObservationOrientation(Enum):
    """
    Declares which axis of the caller's array carries the variables.

    Attributes
    ----------
    VARIABLES_ON_ROWS
        Input is ``(K, S)``. Already canonical; no transpose occurs.

    OBSERVATIONS_ON_ROWS
        Input is ``(S, K)`` — the tabular / DataFrame layout, one row per
        observation date. Canonicalisation transposes exactly once and
        materialises a C-contiguous copy.
    """

    VARIABLES_ON_ROWS = 0
    OBSERVATIONS_ON_ROWS = 1
