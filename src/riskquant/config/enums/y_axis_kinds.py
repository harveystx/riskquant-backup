# src/riskquant/config/enums/y_axis_kinds.py
"""
Y-axis kind taxonomy for surfaces.

Replaces surface_y_axis_types.py. A Y-axis kind tells the surface
interpolator how to interpret a surface point's y coordinate.

Forward-relative coordinates (LOG_MONEYNESS_FWD, MONEYNESS_FWD, DELTA_CALL)
depend on the underlying forward at the relevant expiry; the surface
itself does not store the forward — the pricer derives it from the
underlying curve at runtime.
"""

from __future__ import annotations

from enum import IntEnum


class YAxisKind(IntEnum):
    STRIKE = 0
    LOG_STRIKE = 1
    MONEYNESS_SPOT = 2
    MONEYNESS_FWD = 3
    LOG_MONEYNESS_SPOT = 4
    LOG_MONEYNESS_FWD = 5
    DELTA_CALL = 6


VALID_Y_AXIS_KINDS: frozenset[str] = frozenset(k.name for k in YAxisKind)
