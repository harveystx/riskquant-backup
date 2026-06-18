# src/riskquant/market_data/surfaces/y_axes_types.py

from enum import IntEnum

import numpy as np


class SurfaceYAxisType(IntEnum):
    STRIKE = 1
    LOG_STRIKE = 2
    MONEYNESS_SPOT = 3
    MONEYNESS_FWD = 4
    LOG_MONEYNESS_SPOT = 5
    LOG_MONEYNESS_FWD = 6
    DELTA_CALL = 7


# compiler-friendly int8 constants
SURFACE_Y_AXIS_TYPE_STRIKE: np.int8 = np.int8(
    SurfaceYAxisType.STRIKE,
)
SURFACE_Y_AXIS_TYPE_LOG_STRIKE: np.int8 = np.int8(
    SurfaceYAxisType.LOG_STRIKE,
)
SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT: np.int8 = np.int8(
    SurfaceYAxisType.MONEYNESS_SPOT,
)
SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD: np.int8 = np.int8(
    SurfaceYAxisType.MONEYNESS_FWD,
)
SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT: np.int8 = np.int8(
    SurfaceYAxisType.LOG_MONEYNESS_SPOT,
)
SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD: np.int8 = np.int8(
    SurfaceYAxisType.LOG_MONEYNESS_FWD,
)
SURFACE_Y_AXIS_TYPE_DELTA_CALL: np.int8 = np.int8(
    SurfaceYAxisType.DELTA_CALL,
)

# Names tuple
SURFACE_Y_AXIS_TYPE_NAMES: tuple[str, ...] = tuple(
    method.name.upper() for method in SurfaceYAxisType
)

# Central registry
SURFACE_Y_AXIS_TYPE_REGISTRY: dict[str, SurfaceYAxisType] = {
    "STRIKE": SurfaceYAxisType.STRIKE,
    "LOG_STRIKE": SurfaceYAxisType.LOG_STRIKE,
    "MONEYNESS_SPOT": SurfaceYAxisType.MONEYNESS_SPOT,
    "MONEYNESS_FWD": SurfaceYAxisType.MONEYNESS_FWD,
    "LOG_MONEYNESS_SPOT": SurfaceYAxisType.LOG_MONEYNESS_SPOT,
    "LOG_MONEYNESS_FWD": SurfaceYAxisType.LOG_MONEYNESS_FWD,
    "DELTA_CALL": SurfaceYAxisType.DELTA_CALL,
}


# ------------------------------------------------------------------
# Post-calibration y-axis-kind mapping
# ------------------------------------------------------------------
# Some y-axis kinds declared in the static config are NEVER seen by the
# pricing kernels — the builder calibrates them at hydrate time to a
# kernel-consumable representation. `surface_y_axis_kind_for_kernel`
# returns the int8 id the kernel will branch on, given the static
# config's declared kind string.
#
# Single source of truth: BOTH the ingest-side per-trade y-axis-type
# resolver AND the runtime per-surface y-axis-type-ids array MUST go
# through this helper. That keeps the per-trade and per-surface views
# in lockstep without round-tripping the original-DELTA signal through
# the runner's UniverseMapping.
#
# Current mapping:
#   DELTA_CALL   -> MONEYNESS_FWD  (see market_data.calibrate.delta_to_moneyness)
#   *            -> identity
_POST_CALIBRATION_YAX_ID: dict[SurfaceYAxisType, np.int8] = {
    SurfaceYAxisType.STRIKE: SURFACE_Y_AXIS_TYPE_STRIKE,
    SurfaceYAxisType.LOG_STRIKE: SURFACE_Y_AXIS_TYPE_LOG_STRIKE,
    SurfaceYAxisType.MONEYNESS_SPOT: SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT,
    SurfaceYAxisType.MONEYNESS_FWD: SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
    SurfaceYAxisType.LOG_MONEYNESS_SPOT: SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT,
    SurfaceYAxisType.LOG_MONEYNESS_FWD: SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD,
    # DELTA_CALL gets calibrated to MONEYNESS_FWD by the builder.
    SurfaceYAxisType.DELTA_CALL: SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
}


def surface_y_axis_kind_for_kernel(y_axis_kind: str) -> np.int8:
    """
    Resolve a static-config y-axis-kind string to the int8 id the
    pricing kernel will dispatch on.

    Performs the static_config -> post-calibration mapping in one
    place. DELTA-quoted surfaces are calibrated to MONEYNESS_FWD at
    builder time; this helper returns MONEYNESS_FWD's int8 id so that
    ingest- and runtime-side y-axis-type arrays both see the post-
    calibration value.

    Raises
    ------
    KeyError
        If `y_axis_kind` is not a `SurfaceYAxisType` member.
    """
    return _POST_CALIBRATION_YAX_ID[SurfaceYAxisType[y_axis_kind]]
