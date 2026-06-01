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
