# src/riskquant/market_data/grid.py
"""
Intermediate multi-dimensional market data grids for the compilation pipeline.

Architectural Role (Tier 2)
---------------------------
This module provides the `InterpolatedMarketDataGrid`, which acts as the bridge
between human-readable market data and bare-metal execution arrays.

It is a dense, topological representation of the financial environment. It contains
fully interpolated yield curves, forward curves, and volatility surfaces across
all required dates and scenarios.

**CRITICAL:** This is an internal compiler structure. It is NOT execution-ready.
Pricing kernels must never import or consume this object. The `MarketDataCompiler`
is responsible for taking this grid, flattening its 3D/4D structures, applying
cross-currency topological multiplications, and emitting a flat `ValuationState`.

Axis Conventions
----------------
* Dimension 0 always corresponds to a `_uix` (the dense index for a specific curve,
  surface, or FX pair).
* 2-D scalar grids: `[uix, scenario]`
* 3-D curve grids: `[uix, delivery_date_uix, scenario]`
* 4-D vol surfaces: `[uix, expiry_date_uix, scenario, strike_pillar]`

Vol-Surface Layout
------------------
`vol_surfaces` places the `strike_pillar` in the trailing axis. This guarantees
that the strike pillars for one (surface, expiry, scenario) tuple form a single
C-contiguous run of memory. The compiler exploits this layout when slicing the
surface to perform vectorized 1D strike interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_I8 = np.int8
_I32 = np.int32
_F64 = np.float64

_NDArrayI8 = NDArray[_I8]
_NDArrayI32 = NDArray[_I32]
_NDArrayF64 = NDArray[_F64]


def _mtx_int8_1d() -> _NDArrayI8:
    return np.empty((0,), dtype=_I8)


def _mtx_int32_2d() -> _NDArrayI32:
    return np.empty((0, 0), dtype=_I32)


def _mtx_f64_1d() -> _NDArrayF64:
    return np.empty((0,), dtype=_F64)


def _mtx_f64_2d() -> _NDArrayF64:
    return np.empty((0, 0), dtype=_F64)


def _mtx_f64_3d() -> _NDArrayF64:
    return np.empty((0, 0, 0), dtype=_F64)


def _mtx_f64_4d() -> _NDArrayF64:
    return np.empty((0, 0, 0, 0), dtype=_F64)


@dataclass(frozen=True, slots=True, kw_only=True)
class InterpolatedMarketDataGrid:
    """
    Fully interpolated market topology.

    Internal compiler object.

    Represents the complete continuous market state after
    interpolation but before trade-specific factor synthesis.

    Consumed exclusively by MarketDataCompiler.
    """

    # ============================================================
    # DIMENSIONS
    # ============================================================

    num_scenarios: int

    num_unique_dates: int

    num_discount_curves: int

    num_price_curves: int

    num_vol_surfaces: int

    # ============================================================
    # DATE TOPOLOGY
    # ============================================================

    # (num_unique_dates,)
    date_ordinals: _NDArrayI32

    # (num_unique_dates,)
    tte_grid: _NDArrayF64

    sqrt_tte_grid: _NDArrayF64 = field(init=False)

    half_tte_grid: _NDArrayF64 = field(init=False)

    # ============================================================
    # FX
    # ============================================================

    # (num_fx_effective_pairs, num_scenarios)
    fx_spots_effective: _NDArrayF64

    # ============================================================
    # SCALAR NODES
    # ============================================================

    # (num_prices, num_scenarios)
    spot_prices: _NDArrayF64

    # (num_forward_nodes, num_scenarios)
    forward_prices: _NDArrayF64

    # (num_vol_nodes, num_scenarios)
    vols: _NDArrayF64

    # ============================================================
    # DISCOUNT CURVES
    # ============================================================

    # (curve_uix, date_uix, scenario)
    discount_factor_grid: _NDArrayF64

    # ============================================================
    # FORWARD CURVES
    # ============================================================

    # (curve_uix, delivery_date_uix, scenario)
    forward_curve_grid: _NDArrayF64

    # ============================================================
    # VOL SURFACES
    # ============================================================

    # (surface_uix,
    #  expiry_date_uix,
    #  scenario,
    #  strike_pillar_uix)
    vol_surfaces: _NDArrayF64

    # (surface_uix, strike_pillar_uix)
    vol_surface_strikes: _NDArrayF64

    # (surface_uix,)
    surface_y_axis_type_ids: _NDArrayI8

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "sqrt_tte_grid",
            np.sqrt(self.tte_grid),
        )

        object.__setattr__(
            self,
            "half_tte_grid",
            0.5 * self.tte_grid,
        )

    def freeze(self) -> None:

        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)

            if isinstance(value, np.ndarray):
                value.flags.writeable = False
