# src/riskquant/market_data/state.py
"""
Vectorized market data container for the valuation engine.

`MarketDataMatrix` is the frozen, valuation-ready SoA representation of all
market data required to price a `CompiledPortfolio` across `num_scenarios`
parallel scenarios.

Axis conventions
----------------
* Dimension 0 always corresponds to a `_uix` (price / fx_effective / vol /
  discount universe).
* For 2-D scalar grids the final dimension is the scenario index.
* 3-D grids (`discount_factor_grid`, `price_curve_grid`) carry a middle
  date-uix axis aligned with `CompiledPortfolio.unique_dates_ord32`, and
  scenario in the trailing axis.
* `vol_surfaces` is 4-D and arranged as `[surface, expiry, scenario, pillar]`
  so that the strike-axis pillars for one trade-scenario are a single
  contiguous run of memory — see Vol-surface layout below.

FX semantics
------------
`fx_spots_effective` holds the deduplicated, fully-resolved FX transform
matrix. Row `k` is the spot transform identified by row `k` of the
portfolio's `UixMapper.fx_effective_pairs`. Trades that need no FX carry
`INVALID_UIX` (< 0) in their per-trade `fx_*_effective_uix` field and skip
the FX gather entirely via a hoisted branch in the pricer.

Pricer contract for FX:
    out[i, s] *= fx_spots_effective[fx_effective_uix[i], s]

No inversion math, no `1.0 / rate`, no scenario-level branches. The raw
`fx_spots` matrix is producer-side only and never reaches the pricers.

Vol-surface layout (CRITICAL)
-----------------------------
`vol_surfaces` is `(num_vol_surfaces, num_unique_dates, num_scenarios,
max_vol_strikes)` — pillar in the trailing axis. This guarantees that
the strike pillars for one (surface, expiry, scenario) tuple form one
C-contiguous run, which is exactly what the strike-axis interpolation
in the option-curve kernel needs for SIMD-friendly reads.

The market-data builder is responsible for emitting this layout; the
kernels rely on it for `v_plane[s]` to be a contiguous 1-D pillar row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDataMatrix:
    """Frozen, validated SoA container of market data for vectorized valuation."""

    num_scenarios: int

    # ========== 1. SCALARS / NODES ==========

    # (num_fx_effective_pairs, num_scenarios) — deduplicated, pre-resolved FX
    # transforms. Pricers gather one row per leg per trade and multiply.
    fx_spots_effective: npt.NDArray[np.float64]

    # (num_price_factors, num_scenarios)
    prices: npt.NDArray[np.float64]

    # (num_fwd_nodes, num_scenarios)
    forward_nodes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # (num_vol_nodes, num_scenarios)
    vol_nodes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # ========== 2. DISCOUNT GRID ==========
    # (num_discount_curves, num_unique_dates, num_scenarios)
    discount_factor_grid: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64),
    )

    # ========== 3. CURVE-BASED DATA ==========
    # (num_price_curves, num_unique_dates, num_scenarios)
    price_curve_grid: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64),
    )

    # ========== 4. TIME GRIDS ==========
    # (num_unique_dates,) — year fractions from as_of to each unique date.
    tte_grid: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
    )

    sqrt_tte_grid: npt.NDArray[np.float64] = field(init=False)

    half_tte_grid: npt.NDArray[np.float64] = field(init=False)

    # ========== 5. SURFACES ==========
    # (num_vol_surfaces, max_vol_strikes) — strike pillars per surface.
    vol_surface_strikes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # (num_vol_surfaces, num_unique_dates, num_scenarios, max_vol_strikes)
    # Pillar-in-trailing-axis layout — see module docstring.
    vol_surfaces: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0, 0, 0), dtype=np.float64),
    )

    def __post_init__(self):

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
