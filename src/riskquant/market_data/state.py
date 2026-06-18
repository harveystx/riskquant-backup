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

Per-surface y-axis-type ids — POST-CALIBRATION
----------------------------------------------
`surface_y_axis_type_ids` is `(num_vol_surfaces,)` int8 and carries the
y-axis-type id AFTER the builder's DELTA -> MONEYNESS_FWD calibration
pass. DELTA_CALL rows are rewritten in place to MONEYNESS_FWD; non-
DELTA rows pass through. The pricing kernels do not read this array —
they dispatch on the per-trade copy from `IngestBatch` which is
already MONEYNESS_FWD via `surface_y_axis_kind_for_kernel` at ingest
time. The field is exposed here for diagnostics and any downstream
consumer (e.g. a downstream calibration pass) that needs the runtime
per-surface view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDataMatrix:
    """Frozen, validated SoA container of market data for vectorized valuation."""

    num_scenarios: int

    # ========== 1. SCALARS / NODES ==========

    # (num_fx_effective_pairs, num_scenarios) — deduplicated, pre-resolved FX
    # transforms. Pricers gather one row per leg per trade and multiply.
    fx_spots_effective: npt.NDArray[np.float64]

    # SCALAR-NODE LEVELS — UNIVERSE-ALIGNED (item C, option A1)
    # ------------------------------------------------------------------
    # TODO: Need some description here
    prices: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # (num_market_ids, num_scenarios) — alias of `prices` (see above)
    forward_nodes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # (num_market_ids, num_scenarios) — alias of `prices` (see above)
    vol_nodes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64),
    )

    # ========== 2. DISCOUNT GRID ==========
    # (num_discount_curves, num_unique_dates, num_scenarios)
    discount_factor_grid: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64),
    )

    # ========== 3. CURVE-BASED DATA ==========
    # (num_market_ids, num_unique_dates, num_scenarios) — axis-0 is the
    # `market_id_universe` (item C / A1, curve layer). Forward-curve blocks
    # index `price_curve_grid[price_curve_uix, delivery_uix, s]` where
    # `price_curve_uix` is the container's `market_id_universe` row; the
    # builder scatters the dense per-curve grid into these rows. Non-curve
    # rows are unused.
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

    # (num_vol_surfaces,) — POST-CALIBRATION per-surface y-axis-type id.
    # The builder runs DELTA -> MONEYNESS_FWD calibration via
    # `market_data.calibrate.delta_to_moneyness` and overwrites
    # DELTA_CALL slots with MONEYNESS_FWD; non-DELTA rows pass through.
    # Exposed here for diagnostics and any downstream consumer that
    # needs the per-surface view (the pricing kernel itself reads the
    # per-trade copy from `IngestBatch`, which already carries the
    # post-calibration value via `surface_y_axis_kind_for_kernel` at
    # ingest time — the two views agree by construction).
    surface_y_axis_type_ids: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.empty(0, dtype=np.int8),
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


_F64 = np.float64


def _empty_factor_matrix_f64() -> NDArray[_F64]:
    return np.empty(0, dtype=_F64)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValuationState:
    """
    Fully-resolved valuation factors consumed directly by pricing kernels.

    Frozen, validated SoA container of valuation factors for vectorized valuation.
    This is not a market-data container; it is the final, fully-resolved representation
    of market data consumed by the valuation engine.

    `ValuationState` is the final market-data representation used by the
    valuation engine. It contains only scenario-dependent quantities with
    direct economic meaning in pricing equations.

    NB! THIS IS AN INTERNAL EXECUTION-STATE OBJECT. IT DOES NOT VALIDATE OR SANITIZE
    MARKET DATA. IT IS THE FINAL, FULLY-RESOLVED REPRESENTATION OF MARKET DATA
    CONSUMED BY THE VALUATION ENGINE. IT IS NOT MEANT TO BE VIEWABLE OR AUDITABLE BY
    END USERS. THE COMPILER GUARANTEES THAT ALL MARKET DATA IS VALID AND SANITIZED
    BEFORE IT REACHES THIS OBJECT.

    Architectural role
    ------------------
    RiskQuant deliberately separates:

        Raw Market Data
            ↓
        Market Data Compiler
            ↓
        ValuationState
            ↓
        Valuation Engine
            ↓
        Pricing Kernels

    The market-data compiler is responsible for all market-data interpretation
    and transformation, including:

        * Curve interpolation
        * Surface interpolation
        * Forward extraction
        * Discount-factor generation
        * Vol calibration
        * FX inversion / resolution
        * Date and calendar handling

    Pricing kernels must not perform any of the above operations.

    Kernel contract
    ---------------
    Every array follows the convention:

        [factor_uix, scenario]

    where:

        factor_uix
            Dense execution coordinate produced by the market-data compiler.

        scenario
            Scenario index within the current scenario batch.

    Kernels gather rows from `ValuationState`, combine them with trade-level
    constants from compiled blocks, and perform arithmetic.

    Example:

        forward_price = state.forward_prices[forward_price_uix]
        discount_factor = state.discount_factors[df_uix]

        pv = qty * forward_price * discount_factor

    Design principles
    -----------------
    `ValuationState` contains valuation factors, not market structures.

    Allowed:

        * Spot prices
        * Forward prices
        * Forward rates
        * Discount factors
        * Volatilities
        * FX transforms

    Not allowed:

        * Curves
        * Surfaces
        * Dates
        * Calendars
        * Interpolation metadata
        * Pillar definitions
        * Market-data construction logic

    If a pricing kernel must derive a quantity from a curve, surface, date,
    or calendar, that derivation belongs in the market-data compiler and the
    resulting valuation factor should be materialized here instead.

    Memory layout
    -------------
    All valuation-factor arrays are stored as:

    (num_factors, scenario_batch_size)

    with the scenario axis in the trailing dimension.

    This layout matches the kernel execution model:

        factor_row = factor[factor_uix]

    followed by a contiguous scan across scenarios.

    The trailing scenario dimension is therefore C-contiguous,
    allowing SIMD-friendly streaming reads and efficient CPU
    cache utilization.

    The scenario dimension is batch-local; `ValuationState` represents only
    the currently active scenario batch, not the entire simulation universe.


    """

    # ==================================================================================
    # BATCH PROPERTIES
    # ==================================================================================

    batch_size: int

    # ==================================================================================
    # RESOLVED FACTORS
    # ==================================================================================

    # (num_spot_prices, batch_size)
    spot_prices: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)

    # (num_forward_prices, batch_size)
    forward_prices: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)

    # (num_forward_rates, batch_size)
    # Annualized simple forward interest rates
    forward_rates: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)

    # (num_vols, batch_size)
    vols: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)

    # (num_discount_factors, batch_size)
    discount_factors: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)

    # (num_fx_effective_pairs, batch_size)
    # Effective FX spot rates (already inverted where needed) for direct gather by
    # pricer kernels - pricer kernels are as dumb as possible and do no inversion
    # math or dynamic checks, just one gather + multiply per leg.
    fx_spots_effective: NDArray[_F64] = field(default_factory=_empty_factor_matrix_f64)
