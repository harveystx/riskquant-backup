# src/riskquant/market_data/state.py
"""
Market-data execution containers.

RiskQuant separates raw market data from valuation-ready factors across two
frozen SoA tiers, both defined here:

    Raw market data
        |
        v
    MarketDataMatrix     <-- raw tier: universe-aligned interpolated grids,
        |                    FX-effective spots, calibrated surfaces. One row
        |                    per *raw factor*, shared across many trades.
        v
    assemble_valuation_state (the fold)
        |
        v
    ValuationState       <-- terminus: pre-multiplied EFFECTIVE factors. One
        |                    row per *effective combination*. Kernels gather
        v                    a single row and multiply -- no branching, no
    Valuation Engine         inversion, no interpolation in the hot path.
        |
        v
    Pricing Kernels

`MarketDataMatrix` is the producer-side output of `build_market_data_matrix`.
`ValuationState` is what every block pricer ingests. Neither validates or
sanitizes market data -- the compiler upstream guarantees validity. The
`MarketDataMatrix` post-init is an O(1) *structural* wiring guard only (dtype,
rank, axis agreement); it scans no data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_F64 = np.float64
_I8 = np.int8

_NDArrayF64 = NDArray[_F64]
_NDArrayI8 = NDArray[_I8]


def _mtx_f64_2d() -> _NDArrayF64:
    return np.empty((0, 0), dtype=_F64)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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
    and transformation, including topological synthesis of execution-ready arrays.

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
    """

    # ==================================================================================
    # BATCH PROPERTIES
    # ==================================================================================

    batch_size: int

    # `UixMapper.fingerprint` of the mapper that keyed every factor row (AUD-02).
    # `run_valuation` requires it to equal the portfolio's fingerprint.
    uix_fingerprint: str

    # ==================================================================================
    # RESOLVED FACTORS (RAW PRIMITIVES)
    # ==================================================================================

    # # (num_spot_prices, batch_size)
    # spot_prices: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # # (num_forward_prices, batch_size)
    # forward_prices: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # (num_forward_rates, batch_size)
    # Annualized simple forward interest rates
    forward_rates: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # (num_vols, batch_size)
    vols: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # # (num_discount_factors, batch_size)
    # discount_factors: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # # (num_fx_effective_pairs, batch_size)
    # # Effective FX spot rates (already inverted where needed) for direct gather by
    # # pricer kernels - pricer kernels are as dumb as possible and do no inversion
    # # math or dynamic checks, just one gather + multiply per leg.
    # fx_spots_effective: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # ==================================================================================
    # EXECUTION-READY EFFECTIVE FACTORS (TOPOLOGICALLY RESOLVED)
    # ==================================================================================
    # These arrays are pre-multiplied by the market data compiler to collapse memory
    # reads in the hot path. Kernels gather these instead of combining raw primitives.

    # (num_effective_discount_factors, batch_size)
    # The Universal Multiplier. Combines time-value discounting and FX translation
    # to the settlement currency into a single execution factor.
    #
    # DF_CCY(0, T) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #
    effective_discount_factors: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # (num_effective_forward_prices, batch_size)
    # Fully synthesized linear forward leg. Bakes the raw forward price, its asset
    # discount factor, and its asset-to-settle FX rate into a single PV vector.
    #
    # FWD_PRICE_CCY(0, T) * DF_CCY(0, T) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #
    effective_forward_prices: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # (num_effective_spot_prices, batch_size)
    # Synthesized linear spot leg. Bakes the asset-to-settle FX rate directly
    # into the asset's spot price.
    #
    # SPOT_PRICE_CCY(0) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #
    effective_spot_prices: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # (num_forwards_strike_ccy, batch_size)
    #
    # Forward price re-expressed in strike-currency units for Black-76.
    #
    # Constructed by:
    #
    #   1. Discounting the underlying-currency forward to PV.
    #   2. Translating that PV into strike-currency terms via the engine's
    #      settlement-currency FX numeraires.
    #   3. Re-accumulating to expiry using the strike-currency curve.
    #
    # Flow:
    #
    #     F_ccy(T)
    #         * DF_ccy(0,T)
    #         -> PV_ccy
    #
    #         * FX(ccy->settle_ccy)
    #         / FX(strike_ccy->settle_ccy)
    #         -> PV_strike_ccy
    #
    #         / DF_strike_ccy(0,T)
    #         -> F_strike_ccy(T)
    #
    # Therefore:
    #
    #     F_strike_ccy(T)
    #         = F_ccy(T)
    #         * DF_ccy(0,T)
    #         * FX(ccy->settle_ccy)
    #         / FX(strike_ccy->settle_ccy)
    #         / DF_strike_ccy(0,T)
    #
    # This is an expiry-time forward price (not a PV) expressed in strike
    # currency units and is consumed directly by Black-76 kernels.
    forward_prices_strike_ccy: _NDArrayF64 = field(default_factory=_mtx_f64_2d)

    # ==================================================================================
    # KERNEL CACHES
    #
    # These are pre-computed quantities for the pricing kernels to avoid repeated
    # computation of expensive math functions in the hot path.
    #
    # ==================================================================================

    # Log-forward-price cache used by option pricing kernels.
    #
    # Only forward prices required by option products are materialized.
    # This is a dedicated compiler-generated factor universe and does not
    # share indexing with `forward_prices_strike_ccy`.
    option_log_forward_prices_strike_ccy: _NDArrayF64 = field(
        default_factory=_mtx_f64_2d
    )

    def __post_init__(self) -> None:
        if not self.uix_fingerprint:
            raise ValueError(
                "uix_fingerprint must identify the UixMapper that keyed the state."
            )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MarketDataMatrix:
    """
    Raw, universe-aligned market-data grids: the input tier to the effective-
    factor fold (`assemble_valuation_state`).

    One row per *raw factor*, shared across every trade that references it.
    Nothing here is FX-folded, discounted, or strike-resolved yet -- that is
    the fold's job. The sole producer is `build_market_data_matrix`; all
    eleven fields are always supplied (no defaults), so a half-built matrix
    is a construction-time error rather than a silent gather of garbage.

    Axis-alignment contract (load-bearing -- the fold relies on it)
    --------------------------------------------------------------
    Two distinct universes key the grids, and they are NOT interchangeable:

        market_id_universe       -> prices / forward_nodes / vol_nodes
                                    (axis 0) and price_curve_grid (axis 0)
        discount_curve_universe  -> discount_factor_grid (axis 0)

    `prices`, `forward_nodes`, and `vol_nodes` are the SAME backing array
    (Option A1): the price / forward-node / vol-node id classes are disjoint,
    so each pricer reads only its own rows. They are exposed under three names
    because the compilers stamp three distinct `_uix` classes against them.

    Scenario-axis position is NOT uniform:

        fx_spots_effective       (K_fx,  S)        scenario axis = 1
        prices/forward/vol_nodes (N_mkt, S)        scenario axis = 1
        discount_factor_grid     (C_df,  D, S)     scenario axis = 2
        price_curve_grid         (N_mkt, D, S)     scenario axis = 2
        vol_surfaces             (C_surf, D, S, K) scenario axis = 2  <-- not last
        tte_grid                 (D,)               -- no scenario axis
        vol_surface_strikes      (C_surf, K)        -- no scenario axis
        surface_y_axis_type_ids  (C_surf,)          -- no scenario axis

    where D = len(date_ord32_universe), N_mkt = len(market_id_universe),
    C_df = len(discount_curve_universe), C_surf = number of vol surfaces,
    K = calibrated moneyness pillars, S = num_scenarios.

    This container does NOT validate or sanitize market data. `__post_init__`
    is an O(1) structural wiring guard (dtype, rank, axis agreement) that scans
    no data; it exists to catch the universe-misalignment bug class this tier
    is prone to, not to police values.
    """

    # ==================================================================================
    # BATCH PROPERTIES
    # ==================================================================================

    num_scenarios: int

    # `UixMapper.fingerprint` of the mapper whose universes key every axis of this
    # matrix (D064). Equal shapes do not imply equal identities; the fold refuses
    # a mapper with any other fingerprint.
    uix_fingerprint: str

    # ==================================================================================
    # FX
    # ==================================================================================

    # (K_fx, S) -- effective FX spots, already inverted where needed. Keyed by
    # `UixMapper.fx_effective_pairs` row. The fold and pricers do zero inversion.
    fx_spots_effective: _NDArrayF64

    # ==================================================================================
    # SCALAR NODE LEVELS (market_id_universe-aligned; A1 single backing array)
    # ==================================================================================

    # (N_mkt, S) -- spot / node price levels. Keyed by market_id_universe uix.
    prices: _NDArrayF64

    # (N_mkt, S) -- forward-node levels. Alias of `prices` (disjoint id class).
    forward_nodes: _NDArrayF64

    # (N_mkt, S) -- vol-node levels. Alias of `prices` (disjoint id class).
    vol_nodes: _NDArrayF64

    # ==================================================================================
    # CURVE GRIDS (interpolated onto the unique-date axis D)
    # ==================================================================================

    # (C_df, D, S) -- discount factors. Keyed by DISCOUNT-CURVE uix (axis 0) and
    # date uix (axis 1). NOT market_id_universe-aligned -- distinct universe.
    discount_factor_grid: _NDArrayF64

    # (N_mkt, D, S) -- forward price curves. Keyed by MARKET-ID uix (axis 0) and
    # delivery-date uix (axis 1). Curve-sourced forwards read here; node-sourced
    # forwards read `prices`. (Source kind is a compile-time tag in the
    # effective-forward universe, not a separate block type.)
    price_curve_grid: _NDArrayF64

    # ==================================================================================
    # TIME
    # ==================================================================================

    # (D,) -- act/365.25 time-to-expiry per unique portfolio date.
    tte_grid: _NDArrayF64

    # ==================================================================================
    # VOL SURFACES (post DELTA->MONEYNESS_FWD calibration)
    # ==================================================================================

    # (C_surf, K) -- calibrated moneyness pillars per surface (expiry/scenario
    # independent). The strike axis the surface kernel interpolates over.
    vol_surface_strikes: _NDArrayF64

    # (C_surf, D, S, K) -- kernel-facing layout [surface, expiry, scenario, pillar],
    # C-contiguous. Scenario axis is position 2.
    vol_surfaces: _NDArrayF64

    # (C_surf,) int8 -- post-calibration per-surface y-axis kind (DELTA rows
    # rewritten to MONEYNESS_FWD).
    surface_y_axis_type_ids: _NDArrayI8

    def __post_init__(self) -> None:
        if not self.uix_fingerprint:
            raise ValueError(
                "uix_fingerprint must be the building UixMapper's fingerprint."
            )
        # ---- dtype ----
        for name in (
            "fx_spots_effective",
            "prices",
            "forward_nodes",
            "vol_nodes",
            "discount_factor_grid",
            "price_curve_grid",
            "tte_grid",
            "vol_surface_strikes",
            "vol_surfaces",
        ):
            arr = getattr(self, name)
            if arr.dtype != _F64:
                raise TypeError(f"{name} must be float64; got {arr.dtype}")
        if self.surface_y_axis_type_ids.dtype != _I8:
            raise TypeError(
                "surface_y_axis_type_ids must be int8; "
                f"got {self.surface_y_axis_type_ids.dtype}"
            )

        # ---- rank ----
        for name, ndim in (
            ("fx_spots_effective", 2),
            ("prices", 2),
            ("forward_nodes", 2),
            ("vol_nodes", 2),
            ("discount_factor_grid", 3),
            ("price_curve_grid", 3),
            ("tte_grid", 1),
            ("vol_surface_strikes", 2),
            ("vol_surfaces", 4),
            ("surface_y_axis_type_ids", 1),
        ):
            arr = getattr(self, name)
            if arr.ndim != ndim:
                raise ValueError(f"{name} must be {ndim}D; got shape {arr.shape}")

        # ---- date axis D (anchored on tte_grid) ----
        d_dates = self.tte_grid.shape[0]
        for name, axis in (
            ("discount_factor_grid", 1),
            ("price_curve_grid", 1),
            ("vol_surfaces", 1),
        ):
            arr = getattr(self, name)
            if arr.shape[axis] != d_dates:
                raise ValueError(
                    f"{name} date axis ({arr.shape[axis]}) != len(tte_grid) ({d_dates})"
                )

        # ---- market-id axis N (anchored on prices) ----
        n_mkt = self.prices.shape[0]
        for name in ("forward_nodes", "vol_nodes", "price_curve_grid"):
            arr = getattr(self, name)
            if arr.shape[0] != n_mkt:
                raise ValueError(
                    f"{name} market-id axis ({arr.shape[0]}) != prices rows ({n_mkt})"
                )

        # ---- surface axis C_surf (anchored on vol_surfaces) ----
        c_surf = self.vol_surfaces.shape[0]
        if self.vol_surface_strikes.shape[0] != c_surf:
            raise ValueError(
                f"vol_surface_strikes surfaces ({self.vol_surface_strikes.shape[0]}) "
                f"!= vol_surfaces surfaces ({c_surf})"
            )
        if self.surface_y_axis_type_ids.shape[0] != c_surf:
            raise ValueError(
                f"surface_y_axis_type_ids surfaces "
                f"({self.surface_y_axis_type_ids.shape[0]}) "
                f"!= vol_surfaces surfaces ({c_surf})"
            )
        if self.vol_surfaces.shape[3] != self.vol_surface_strikes.shape[1]:
            raise ValueError(
                f"vol_surfaces pillar axis ({self.vol_surfaces.shape[3]}) != "
                f"vol_surface_strikes pillars ({self.vol_surface_strikes.shape[1]})"
            )

        # ---- scenario axis S (anchored on num_scenarios) ----
        s = self.num_scenarios
        for name, axis in (
            ("fx_spots_effective", 1),
            ("prices", 1),
            ("discount_factor_grid", 2),
            ("price_curve_grid", 2),
            ("vol_surfaces", 2),
        ):
            arr = getattr(self, name)
            if arr.shape[axis] != s:
                raise ValueError(
                    f"{name} scenario axis ({arr.shape[axis]}) != num_scenarios ({s})"
                )
