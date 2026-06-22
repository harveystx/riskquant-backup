# src/riskquant/market_data/state.py
"""
TODO: WRITE DOCSTRING
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_F64 = np.float64

_NDArrayF64 = NDArray[_F64]


def _mtx_f64_2d() -> _NDArrayF64:
    return np.empty((0, 0), dtype=_F64)


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
