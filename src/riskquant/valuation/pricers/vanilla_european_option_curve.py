# src/riskquant/valuation/pricers/vanilla_european_option_curve.py
"""
Vanilla European option pricer, curve variant.

Identical CIP-forward + Black-76 math to the node variant, with two
structural differences:

  - The underlying scalar comes from `price_curve_grid[curve_uix, delivery_uix, s]`.
  - Volatility is sourced from a 4D vol surface and requires per-scenario
    strike-axis interpolation: `np.interp` over the y-axis pillars.

The y-axis-type id determines the strike coordinate used for the
interpolation (strike, log-strike, moneyness vs forward, log-moneyness, ...).

FX rows are already EFFECTIVE — no inversion math; the strike-FX rate is
loaded once per scenario, used as a divisor for CIP forward synthesis and
as a multiplier for the PV translation back to settle ccy.

Mock-vol fallback
-----------------
When `MarketDataMatrix.vol_surfaces` is empty (e.g., calibration not yet
landed), a flat `FALLBACK_MOCK_VOLATILITY_DO_NOT_SHIP` is used. This is a
development convenience and MUST not ship in production reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.constants import (
    NOT_IMPLEMENTED_SENTINEL,
)
from riskquant.valuation.kernels.vanilla_european_option_curve import (
    _kernel_price_vanilla_european_option_curve,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.portfolio.blocks.vanilla_european_option_block import (
        VanillaEuropeanOptionCurveBlock,
    )


def price_vanilla_european_option_curve_block(
    *,
    block: VanillaEuropeanOptionCurveBlock,
    market_data_mtx: MarketDataMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    """European-option-curve pricer entrypoint."""
    if np.any(block.vol_surface_y_axis_type_id == np.int8(NOT_IMPLEMENTED_SENTINEL)):
        raise NotImplementedError(
            "vol_surface_y_axis_type_id contains unresolved sentinel values. "
            "CompiledIngestSoA must supply vol_surface_y_axis_type_ids before "
            "vanilla European option curve trades can be priced."
        )

    _kernel_price_vanilla_european_option_curve(
        out,
        market_data_mtx.price_curve_grid,
        market_data_mtx.vol_surface_strikes,
        market_data_mtx.vol_surfaces,
        market_data_mtx.fx_spots_effective,
        market_data_mtx.discount_factor_grid,
        market_data_mtx.tte_grid,
        block.qty,
        block.strike,
        block.cp_mult,
        block.underlying_price_curve_uix,
        block.vol_surface_uix,
        block.vol_surface_y_axis_type_id,
        block.fx_under_to_settle_effective_uix,
        block.under_discount_curve_uix,
        block.fx_strike_to_settle_effective_uix,
        block.strike_discount_curve_uix,
        block.delivery_date_uix,
        block.expiry_date_uix,
        block.settlement_date_uix,
    )
