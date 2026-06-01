# src/riskquant/valuation/pricers/vanilla_european_option_node.py
"""
Vanilla European option pricer, node variant.

Synthesises the forward in strike currency via Covered Interest Parity,
prices via Black-76, then discounts and FX-translates back to settle ccy.

Math (per trade i, scenario s)
------------------------------
    rate_k = fx_spots_effective[fx_k_eff_uix(i), s]   # k_ccy -> settle_ccy

    # CIP forward in strike currency
    fwd_strike_ccy
        = forward_node[p, s]
          * fx_spots_effective[fx_u_eff_uix(i), s]    # u_ccy -> settle_ccy
          * df_grid[df_u_uix(i), expiry_uix(i), s]    # under-DF at expiry
          / rate_k                                    # settle_ccy -> k_ccy
          / df_grid[df_k_uix(i), expiry_uix(i), s]    # strike-DF at expiry

    # Black-76 future value (cp_mult = +1 call, -1 put)
    sigma = vol_nodes[v, s]
    std_dev = max(sigma * sqrt(tte), eps)
    d1 = (log(fwd / k) + 0.5 * sigma**2 * tte) / std_dev
    d2 = d1 - std_dev
    fv = cp * (fwd * N(cp * d1) - k * N(cp * d2))

    # PV discount in strike ccy, then translate back to settle ccy
    pv_strike_ccy = fv * df_grid[df_k_uix(i), settle_uix(i), s]
    pv_settle_ccy = pv_strike_ccy * rate_k

    out[i, s] = qty[i] * pv_settle_ccy

FX rows are already EFFECTIVE in `MarketDataMatrix.fx_spots_effective`; the
kernel does NO inversion math. The strike-FX rate `rate_k` is loaded once
per scenario and used as a divisor in the forward synthesis and as a
multiplier in the PV translation — equivalent counts to the legacy code
but with the `(1.0 / rate_k) if inv_k else rate_k` gymnastics gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.kernels.vanilla_european_option_node import (
    _kernel_price_vanilla_european_option_node,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.portfolio.blocks.vanilla_european_option_block import (
        VanillaEuropeanOptionNodeBlock,
    )


def price_vanilla_european_option_node_block(
    *,
    block: VanillaEuropeanOptionNodeBlock,
    market_data_mtx: MarketDataMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    """European-option-node pricer entrypoint."""
    _kernel_price_vanilla_european_option_node(
        out,
        market_data_mtx.forward_nodes,
        market_data_mtx.vol_nodes,
        market_data_mtx.fx_spots_effective,
        market_data_mtx.discount_factor_grid,
        market_data_mtx.sqrt_tte_grid,
        market_data_mtx.half_tte_grid,
        block.qty,
        block.strike,
        block.cp_mult,
        block.underlying_price_uix,
        block.vol_uix,
        block.fx_under_to_settle_effective_uix,
        block.under_discount_curve_uix,
        block.fx_strike_to_settle_effective_uix,
        block.strike_discount_curve_uix,
        block.expiry_date_uix,
        block.settlement_date_uix,
    )
