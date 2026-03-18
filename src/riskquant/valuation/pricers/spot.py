# src/riskquant/valuation/pricers/spot.py
"""Vectorized pricer for spot positions (SpotBlock) in the CompiledPortfolio framework."""
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from riskquant.portfolio.blocks.spot_block import SpotBlock
    from riskquant.market_data.state import MarketDataMatrix


def price_spot_block(
    *,
    block: "SpotBlock",
    market_data_mtx: "MarketDataMatrix",
) -> np.ndarray:
    """
    God-Tier Vectorized Spot Pricer.
    Calculates PV = Qty * Price * FX in-place to minimize RAM usage and GC overhead.
    """
    # ==========================================
    # 1. BASE PV (Allocates the Master Output Array)
    # ==========================================
    # Fancy indexing allocates the (M, S) array. We own this memory,
    # so we will mutate it directly to save RAM.
    pvs = market_data_mtx.prices[block.price_uix, :]

    # Scale by Quantity in-place
    pvs *= block.qty[:, None]

    # ==========================================
    # 2. FX TRANSLATION (Sentinel-Safe & In-Place)
    # ==========================================
    # Identify only the trades that require an FX conversion (uix >= 0)
    # This prevents the NO_FX_RF_REQUIRED sentinel (-6) from wrapping around the array
    fx_mask = block.fx_uix >= 0

    if np.any(fx_mask):
        valid_fx_uix = block.fx_uix[fx_mask]

        # Fetch only the FX rates we actually need
        fx_rates = market_data_mtx.fx_spots[valid_fx_uix, :]

        # Identify which of these specific valid FX rates need mathematical inversion
        inv_mask = block.fx_invert[fx_mask].astype(bool, copy=False)

        if np.any(inv_mask):
            # In-place mathematical inversion
            fx_rates[inv_mask, :] = 1.0 / fx_rates[inv_mask, :]

        # Multiply into the master PV array in-place
        pvs[fx_mask, :] *= fx_rates

    # ==========================================
    # 3. RETURN
    # ==========================================
    return pvs
