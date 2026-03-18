# src/riskquant/valuation/pricers/forward_curve.py
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from riskquant.portfolio.blocks.forward_block import ForwardCurveBlock
    from riskquant.market_data.state import MarketDataMatrix


def price_forward_curve_block(
    *,
    block: "ForwardCurveBlock",
    market_data_mtx: "MarketDataMatrix",
) -> np.ndarray:
    """
    God-Tier Vectorized Forward Curve Pricer.
    Calculates PV = Qty * (Fwd * FX_u2s * DF_under - Strike * FX_k2s * DF_strike)
    Uses in-place mutations and sentinel-skipping to minimize RAM and maximize CPU efficiency.
    """
    # ==========================================
    # 1. UNDERLYING LEG (Allocates Master Array)
    # ==========================================
    # Fancy indexing returns a copy. We own this memory and will use it
    # as our running PV accumulator to save RAM.
    # Shape: (N, S)
    pvs = market_data_mtx.price_curve_grid[
        block.price_curve_uix,
        block.delivery_date_uix,
        :,
    ]

    # --- Apply FX (Underlying to Settlement) ---
    mask_fx_u = block.fx_under_to_settle_uix >= 0
    if np.any(mask_fx_u):
        valid_fx_u = block.fx_under_to_settle_uix[mask_fx_u]
        fx_u = market_data_mtx.fx_spots[valid_fx_u, :]

        # In-place mathematical inversion
        inv_u = block.fx_under_to_settle_invert[mask_fx_u].astype(
            bool, copy=False
        )
        if np.any(inv_u):
            fx_u[inv_u, :] = 1.0 / fx_u[inv_u, :]

        # In-place multiplication into the master accumulator
        pvs[mask_fx_u, :] *= fx_u

    # --- Apply Discounting (Underlying) ---
    mask_df_u = block.under_discount_curve_uix >= 0
    if np.any(mask_df_u):
        valid_df_u = block.under_discount_curve_uix[mask_df_u]
        valid_settle_u = block.settlement_date_uix[mask_df_u]
        df_u = market_data_mtx.discount_factor_grid[
            valid_df_u, valid_settle_u, :
        ]

        pvs[mask_df_u, :] *= df_u

    # ==========================================
    # 2. STRIKE LEG (Allocates Secondary Array)
    # ==========================================
    # We allocate exactly one temporary array for the strike leg.
    # Shape: (N, S)
    leg_strike = np.empty_like(pvs, dtype=np.float64)

    # Broadcast the 1D strike array across the S scenarios
    leg_strike[:] = block.strike[:, None]

    # --- Apply FX (Strike to Settlement) ---
    mask_fx_k = block.fx_strike_to_settle_uix >= 0
    if np.any(mask_fx_k):
        valid_fx_k = block.fx_strike_to_settle_uix[mask_fx_k]
        fx_k = market_data_mtx.fx_spots[valid_fx_k, :]

        inv_k = block.fx_strike_to_settle_invert[mask_fx_k].astype(
            bool, copy=False
        )
        if np.any(inv_k):
            fx_k[inv_k, :] = 1.0 / fx_k[inv_k, :]

        leg_strike[mask_fx_k, :] *= fx_k

    # --- Apply Discounting (Strike) ---
    mask_df_k = block.strike_discount_curve_uix >= 0
    if np.any(mask_df_k):
        valid_df_k = block.strike_discount_curve_uix[mask_df_k]
        valid_settle_k = block.settlement_date_uix[mask_df_k]
        df_k = market_data_mtx.discount_factor_grid[
            valid_df_k, valid_settle_k, :
        ]

        leg_strike[mask_df_k, :] *= df_k

    # ==========================================
    # 3. FINAL PV ASSEMBLY
    # ==========================================
    # PV = (leg_under - leg_strike) * qty
    # All operations done in-place to avoid final allocations
    pvs -= leg_strike
    pvs *= block.qty[:, None]

    return pvs
