# src/riskquant/valuation/pricers/vanilla_european_option_node.py
import numpy as np
from typing import TYPE_CHECKING

# Note: We import the Future Value (FV) kernel to maintain ownership of the memory
# so we can apply final discounting and FX conversions in-place.
from riskquant.valuation.models.black_scholes import fv_black_scholes_vanilla_3d

if TYPE_CHECKING:
    from riskquant.portfolio.blocks.vanilla_european_option_block import (
        VanillaEuropeanOptionNodeBlock,
    )
    from riskquant.market_data.state import MarketDataMatrix


def price_vanilla_european_option_node_block(
    *,
    block: "VanillaEuropeanOptionNodeBlock",
    market_data_mtx: "MarketDataMatrix",
) -> np.ndarray:
    """
    God-Tier Vectorized Vanilla European Option Node Pricer.
    Uses in-place mutations, sentinel-skipping, and caching to minimize RAM.
    """
    # ==========================================
    # 1. FETCH UNDERLYING (Allocates Master Array)
    # ==========================================
    # Fancy indexing returns a copy. We own this memory and will use it
    # to synthesize the Forward Price in the Strike Currency in-place.
    fwd_in_strike_ccy = market_data_mtx.forward_nodes[
        block.underlying_price_uix, :
    ]

    # ==========================================
    # 2. SYNTHESIZE FORWARD FX (In-Place)
    # ==========================================

    # --- Underlying FX ---
    mask_fx_u = block.fx_under_to_settle_uix >= 0
    if np.any(mask_fx_u):
        valid_fx_u = block.fx_under_to_settle_uix[mask_fx_u]
        fx_u = market_data_mtx.fx_spots[valid_fx_u, :]

        inv_u = block.fx_under_to_settle_invert[mask_fx_u].astype(
            bool, copy=False
        )
        if np.any(inv_u):
            fx_u[inv_u, :] = 1.0 / fx_u[inv_u, :]

        fwd_in_strike_ccy[mask_fx_u, :] *= fx_u

    # --- Underlying Discount (Expiry) ---
    mask_df_u = block.under_discount_curve_uix >= 0
    if np.any(mask_df_u):
        valid_df_u = block.under_discount_curve_uix[mask_df_u]
        valid_exp_u = block.expiry_date_uix[mask_df_u]
        df_u = market_data_mtx.discount_factor_grid[valid_df_u, valid_exp_u, :]

        fwd_in_strike_ccy[mask_df_u, :] *= df_u

    # --- Strike FX (Cached for later use) ---
    mask_fx_k = block.fx_strike_to_settle_uix >= 0
    fx_k_cache = None
    if np.any(mask_fx_k):
        valid_fx_k = block.fx_strike_to_settle_uix[mask_fx_k]
        fx_k_cache = market_data_mtx.fx_spots[valid_fx_k, :]

        inv_k = block.fx_strike_to_settle_invert[mask_fx_k].astype(
            bool, copy=False
        )
        if np.any(inv_k):
            fx_k_cache[inv_k, :] = 1.0 / fx_k_cache[inv_k, :]

        fwd_in_strike_ccy[mask_fx_k, :] /= fx_k_cache

    # --- Strike Discount (Expiry) ---
    mask_df_k = block.strike_discount_curve_uix >= 0
    if np.any(mask_df_k):
        valid_df_k = block.strike_discount_curve_uix[mask_df_k]
        valid_exp_k = block.expiry_date_uix[mask_df_k]
        df_k = market_data_mtx.discount_factor_grid[valid_df_k, valid_exp_k, :]

        fwd_in_strike_ccy[mask_df_k, :] /= df_k

    # ==========================================
    # 3. EXECUTE BLACK-SCHOLES (FV)
    # ==========================================
    # This allocates our final PV array. We fetch TTE and Vols natively here.
    pvs = fv_black_scholes_vanilla_3d(
        fwd=fwd_in_strike_ccy,
        strike=block.strike[:, None],
        vol=market_data_mtx.vol_nodes[block.vol_uix, :],
        tte=market_data_mtx.tte_grid[block.expiry_date_uix][:, None],
        is_call=block.is_call[:, None],
    )

    # ==========================================
    # 4. FINAL PV ASSEMBLY (In-Place)
    # ==========================================
    # Discount the Future Value back to today
    if np.any(mask_df_k):
        # Note: We reuse the mask_df_k since the settlement discount curve
        # is the same curve used for the strike leg
        valid_df_k = block.strike_discount_curve_uix[mask_df_k]
        valid_stl_k = block.settlement_date_uix[mask_df_k]
        df_settle = market_data_mtx.discount_factor_grid[
            valid_df_k, valid_stl_k, :
        ]

        pvs[mask_df_k, :] *= df_settle

    # Translate PV from Strike Currency to Settlement Currency
    if fx_k_cache is not None:
        pvs[mask_fx_k, :] *= fx_k_cache

    # Scale by quantity
    pvs *= block.qty[:, None]

    return pvs
