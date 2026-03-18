# src/riskquant/valuation/pricers/vanilla_european_option_curve.py
import numpy as np
from typing import TYPE_CHECKING

from riskquant.constants import MOCK_VOLATILITY

# FIXED: We use the Future Value (FV) kernel to maintain ownership of the memory
# and avoid passing massive df_settle arrays.
from riskquant.valuation.models.black_scholes import fv_black_scholes_vanilla_3d
from riskquant.market_data.curves.interp_defs import (
    interp_strike_linear_flat_extrap,
)
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_STRIKE,
    SURFACE_Y_AXIS_TYPE_LOG_STRIKE,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT,
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD,
)

if TYPE_CHECKING:
    from riskquant.portfolio.blocks.vanilla_european_option_block import (
        VanillaEuropeanOptionCurveBlock,
    )
    from riskquant.market_data.state import MarketDataMatrix


def price_vanilla_european_option_curve_block(
    *,
    block: "VanillaEuropeanOptionCurveBlock",
    market_data_mtx: "MarketDataMatrix",
) -> np.ndarray:
    """
    God-Tier Vectorized Vanilla European Option Curve Pricer.
    Optimized for minimal allocations, in-place transformations, and zero redundant arrays.
    """

    # ==========================================
    # 1. FETCH FORWARD PRICES (The Master Accumulator)
    # ==========================================
    # Fancy indexing returns a copy. We own this memory.
    # Shape: (N, S)
    fwd_in_strike_ccy = market_data_mtx.price_curve_grid[
        block.underlying_price_curve_uix,
        block.delivery_date_uix,
        :,
    ]

    # Time to Expiry is standard across the block
    tte = market_data_mtx.tte_grid[block.expiry_date_uix][:, None]

    # ==========================================
    # 2. UNDERLYING FX & DISCOUNTING (In-Place)
    # ==========================================
    # Covered Interest Parity: Fwd = Fwd * Spot_FX * DF_under

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

    mask_df_u = block.under_discount_curve_uix >= 0
    if np.any(mask_df_u):
        valid_df_u = block.under_discount_curve_uix[mask_df_u]
        valid_exp_u = block.expiry_date_uix[mask_df_u]
        df_u = market_data_mtx.discount_factor_grid[valid_df_u, valid_exp_u, :]

        fwd_in_strike_ccy[mask_df_u, :] *= df_u

    # ==========================================
    # 3. STRIKE FX & DISCOUNTING (In-Place)
    # ==========================================
    # Covered Interest Parity: Fwd = Fwd / (Spot_FX * DF_strike)

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

    mask_df_k = block.strike_discount_curve_uix >= 0
    if np.any(mask_df_k):
        valid_df_k = block.strike_discount_curve_uix[mask_df_k]
        valid_exp_k = block.expiry_date_uix[mask_df_k]
        df_k_exp = market_data_mtx.discount_factor_grid[
            valid_df_k, valid_exp_k, :
        ]

        fwd_in_strike_ccy[mask_df_k, :] /= df_k_exp

    # ==========================================
    # 4. Y-COORDINATE TRANSFORMER
    # ==========================================
    K = block.strike[:, None]
    F = fwd_in_strike_ccy

    # Pre-allocate target array. Unused slots will be overwritten.
    y_target = np.empty_like(F, dtype=np.float64)
    y_types = block.vol_surface_y_axis_type_id

    # --- STRIKE ---
    m_strike = y_types == SURFACE_Y_AXIS_TYPE_STRIKE
    if np.any(m_strike):
        y_target[m_strike] = K[m_strike]

    # --- LOG STRIKE ---
    m_log_strike = y_types == SURFACE_Y_AXIS_TYPE_LOG_STRIKE
    if np.any(m_log_strike):
        y_target[m_log_strike] = np.log(K[m_log_strike])

    # --- MONEYNESS (Fwd & Spot Combined) ---
    m_fwd_mon = (y_types == SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD) | (
        y_types == SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT
    )
    if np.any(m_fwd_mon):
        y_target[m_fwd_mon] = K[m_fwd_mon] / F[m_fwd_mon]

    # --- LOG MONEYNESS (Fwd & Spot Combined) ---
    m_log_fwd_mon = (y_types == SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD) | (
        y_types == SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT
    )
    if np.any(m_log_fwd_mon):
        y_target[m_log_fwd_mon] = np.log(K[m_log_fwd_mon] / F[m_log_fwd_mon])

    # ==========================================
    # 5. DYNAMIC STRIKE INTERPOLATION (PHASE 2)
    # ==========================================
    if market_data_mtx.vol_surfaces.size == 0:
        vols = np.full_like(F, MOCK_VOLATILITY, dtype=np.float64)
    else:
        vols = interp_strike_linear_flat_extrap(
            pillars=market_data_mtx.vol_surface_strikes[block.vol_surface_uix],
            values=market_data_mtx.vol_surfaces[
                block.vol_surface_uix, block.expiry_date_uix
            ],
            targets=y_target,
        )

    # ==========================================
    # 6. EXECUTE BLACK-SCHOLES (FV)
    # ==========================================
    # This allocates our final PV array as Future Value.
    pvs = fv_black_scholes_vanilla_3d(
        fwd=fwd_in_strike_ccy,
        strike=K,
        vol=vols,
        tte=tte,
        is_call=block.is_call[:, None],
    )

    # ==========================================
    # 7. FINAL PV ASSEMBLY (In-Place)
    # ==========================================
    # Discount the Future Value back to today
    if np.any(mask_df_k):
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
