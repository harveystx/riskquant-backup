# src/riskquant/market_data/builder.py
from datetime import date
import numpy as np

from riskquant.market_data.curves.interp_defs import (
    interp_log_linear_flat_extrap_3d,
)
from riskquant.market_data.state import MarketDataMatrix
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_DELTA_CALL,
)
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.utils.dates import date_to_ordinal32


def _build_curve_grid(
    raw_tenors: np.ndarray,  # Shape: (C, M)
    raw_values: np.ndarray,  # Shape: (C, M, S)
    target_dates: np.ndarray,  # Shape: (D,)
    interp_methods: np.ndarray,  # Shape: (C,) int8
    extrap_methods: np.ndarray,  # Shape: (C,) int8
) -> np.ndarray:
    C, M, S = raw_values.shape
    D = len(target_dates)
    out_grid = np.empty((C, D, S), dtype=np.float64)
    if D == 0:
        return out_grid

    for (interp_id, extrap_id), math_kernel in CURVE_MATH_REGISTRY.items():
        mask = (interp_methods == interp_id) & (extrap_methods == extrap_id)
        if np.any(mask):
            out_grid[mask] = math_kernel(
                raw_tenors[mask], raw_values[mask], target_dates
            )
    return out_grid


def build_market_data_matrix(
    *,
    as_of_date: date,
    portfolio: CompiledPortfolio,
    num_scenarios: int,
    # Scalars
    raw_fx_spots: np.ndarray,
    raw_prices: np.ndarray,
    raw_forward_nodes: np.ndarray,
    raw_vol_nodes: np.ndarray,
    # 1D Configs
    curve_interp_methods: np.ndarray,
    curve_extrap_methods: np.ndarray,
    raw_discount_tenors: np.ndarray,
    raw_discount_ln_dfs: np.ndarray,
    raw_price_curve_tenors: np.ndarray,
    raw_price_curve_values: np.ndarray,
    # 2D Configs & Surfaces
    surface_interp_x_methods: np.ndarray,  # Time interpolation config
    surface_extrap_x_methods: np.ndarray,  # Time extrapolation config
    surface_y_axis_type_ids: np.ndarray,  # Y-Axis Config
    raw_vol_surface_tenors: np.ndarray,  # (C_surf, M_tenors)
    raw_vol_surface_strikes: np.ndarray,  # (C_surf, K_strikes)
    raw_vol_surface_values: np.ndarray,  # (C_surf, M_tenors, K_strikes, S_scenarios)
) -> MarketDataMatrix:

    D = len(portfolio.unique_dates_ordinal32)

    # 1. DISCOUNT GRID
    if D == 0:
        discount_grid = np.empty(
            (raw_discount_ln_dfs.shape[0], 0, num_scenarios), dtype=np.float64
        )
    else:
        discount_grid = interp_log_linear_flat_extrap_3d(
            tenors=raw_discount_tenors,
            values=raw_discount_ln_dfs,
            targets=portfolio.unique_dates_ordinal32,
        )

    # 2. PRICE GRID
    price_curve_grid = _build_curve_grid(
        raw_tenors=raw_price_curve_tenors,
        raw_values=raw_price_curve_values,
        target_dates=portfolio.unique_dates_ordinal32,
        interp_methods=curve_interp_methods,
        extrap_methods=curve_extrap_methods,
    )

    # 3. TTE GRID
    as_of_ord = date_to_ordinal32(as_of_date)
    tte_grid = np.maximum(
        (portfolio.unique_dates_ordinal32 - as_of_ord) / 365.25, 1e-9
    )

    # 4. VOLATILITY SURFACES (Phase 1 Time-Slicing via Reshape Trick)
    if np.any(surface_y_axis_type_ids == SURFACE_Y_AXIS_TYPE_DELTA_CALL):
        raise ValueError(
            "FATAL: Delta-quoted Volatility Surfaces detected in runtime builder. "
            "DELTA_CALL surfaces must be pre-processed and converted to STRIKE or MONEYNESS "
            "by the Calibration module before being passed to the MarketDataMatrix."
        )

    if D == 0 or raw_vol_surface_values.size == 0:
        C_surf = (
            raw_vol_surface_tenors.shape[0]
            if raw_vol_surface_tenors.ndim > 0
            else 0
        )
        K_strikes = (
            raw_vol_surface_strikes.shape[1]
            if raw_vol_surface_strikes.ndim > 1
            else 0
        )
        vol_surfaces = np.empty(
            (C_surf, 0, K_strikes, num_scenarios),
            dtype=np.float64,
        )
    else:
        C_surf, M_tenors, K_strikes, S_scen = raw_vol_surface_values.shape

        # Flatten Strike & Scenarios
        flat_surfaces = raw_vol_surface_values.reshape(
            C_surf,
            M_tenors,
            K_strikes * S_scen,
        )

        time_sliced_smiles_flat = _build_curve_grid(
            raw_tenors=raw_vol_surface_tenors,
            raw_values=flat_surfaces,
            target_dates=portfolio.unique_dates_ordinal32,
            interp_methods=surface_interp_x_methods,
            extrap_methods=surface_extrap_x_methods,
        )

        # Reshape back to 4D
        vol_surfaces = time_sliced_smiles_flat.reshape(
            C_surf,
            D,
            K_strikes,
            S_scen,
        )

    return MarketDataMatrix(
        num_scenarios=num_scenarios,
        fx_spots=raw_fx_spots,
        prices=raw_prices,
        forward_nodes=raw_forward_nodes,
        vol_nodes=raw_vol_nodes,
        discount_factor_grid=discount_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=tte_grid,
        vol_surface_strikes=raw_vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )
