# src/riskquant/market_data/state.py

# src/riskquant/market_data/state.py

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDataMatrix:
    """
    Vectorized market data container.
    Dimension 0 always corresponds to the CompiledPortfolio UIX spaces.
    Dimension -1 always corresponds to the Scenario Index (num_scenarios).
    """

    num_scenarios: int

    # ========== 1. SCALARS / NODES ==========

    # FX Spot Rates. Shape: (num_fx_factors, num_scenarios)
    # Accessed via: block.fx_under_to_settle_uix and block.fx_strike_to_settle_uix
    fx_spots: np.ndarray

    # Underlying Spot Prices. Shape: (num_price_factors, num_scenarios)
    # Accessed via: block.price_uix (for SpotBlock)
    prices: np.ndarray

    # Forward Prices (Discrete Nodes). Shape: (num_fwd_nodes, num_scenarios)
    # Accessed via: block.price_uix (for ForwardNodeBlock)
    forward_nodes: np.ndarray

    # Volatility (Discrete Nodes). Shape: (num_vol_nodes, num_scenarios)
    # Accessed via: block.vol_uix (for VanillaEuropeanOptionNodeBlock)
    vol_nodes: np.ndarray

    # ========== 2. THE MASTER DISCOUNT GRID ==========
    # Pre-computed Discount Factors for every curve at every unique portfolio date.
    # Shape: (num_discount_curves, num_unique_dates, num_scenarios)
    # Accessed via: market_state.discount_factor_grid[curve_uix, date_uix, :]
    discount_factor_grid: np.ndarray

    # ========== 3. CURVE-BASED DATA (CURVE BLOCKS) ==========

    # Pre-computed Forward Prices for every curve at every unique portfolio date.
    # Shape: (num_price_curves, num_unique_dates, num_scenarios)
    # Accessed via: market_state.price_curve_grid[curve_uix, date_uix, :]
    price_curve_grid: np.ndarray

    # ========== 4. TIME GRIDS ==========
    # Pre-computed Time to Expiry (in years) for every unique portfolio date.
    # Shape: (num_unique_dates,)
    # Accessed via: market_state.tte_grid[block.expiry_date_uix]
    tte_grid: np.ndarray

    # ========== 5. SURFACES ==========
    # The strike/moneyness coordinates for each surface's smile.
    # Shape: (num_vol_surfaces, max_vol_strikes)
    vol_surface_strikes: np.ndarray

    # Pre-computed Time-Sliced Volatility Smiles for every unique portfolio date.
    # Shape: (num_vol_surfaces, num_unique_dates, max_vol_strikes, num_scenarios)
    # Accessed via: market_state.vol_surfaces[block.vol_surface_uix, block.expiry_date_uix, :, :]
    vol_surfaces: np.ndarray
