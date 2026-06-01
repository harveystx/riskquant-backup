# src/riskquant/ingest_fixtures/constants.py

"""
Constants used across the ingest fixtures codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ==============================================================================
# RANDOM NUMBER GENERATION
# ==============================================================================

# Seed for RNG in ingest fixtures. This is a constant to ensure reproducibility of
# any random data generated for testing purposes.
CANONICAL_RNG_SEED: Final[int] = 49

# ==============================================================================
# BUSINESS DAY CALCULATIONS
# ==============================================================================

# Business days per year for Brownian-motion scaling.
BUSINESS_DAYS_PER_YEAR: Final[int] = 252

# Number of business days of history to synthesize.
CANONICAL_HISTORY_BUSINESS_DAYS: Final[int] = 260


# ==============================================================================
# GLOBAL HISTORICAL SIMULATION CONSTANTS
# ==============================================================================


# Output dtype for the synthetic history. float32 matches the downstream
# RiskFactorMatrix default; float64 would be wasted precision.
HISTORY_DTYPE: Final[type[np.floating]] = np.float32


# ==============================================================================
# MARKET RISK FACTOR SIMULATION CONSTANTS
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class _MarketRiskFactorParams:
    """Per-factor GBM/ABM parameters used to seed the synthetic history."""

    s0: float
    annualized_vol: float
    is_rate: bool  # True -> arithmetic BM; False -> geometric BM


# Defaults applied per asset class / unit type. Keep them in one block so
# tuning is obvious and no values are buried inside the universe loop.
EQUITY_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=185.50,
    annualized_vol=0.25,
    is_rate=False,
)
FX_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=0.9250,
    annualized_vol=0.08,
    is_rate=False,
)
RATE_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=4.00,
    annualized_vol=0.0075,
    is_rate=True,
)
GAS_COMMODITY_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=2.50,
    annualized_vol=0.35,
    is_rate=False,
)
OIL_COMMODITY_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=80.00,
    annualized_vol=0.35,
    is_rate=False,
)
VOL_OF_VOL_DEFAULTS: Final[_MarketRiskFactorParams] = _MarketRiskFactorParams(
    s0=0.35,
    annualized_vol=0.50,
    is_rate=False,
)
