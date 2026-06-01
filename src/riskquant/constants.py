# src/riskquant/constants.py

"""
Common constants used across the riskquant codebase.

This module separates:
- typed sentinel values used in numeric payloads / arrays
- general configuration constants used by the portfolio engine
"""

from __future__ import annotations

from datetime import date

import numpy as np

# ==============================================================================
# Typed sentinel values for payload fields
# ==============================================================================

# Universal sentinel for "no valid ID".
# Used across instrument_ids, underlying_ids, volatility_ids, and discount_curve_ids.
NO_ID = np.int32(-999_999)

# Universal sentinel for missing dates (settlement, delivery, expiry).
# Using -1 allows for clean `date_ord >= 0` masking.
NO_DATE_ORD32 = np.int32(-99_999)

# Specific sentinel for FX: No conversion required (Underlying CCY == Settlement CCY)
NO_FX_RF_REQUIRED = np.int32(-1)

# Invalid UIX sentinel for any universe index (*_uix fields).
# Used exclusively in the Execution Engine (blocks/compilers), never in Ingest.
INVALID_UIX = np.int32(-999)

# NOT IMPLEMENTED sentinels for fields expected to be populated but aren't yet.
NOT_IMPLEMENTED_SENTINEL: np.int8 = np.int8(-99)

# ==============================================================================
# Option encoding
# ==============================================================================

# 1 for call options, -1 for put options, 0 for not-an-option.
# Allows for branchless payoff calculation: payoff * cp_mult
CALL_OPTION_MULT = np.int8(1)
PUT_OPTION_MULT = np.int8(-1)
NOT_OPTION_MULT = np.int8(0)

# ==============================================================================
# Date encoding
# ==============================================================================

DATE_EPOCH: date = date(1970, 1, 1)

# ==============================================================================
# Numeric configuration
# ==============================================================================

ROUND_DECIMALS_LEVEL = 8
ROUND_DECIMALS_QUANTITY = 8
ROUND_DECIMALS_STRIKE_PRICE = 8

POSITION_DUST_TOLERANCE = 1e-8
PD_DUST_TOLERANCE = 1e-10

# ============================================
# --- DEVELOPMENT-ONLY FALLBACK VALUES
# ============================================

FALLBACK_MOCK_VOLATILITY_DO_NOT_SHIP: float = 0.50

# ==============================================================================
# --- EWMA Volatility Calculation Constants
# ==============================================================================

EWMA_DECAY_FACTOR: float = 0.94
MAX_EWMA_VOL_RATIO: float = 10.0
MIN_EWMA_VARIANCE: float = 1e-12

# ==============================================================================
# --- Credit Risk Constants
# ==============================================================================

NO_PARENT_OBLIGOR: np.int64 = np.int64(-1)
NO_RATING_ID: np.int8 = np.int8(0)
NO_SECTOR_ID: np.int32 = np.int32(-1)
NO_COUNTRY_ID: np.int32 = np.int32(-1)

# ==============================================================================
# RANDOM SEED FOR TESTING
# ==============================================================================

RNG_SEED: int = 42
