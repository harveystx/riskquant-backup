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
# These must stay numpy scalars where dtype compatibility matters.

# Sentinel for "no valid ID" in any of the ID fields (instrument, universe index, risk factor, curve).
NO_UIX = np.int32(-1)
NO_PRICE_ID = np.int32(-2)
NO_PRICE_CURVE_ID = np.int32(-3)
NO_VOL_ID = np.int32(-4)

# Sentinel for "no discount curve" in the discount curve linkage fields.
NO_DISCOUNT_CURVE_ID = np.int32(-5)

# No FX rate required (e.g. for SPOT trades where the underlying and settlement currencies are the same).
NO_FX_RF_REQUIRED = np.int32(-6)

# Sentinels for missing dates in the date fields (settlement, delivery, expiry).
NO_DELIVERY_DATE_ORDINAL32 = np.int32(-7)
NO_SETTLEMENT_DATE_ORDINAL32 = np.int32(-8)
NO_EXPIRY_DATE_ORDINAL32 = np.int32(-9)

# Sentinel for "not an option" in the is_call field.
NOT_OPTION = np.int8(-10)


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

# ==========================================
# --- NAMING CONVENTIONS
# ==========================================

# Prefix expected by the engine to map an underlying asset to its volatility surface/node
VOL_RISK_FACTOR_PREFIX: str = "vol_"
