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

NO_ID = np.int32(-1)
NO_UIX = np.int32(-2)
NO_FX_RF_REQUIRED = np.int32(-3)

NO_DELIVERY_DATE_ORDINAL32 = np.int32(-4)
NO_SETTLEMENT_DATE_ORDINAL32 = np.int32(-5)
NO_EXPIRY_DATE_ORDINAL32 = np.int32(-6)

NOT_OPTION = np.int8(-7)


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
