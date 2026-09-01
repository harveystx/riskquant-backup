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
# GLOBAL SENTINELS
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
# OPTION ENCODING
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
# CONFIGURATION CONSTANTS
# ==============================================================================

ROUND_DECIMALS_LEVEL = 8
ROUND_DECIMALS_QUANTITY = 8
ROUND_DECIMALS_STRIKE_PRICE = 8

POSITION_DUST_TOLERANCE = 1e-8
PD_DUST_TOLERANCE = 1e-10

# ==============================================================================
# DEVELOPMENT-ONLY FALLBACK VALUES
# ==============================================================================

FALLBACK_MOCK_VOLATILITY_DO_NOT_SHIP: float = 0.50

# ==============================================================================
# EWMA Volatility Calculation Constants
# ==============================================================================

EWMA_DECAY_FACTOR: float = 0.94
MAX_EWMA_VOL_RATIO: float = 10.0
MIN_EWMA_VARIANCE: float = 1e-12

# ==============================================================================
# RANDOM SEED FOR TESTING
# ==============================================================================

RNG_SEED: int = 42

# ==============================================================================
# DAYS PER YEAR FOR TTE CALCULATION
# ==============================================================================

DAYS_PER_YR_ACT_365_25: float = 365.25
TTE_FLOOR: float = 1e-9

# ==============================================================================
# IMPLIED VOL CALIBRATION CONSTANTS
# ==============================================================================

#: Failure sentinel written into the iv output for non-invertible option quotes.
#: Negative by construction, so the downstream contract ``success_mask = iv >= 0.0``
#: flags it as a failed solve. Imported by the Black-76 scalar solver and asserted
#: against by the CalibratedOptionQuotes master.
IV_SOLVER_FAILED_SENTINEL = -1.0

#: Reliability floor for calibrated implied vols, expressed as vega / forward.
#: A solved iv reprices its target even when vega is ~0, but such a vol is
#: under-determined (deep ITM/OTM, ultra-short tenors). Quotes with
#: ``vega < MIN_RELIABLE_VEGA_OVER_FWD * fwd`` are gated out by
#: ``CalibratedOptionQuotes.reliable_mask``. Heuristic; tune per surface.
MIN_RELIABLE_VEGA_OVER_FWD: float = 1e-3


# ==============================================================================
# DEPENDENCE ESTIMATION CONSTANTS
# ==============================================================================

#: Minimum observations (S) required by every dependence estimator. Two points
#: define a line, so Pearson is computable at S = 2 (and degenerate: |rho| = 1).
#: Below that the sample covariance is undefined.
MIN_DEPENDENCE_OBSERVATIONS: int = 2

#: Minimum variables (K) required. A 1x1 dependence matrix carries no
#: information; pairwise estimators need at least one off-diagonal pair.
MIN_DEPENDENCE_VARIABLES: int = 2

#: Maximum observation count. Forced by ``sort_order`` being ``np.int32`` and,
#: independently, by the requirement that average ranks (integers and
#: half-integers up to S) be exactly representable in float64.
MAX_DEPENDENCE_OBSERVATIONS: int = 2**31

#: Closed interval containing every value a correlation-like estimator can
#: return in exact arithmetic.
CORRELATION_LOWER_BOUND: float = -1.0
CORRELATION_UPPER_BOUND: float = +1.0

#: Exact diagonal of a correlation-like dependence matrix. Pearson, Spearman
#: and Kendall tau-b all equal 1 for a variable against itself; the value is
#: asserted, never estimated.
UNIT_DEPENDENCE_DIAGONAL: float = 1.0

#: Engine-wide numerical excursion tolerance for declared value ranges.
#:
#: This is a NUMERICAL tolerance, not a statistical parameter. It bounds the
#: floating-point error of a length-S inner product of unit-norm rows, whose
#: worst case grows like S * eps: at S = 1e6 that is about 2.2e-10. A tolerance
#: of 1e-9 sits two orders above the achievable rounding error and nine orders
#: below any statistically meaningful difference.
#:
#: An excursion inside the tolerance is clamped to the declared bound. An
#: excursion beyond it is an implementation defect and RAISES. A finite declared
#: range alone never licenses clamping.
NUMERICAL_EXCURSION_TOLERANCE: float = 1e-9

#: MAD-to-sigma consistency factor for a normal distribution: 1 / Phi^-1(0.75).
MAD_TO_SIGMA_NORMAL: float = 1.4826

#: Default degrees-of-freedom correction for cross-sectional volatility
#: normalisation in the market-state indices.
MARKET_STATE_DEFAULT_DDOF: int = 1


###############################################################################
#                                                                             #
#                            CREDIT RISK CONSTANTS                            #
#                                                                             #
###############################################################################

# ==============================================================================
# CREDIT RISK: OBLIGOR RELATED CONSTANTS
# ==============================================================================

NO_PARENT_OBLIGOR: np.int64 = np.int64(-1)
NO_RATING_ID: np.int8 = np.int8(-1)
NO_SECTOR_ID: np.int32 = np.int32(-1)
NO_COUNTRY_ID: np.int32 = np.int32(-1)

# Degrees-of-freedom sentinel for non-chi credit factors. nu is strictly
# positive for a chi-square mixing factor, so a negative value is an
# unambiguous "this factor carries no nu" marker (sentinel over NaN, per the
# core blocks contract).
NO_CHI_NU: np.float64 = np.float64(-1.0)

# Beta(alpha, beta) parameter sentinel for non-BETA_PD credit factors. Both
# shape parameters are strictly positive for a Beta factor, so a negative value
# is an unambiguous "this factor carries no alpha/beta" marker.
NO_BETA_PARAM: np.float64 = np.float64(-1.0)


# ==============================================================================
# CREDIT RISK: EXPOSURE ENGINE CONSTANTS
# ==============================================================================


# --- Current Exposure Method (CEM) --------------------------------------------

#: CEM net-PFE compression: ``Net_PFE = gross_PFE * (FLOOR + COEFF * NGR)``.
#: The 0.4 floor retains 40% of the gross add-on even under perfect netting
#: (NGR -> 0) as a supervisory buffer for basis / operational / roll risk; the
#: 0.6 coefficient scales the remaining fraction by the netting-to-gross ratio.
#: Legacy Basel CEM values, kept for the interim CEM engine (SA-CCR's PFE
#: multiplier supersedes them).
CEM_PFE_FLOOR_DEFAULT: float = 0.4
CEM_PFE_COEFFICIENT_DEFAULT: float = 0.6


# --- SA-CCR (config-driven; these are overridable defaults, not the taxonomy) ---
#: Supervisory alpha in EAD = alpha * (RC + PFE). Basel default 1.4; a house
#: model may recalibrate it.
SACCR_ALPHA_DEFAULT: float = 1.4

#: PFE-multiplier floor in min(1, floor + (1-floor)*exp((V-C)/(2*(1-floor)*AddOn))).
SACCR_MULTIPLIER_FLOOR_DEFAULT: float = 0.05
