# src/riskquant/factor_regression/codes.py
"""
Integer codes of the factor regression core. PRIVATE.

The core holds no names (C.22). Every transform, covariance kind, exclusion
reason, diagnostic and error reason is an ``np.int8`` constant here. The facade
maps each code to its name in ``riskquant.solutions.factor_regression.labels``.

Each group has its own value space and its own ``*_CODES`` tuple. A value is
unique inside its group only, so a label lookup always names the group. The
group tuples are the closed sets that the facade label map must cover (S6.T10).

The transform values equal the ``ShiftType`` values of
``riskquant.market_data.risk_factors.shift_types`` for the three return
transforms. The core does not import that module (S2 prohibition, F-5): the
equality is a reading aid, not a link.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# ------------------------------------------------------------------ transforms

LEVEL: Final = np.int8(0)
LOG_RETURN: Final = np.int8(1)
SIMPLE_RETURN: Final = np.int8(2)
DIFFERENCE: Final = np.int8(3)

TRANSFORM_CODES: Final = (LEVEL, LOG_RETURN, SIMPLE_RETURN, DIFFERENCE)

# ------------------------------------------------------------ covariance kinds
# The values are part of the canonical spec bytes (A.6: IID 0, HC0 1,
# HAC_BARTLETT 2). Do not renumber.

IID: Final = np.int8(0)
HC0: Final = np.int8(1)
HAC_BARTLETT: Final = np.int8(2)

COVARIANCE_CODES: Final = (IID, HC0, HAC_BARTLETT)

# ----------------------------------------------------------- exclusion reasons

OUTSIDE_WINDOW: Final = np.int8(0)
NOT_COMMON_DATE: Final = np.int8(1)
NOT_USED_BY_ROWS: Final = np.int8(2)

EXCLUSION_CODES: Final = (OUTSIDE_WINDOW, NOT_COMMON_DATE, NOT_USED_BY_ROWS)

# ----------------------------------------------------------------- diagnostics

CONDITION_ABOVE_THRESHOLD: Final = np.int8(0)
LEVEL_TERM_PRESENT: Final = np.int8(1)
R2_ABOVE_DURBIN_WATSON: Final = np.int8(2)
OVERLAP_NOT_CORRECTED: Final = np.int8(3)

DIAGNOSTIC_CODES: Final = (
    CONDITION_ABOVE_THRESHOLD,
    LEVEL_TERM_PRESENT,
    R2_ABOVE_DURBIN_WATSON,
    OVERLAP_NOT_CORRECTED,
)

# ------------------------------------------------- CoreInputError reasons

INPUT_NO_SERIES: Final = np.int8(0)
INPUT_COUNT_MISMATCH: Final = np.int8(1)
INPUT_NOT_NDARRAY: Final = np.int8(2)
INPUT_WRONG_DTYPE: Final = np.int8(3)
INPUT_WRONG_NDIM: Final = np.int8(4)
INPUT_WRONG_SHAPE: Final = np.int8(5)
INPUT_EMPTY: Final = np.int8(6)
INPUT_TOO_LONG: Final = np.int8(7)
INPUT_NOT_STRICTLY_INCREASING: Final = np.int8(8)
INPUT_NON_FINITE: Final = np.int8(9)

INPUT_ERROR_CODES: Final = (
    INPUT_NO_SERIES,
    INPUT_COUNT_MISMATCH,
    INPUT_NOT_NDARRAY,
    INPUT_WRONG_DTYPE,
    INPUT_WRONG_NDIM,
    INPUT_WRONG_SHAPE,
    INPUT_EMPTY,
    INPUT_TOO_LONG,
    INPUT_NOT_STRICTLY_INCREASING,
    INPUT_NON_FINITE,
)

# --------------------------------------------- CoreAlignmentError reasons

ALIGN_EMPTY_WINDOW: Final = np.int8(0)
ALIGN_WINDOW_OUT_OF_RANGE: Final = np.int8(1)
ALIGN_EMPTY_INTERSECTION: Final = np.int8(2)

ALIGNMENT_ERROR_CODES: Final = (
    ALIGN_EMPTY_WINDOW,
    ALIGN_WINDOW_OUT_OF_RANGE,
    ALIGN_EMPTY_INTERSECTION,
)

# --------------------------------------------- CoreParameterError reasons

PARAM_EMPTY_DESIGN: Final = np.int8(0)
PARAM_UNKNOWN_TRANSFORM: Final = np.int8(1)
PARAM_NEGATIVE_STEP: Final = np.int8(2)
PARAM_STRIDE_BELOW_ONE: Final = np.int8(3)
PARAM_ZERO_STEP_WITH_RETURN: Final = np.int8(4)
PARAM_POSITIVE_STEP_LEVEL_ONLY: Final = np.int8(5)
PARAM_RANK_RTOL_OUT_OF_RANGE: Final = np.int8(6)
PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE: Final = np.int8(7)
PARAM_UNKNOWN_COVARIANCE: Final = np.int8(8)
PARAM_LAG_NOT_ALLOWED: Final = np.int8(9)
PARAM_NEGATIVE_LAG: Final = np.int8(10)
PARAM_LAG_ABOVE_LIMIT: Final = np.int8(11)
# A term lag with |lag| above the declared max_lag of build_rows
# (WS_FACTOR_REGRESSION_LAGS C.9). Appended last: the values above are in use.
PARAM_LAG_OUTSIDE_WINDOW: Final = np.int8(12)

PARAMETER_ERROR_CODES: Final = (
    PARAM_EMPTY_DESIGN,
    PARAM_UNKNOWN_TRANSFORM,
    PARAM_NEGATIVE_STEP,
    PARAM_STRIDE_BELOW_ONE,
    PARAM_ZERO_STEP_WITH_RETURN,
    PARAM_POSITIVE_STEP_LEVEL_ONLY,
    PARAM_RANK_RTOL_OUT_OF_RANGE,
    PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE,
    PARAM_UNKNOWN_COVARIANCE,
    PARAM_LAG_NOT_ALLOWED,
    PARAM_NEGATIVE_LAG,
    PARAM_LAG_ABOVE_LIMIT,
    PARAM_LAG_OUTSIDE_WINDOW,
)

# -------------------------------------------------- CoreRankError reasons

RANK_ZERO_COLUMN: Final = np.int8(0)
RANK_DEFICIENT: Final = np.int8(1)

RANK_ERROR_CODES: Final = (RANK_ZERO_COLUMN, RANK_DEFICIENT)

# -------------------------------------------- CoreSampleSizeError reasons

SAMPLE_STEP_NOT_BELOW_COUNT: Final = np.int8(0)
SAMPLE_ROWS_NOT_ABOVE_COLUMNS: Final = np.int8(1)
# ADF regression row count n below 20, the smallest sample size of the
# MacKinnon (2010) tables (WS_FACTOR_REGRESSION_COINTEGRATION C.6, A.8).
SAMPLE_ROWS_BELOW_TABLE_MINIMUM: Final = np.int8(2)

SAMPLE_SIZE_ERROR_CODES: Final = (
    SAMPLE_STEP_NOT_BELOW_COUNT,
    SAMPLE_ROWS_NOT_ABOVE_COLUMNS,
    SAMPLE_ROWS_BELOW_TABLE_MINIMUM,
)

# ----------------------------------------- CoreDegenerateFitError reasons

DEGENERATE_EXACT_FIT: Final = np.int8(0)
DEGENERATE_ZERO_RESPONSE_VARIANCE: Final = np.int8(1)
DEGENERATE_NON_POSITIVE_VARIANCE: Final = np.int8(2)

DEGENERATE_FIT_ERROR_CODES: Final = (
    DEGENERATE_EXACT_FIT,
    DEGENERATE_ZERO_RESPONSE_VARIANCE,
    DEGENERATE_NON_POSITIVE_VARIANCE,
)

# ------------------------------------------- CoreRestrictionError reasons

RESTRICTION_WRONG_SHAPE: Final = np.int8(0)
RESTRICTION_NON_FINITE: Final = np.int8(1)
RESTRICTION_RANK_DEFICIENT: Final = np.int8(2)
RESTRICTION_NOT_POSITIVE_DEFINITE: Final = np.int8(3)

RESTRICTION_ERROR_CODES: Final = (
    RESTRICTION_WRONG_SHAPE,
    RESTRICTION_NON_FINITE,
    RESTRICTION_RANK_DEFICIENT,
    RESTRICTION_NOT_POSITIVE_DEFINITE,
)

# ---------------------------------------------------------- combination scales
# WS_FACTOR_REGRESSION_COINTEGRATION A.8. The values enter the unit-root spec
# digest (A.3). Do not renumber. The prefix keeps SCALE_LEVEL apart from the
# transform code LEVEL (F-16).

SCALE_LEVEL: Final = np.int8(0)
SCALE_LOG: Final = np.int8(1)

SCALE_CODES: Final = (SCALE_LEVEL, SCALE_LOG)

# ---------------------------------------------------- ADF deterministic cases
# A.8. The values enter the unit-root spec digest (A.3). Do not renumber.

DET_CONSTANT: Final = np.int8(0)
DET_CONSTANT_TREND: Final = np.int8(1)

DETERMINISTIC_CODES: Final = (DET_CONSTANT, DET_CONSTANT_TREND)

# ------------------------------------------------------------ unit-root tests
# A.8. The value enters the unit-root spec digest (A.3). Do not renumber.

TEST_ADF: Final = np.int8(0)

TEST_KIND_CODES: Final = (TEST_ADF,)

# --------------------------------------------- CoreScaleDomainError reasons

SCALE_LOG_LEVEL_NOT_POSITIVE: Final = np.int8(0)

SCALE_DOMAIN_ERROR_CODES: Final = (SCALE_LOG_LEVEL_NOT_POSITIVE,)
