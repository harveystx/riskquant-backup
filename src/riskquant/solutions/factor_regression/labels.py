# src/riskquant/solutions/factor_regression/labels.py
"""
Code-to-name and name-to-code mappings for the factor-regression library (C.22).

Every ``*_CODES`` tuple of ``codes.py`` has exactly one name per code (the
constant name from ``codes.py``) and one reverse map from name to ``np.int8``
code.  ``DIAGNOSTIC_MESSAGES`` supplies the default message text for each
diagnostic code (A.4, C.11).

S6.T10 verifies exhaustive coverage: for every group, every code in
``codes.*_CODES`` is in the corresponding ``*_NAMES`` dict, and the reverse
map recovers the original code.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from riskquant.factor_regression import codes

# ------------------------------------------------------------------ transforms

TRANSFORM_NAMES: Final[dict[int, str]] = {
    int(codes.LEVEL): "LEVEL",
    int(codes.LOG_RETURN): "LOG_RETURN",
    int(codes.SIMPLE_RETURN): "SIMPLE_RETURN",
    int(codes.DIFFERENCE): "DIFFERENCE",
}

TRANSFORM_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "LEVEL": codes.LEVEL,
    "LOG_RETURN": codes.LOG_RETURN,
    "SIMPLE_RETURN": codes.SIMPLE_RETURN,
    "DIFFERENCE": codes.DIFFERENCE,
}

# ------------------------------------------------------------ covariance kinds
# Values (0, 1, 2) are part of the canonical spec bytes (A.6); do not renumber.

COVARIANCE_NAMES: Final[dict[int, str]] = {
    int(codes.IID): "IID",
    int(codes.HC0): "HC0",
    int(codes.HAC_BARTLETT): "HAC_BARTLETT",
}

COVARIANCE_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "IID": codes.IID,
    "HC0": codes.HC0,
    "HAC_BARTLETT": codes.HAC_BARTLETT,
}

# ----------------------------------------------------------- exclusion reasons

EXCLUSION_NAMES: Final[dict[int, str]] = {
    int(codes.OUTSIDE_WINDOW): "OUTSIDE_WINDOW",
    int(codes.NOT_COMMON_DATE): "NOT_COMMON_DATE",
    int(codes.NOT_USED_BY_ROWS): "NOT_USED_BY_ROWS",
}

EXCLUSION_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "OUTSIDE_WINDOW": codes.OUTSIDE_WINDOW,
    "NOT_COMMON_DATE": codes.NOT_COMMON_DATE,
    "NOT_USED_BY_ROWS": codes.NOT_USED_BY_ROWS,
}

# ----------------------------------------------------------------- diagnostics

DIAGNOSTIC_NAMES: Final[dict[int, str]] = {
    int(codes.CONDITION_ABOVE_THRESHOLD): "CONDITION_ABOVE_THRESHOLD",
    int(codes.LEVEL_TERM_PRESENT): "LEVEL_TERM_PRESENT",
    int(codes.R2_ABOVE_DURBIN_WATSON): "R2_ABOVE_DURBIN_WATSON",
    int(codes.OVERLAP_NOT_CORRECTED): "OVERLAP_NOT_CORRECTED",
}

DIAGNOSTIC_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "CONDITION_ABOVE_THRESHOLD": codes.CONDITION_ABOVE_THRESHOLD,
    "LEVEL_TERM_PRESENT": codes.LEVEL_TERM_PRESENT,
    "R2_ABOVE_DURBIN_WATSON": codes.R2_ABOVE_DURBIN_WATSON,
    "OVERLAP_NOT_CORRECTED": codes.OVERLAP_NOT_CORRECTED,
}

# Default message text for each diagnostic code (A.4).
# The facade uses these to build Diagnostic.message when no override is given.
DIAGNOSTIC_MESSAGES: Final[dict[str, str]] = {
    "CONDITION_ABOVE_THRESHOLD": (
        "The condition number of the scaled design exceeds the declared threshold."
    ),
    "LEVEL_TERM_PRESENT": (
        "At least one term uses the LEVEL transform. Level regressions may be spurious."
    ),
    "R2_ABOVE_DURBIN_WATSON": (
        "R-squared exceeds the Durbin-Watson statistic with a LEVEL term present. "
        "The regression may be spurious (Granger-Newbold heuristic)."
    ),
    "OVERLAP_NOT_CORRECTED": (
        "Overlapping return windows are present (q > 0) but the declared covariance "
        "estimator does not correct for the resulting serial correlation."
    ),
}

# ---------------------------------------- CoreInputError reasons

INPUT_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.INPUT_NO_SERIES): "INPUT_NO_SERIES",
    int(codes.INPUT_COUNT_MISMATCH): "INPUT_COUNT_MISMATCH",
    int(codes.INPUT_NOT_NDARRAY): "INPUT_NOT_NDARRAY",
    int(codes.INPUT_WRONG_DTYPE): "INPUT_WRONG_DTYPE",
    int(codes.INPUT_WRONG_NDIM): "INPUT_WRONG_NDIM",
    int(codes.INPUT_WRONG_SHAPE): "INPUT_WRONG_SHAPE",
    int(codes.INPUT_EMPTY): "INPUT_EMPTY",
    int(codes.INPUT_TOO_LONG): "INPUT_TOO_LONG",
    int(codes.INPUT_NOT_STRICTLY_INCREASING): "INPUT_NOT_STRICTLY_INCREASING",
    int(codes.INPUT_NON_FINITE): "INPUT_NON_FINITE",
}

INPUT_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "INPUT_NO_SERIES": codes.INPUT_NO_SERIES,
    "INPUT_COUNT_MISMATCH": codes.INPUT_COUNT_MISMATCH,
    "INPUT_NOT_NDARRAY": codes.INPUT_NOT_NDARRAY,
    "INPUT_WRONG_DTYPE": codes.INPUT_WRONG_DTYPE,
    "INPUT_WRONG_NDIM": codes.INPUT_WRONG_NDIM,
    "INPUT_WRONG_SHAPE": codes.INPUT_WRONG_SHAPE,
    "INPUT_EMPTY": codes.INPUT_EMPTY,
    "INPUT_TOO_LONG": codes.INPUT_TOO_LONG,
    "INPUT_NOT_STRICTLY_INCREASING": codes.INPUT_NOT_STRICTLY_INCREASING,
    "INPUT_NON_FINITE": codes.INPUT_NON_FINITE,
}

# ---------------------------------------- CoreAlignmentError reasons

ALIGNMENT_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.ALIGN_EMPTY_WINDOW): "ALIGN_EMPTY_WINDOW",
    int(codes.ALIGN_WINDOW_OUT_OF_RANGE): "ALIGN_WINDOW_OUT_OF_RANGE",
    int(codes.ALIGN_EMPTY_INTERSECTION): "ALIGN_EMPTY_INTERSECTION",
}

ALIGNMENT_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "ALIGN_EMPTY_WINDOW": codes.ALIGN_EMPTY_WINDOW,
    "ALIGN_WINDOW_OUT_OF_RANGE": codes.ALIGN_WINDOW_OUT_OF_RANGE,
    "ALIGN_EMPTY_INTERSECTION": codes.ALIGN_EMPTY_INTERSECTION,
}

# ---------------------------------------- CoreParameterError reasons

PARAMETER_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.PARAM_EMPTY_DESIGN): "PARAM_EMPTY_DESIGN",
    int(codes.PARAM_UNKNOWN_TRANSFORM): "PARAM_UNKNOWN_TRANSFORM",
    int(codes.PARAM_NEGATIVE_STEP): "PARAM_NEGATIVE_STEP",
    int(codes.PARAM_STRIDE_BELOW_ONE): "PARAM_STRIDE_BELOW_ONE",
    int(codes.PARAM_ZERO_STEP_WITH_RETURN): "PARAM_ZERO_STEP_WITH_RETURN",
    int(codes.PARAM_POSITIVE_STEP_LEVEL_ONLY): "PARAM_POSITIVE_STEP_LEVEL_ONLY",
    int(codes.PARAM_RANK_RTOL_OUT_OF_RANGE): "PARAM_RANK_RTOL_OUT_OF_RANGE",
    int(codes.PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE): (
        "PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE"
    ),
    int(codes.PARAM_UNKNOWN_COVARIANCE): "PARAM_UNKNOWN_COVARIANCE",
    int(codes.PARAM_LAG_NOT_ALLOWED): "PARAM_LAG_NOT_ALLOWED",
    int(codes.PARAM_NEGATIVE_LAG): "PARAM_NEGATIVE_LAG",
    int(codes.PARAM_LAG_ABOVE_LIMIT): "PARAM_LAG_ABOVE_LIMIT",
}

PARAMETER_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "PARAM_EMPTY_DESIGN": codes.PARAM_EMPTY_DESIGN,
    "PARAM_UNKNOWN_TRANSFORM": codes.PARAM_UNKNOWN_TRANSFORM,
    "PARAM_NEGATIVE_STEP": codes.PARAM_NEGATIVE_STEP,
    "PARAM_STRIDE_BELOW_ONE": codes.PARAM_STRIDE_BELOW_ONE,
    "PARAM_ZERO_STEP_WITH_RETURN": codes.PARAM_ZERO_STEP_WITH_RETURN,
    "PARAM_POSITIVE_STEP_LEVEL_ONLY": codes.PARAM_POSITIVE_STEP_LEVEL_ONLY,
    "PARAM_RANK_RTOL_OUT_OF_RANGE": codes.PARAM_RANK_RTOL_OUT_OF_RANGE,
    "PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE": (
        codes.PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE
    ),
    "PARAM_UNKNOWN_COVARIANCE": codes.PARAM_UNKNOWN_COVARIANCE,
    "PARAM_LAG_NOT_ALLOWED": codes.PARAM_LAG_NOT_ALLOWED,
    "PARAM_NEGATIVE_LAG": codes.PARAM_NEGATIVE_LAG,
    "PARAM_LAG_ABOVE_LIMIT": codes.PARAM_LAG_ABOVE_LIMIT,
}

# ---------------------------------------- CoreRankError reasons

RANK_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.RANK_ZERO_COLUMN): "RANK_ZERO_COLUMN",
    int(codes.RANK_DEFICIENT): "RANK_DEFICIENT",
}

RANK_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "RANK_ZERO_COLUMN": codes.RANK_ZERO_COLUMN,
    "RANK_DEFICIENT": codes.RANK_DEFICIENT,
}

# ---------------------------------------- CoreSampleSizeError reasons

SAMPLE_SIZE_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.SAMPLE_STEP_NOT_BELOW_COUNT): "SAMPLE_STEP_NOT_BELOW_COUNT",
    int(codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS): "SAMPLE_ROWS_NOT_ABOVE_COLUMNS",
    int(codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM): "SAMPLE_ROWS_BELOW_TABLE_MINIMUM",
}

SAMPLE_SIZE_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "SAMPLE_STEP_NOT_BELOW_COUNT": codes.SAMPLE_STEP_NOT_BELOW_COUNT,
    "SAMPLE_ROWS_NOT_ABOVE_COLUMNS": codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS,
    "SAMPLE_ROWS_BELOW_TABLE_MINIMUM": codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM,
}

# ---------------------------------------- CoreDegenerateFitError reasons

DEGENERATE_FIT_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.DEGENERATE_EXACT_FIT): "DEGENERATE_EXACT_FIT",
    int(codes.DEGENERATE_ZERO_RESPONSE_VARIANCE): "DEGENERATE_ZERO_RESPONSE_VARIANCE",
    int(codes.DEGENERATE_NON_POSITIVE_VARIANCE): "DEGENERATE_NON_POSITIVE_VARIANCE",
}

DEGENERATE_FIT_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "DEGENERATE_EXACT_FIT": codes.DEGENERATE_EXACT_FIT,
    "DEGENERATE_ZERO_RESPONSE_VARIANCE": codes.DEGENERATE_ZERO_RESPONSE_VARIANCE,
    "DEGENERATE_NON_POSITIVE_VARIANCE": codes.DEGENERATE_NON_POSITIVE_VARIANCE,
}

# ---------------------------------------- CoreRestrictionError reasons

RESTRICTION_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.RESTRICTION_WRONG_SHAPE): "RESTRICTION_WRONG_SHAPE",
    int(codes.RESTRICTION_NON_FINITE): "RESTRICTION_NON_FINITE",
    int(codes.RESTRICTION_RANK_DEFICIENT): "RESTRICTION_RANK_DEFICIENT",
    int(codes.RESTRICTION_NOT_POSITIVE_DEFINITE): "RESTRICTION_NOT_POSITIVE_DEFINITE",
}

RESTRICTION_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "RESTRICTION_WRONG_SHAPE": codes.RESTRICTION_WRONG_SHAPE,
    "RESTRICTION_NON_FINITE": codes.RESTRICTION_NON_FINITE,
    "RESTRICTION_RANK_DEFICIENT": codes.RESTRICTION_RANK_DEFICIENT,
    "RESTRICTION_NOT_POSITIVE_DEFINITE": codes.RESTRICTION_NOT_POSITIVE_DEFINITE,
}

# ---------------------------------------------------------- combination scales
# Values (0, 1) enter the unit-root spec digest (A.3 of
# WS_FACTOR_REGRESSION_COINTEGRATION); do not renumber.

SCALE_NAMES: Final[dict[int, str]] = {
    int(codes.SCALE_LEVEL): "SCALE_LEVEL",
    int(codes.SCALE_LOG): "SCALE_LOG",
}

SCALE_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "SCALE_LEVEL": codes.SCALE_LEVEL,
    "SCALE_LOG": codes.SCALE_LOG,
}

# ---------------------------------------------------- ADF deterministic cases

DETERMINISTIC_NAMES: Final[dict[int, str]] = {
    int(codes.DET_CONSTANT): "DET_CONSTANT",
    int(codes.DET_CONSTANT_TREND): "DET_CONSTANT_TREND",
}

DETERMINISTIC_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "DET_CONSTANT": codes.DET_CONSTANT,
    "DET_CONSTANT_TREND": codes.DET_CONSTANT_TREND,
}

# ------------------------------------------------------------ unit-root tests

TEST_KIND_NAMES: Final[dict[int, str]] = {
    int(codes.TEST_ADF): "TEST_ADF",
}

TEST_KIND_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "TEST_ADF": codes.TEST_ADF,
}

# ---------------------------------------- CoreScaleDomainError reasons

SCALE_DOMAIN_ERROR_NAMES: Final[dict[int, str]] = {
    int(codes.SCALE_LOG_LEVEL_NOT_POSITIVE): "SCALE_LOG_LEVEL_NOT_POSITIVE",
}

SCALE_DOMAIN_ERROR_NAME_TO_CODE: Final[dict[str, np.int8]] = {
    "SCALE_LOG_LEVEL_NOT_POSITIVE": codes.SCALE_LOG_LEVEL_NOT_POSITIVE,
}

# ------------------------------------------------------------------ registry
# One entry per *_CODES group.  Used by S6.T10 to verify exhaustive coverage.
# Each entry: (codes_tuple, name_map, code_from_name_map).
# The type uses object to keep the declaration simple; the test checks values.

ALL_CODE_GROUPS: Final[
    dict[
        str,
        tuple[
            tuple[np.int8, ...],
            dict[int, str],
            dict[str, np.int8],
        ],
    ]
] = {
    "TRANSFORM": (
        codes.TRANSFORM_CODES,
        TRANSFORM_NAMES,
        TRANSFORM_NAME_TO_CODE,
    ),
    "COVARIANCE": (
        codes.COVARIANCE_CODES,
        COVARIANCE_NAMES,
        COVARIANCE_NAME_TO_CODE,
    ),
    "EXCLUSION": (
        codes.EXCLUSION_CODES,
        EXCLUSION_NAMES,
        EXCLUSION_NAME_TO_CODE,
    ),
    "DIAGNOSTIC": (
        codes.DIAGNOSTIC_CODES,
        DIAGNOSTIC_NAMES,
        DIAGNOSTIC_NAME_TO_CODE,
    ),
    "INPUT_ERROR": (
        codes.INPUT_ERROR_CODES,
        INPUT_ERROR_NAMES,
        INPUT_ERROR_NAME_TO_CODE,
    ),
    "ALIGNMENT_ERROR": (
        codes.ALIGNMENT_ERROR_CODES,
        ALIGNMENT_ERROR_NAMES,
        ALIGNMENT_ERROR_NAME_TO_CODE,
    ),
    "PARAMETER_ERROR": (
        codes.PARAMETER_ERROR_CODES,
        PARAMETER_ERROR_NAMES,
        PARAMETER_ERROR_NAME_TO_CODE,
    ),
    "RANK_ERROR": (
        codes.RANK_ERROR_CODES,
        RANK_ERROR_NAMES,
        RANK_ERROR_NAME_TO_CODE,
    ),
    "SAMPLE_SIZE_ERROR": (
        codes.SAMPLE_SIZE_ERROR_CODES,
        SAMPLE_SIZE_ERROR_NAMES,
        SAMPLE_SIZE_ERROR_NAME_TO_CODE,
    ),
    "DEGENERATE_FIT_ERROR": (
        codes.DEGENERATE_FIT_ERROR_CODES,
        DEGENERATE_FIT_ERROR_NAMES,
        DEGENERATE_FIT_ERROR_NAME_TO_CODE,
    ),
    "RESTRICTION_ERROR": (
        codes.RESTRICTION_ERROR_CODES,
        RESTRICTION_ERROR_NAMES,
        RESTRICTION_ERROR_NAME_TO_CODE,
    ),
    "SCALE": (
        codes.SCALE_CODES,
        SCALE_NAMES,
        SCALE_NAME_TO_CODE,
    ),
    "DETERMINISTIC": (
        codes.DETERMINISTIC_CODES,
        DETERMINISTIC_NAMES,
        DETERMINISTIC_NAME_TO_CODE,
    ),
    "TEST_KIND": (
        codes.TEST_KIND_CODES,
        TEST_KIND_NAMES,
        TEST_KIND_NAME_TO_CODE,
    ),
    "SCALE_DOMAIN_ERROR": (
        codes.SCALE_DOMAIN_ERROR_CODES,
        SCALE_DOMAIN_ERROR_NAMES,
        SCALE_DOMAIN_ERROR_NAME_TO_CODE,
    ),
}

# ------------------------------------------------------------------ unit-root texts
# Fixed text strings per deterministic case (A.2, A.4 of
# WS_FACTOR_REGRESSION_COINTEGRATION).  The keys are the int forms of the
# DET_CONSTANT and DET_CONSTANT_TREND codes so that api.py can look them up
# directly from ``int(spec.deterministic.value)``.

NULL_HYPOTHESIS_TEXTS: Final[dict[int, str]] = {
    int(codes.DET_CONSTANT): (
        "H0: the combination has a unit root (gamma = 0) and no drift. "
        "H1: it is stationary around a constant."
    ),
    int(codes.DET_CONSTANT_TREND): (
        "H0: the combination has a unit root (gamma = 0). "
        "H1: it is stationary around a linear trend."
    ),
}

CRITICAL_VALUE_SOURCES: Final[dict[int, str]] = {
    int(codes.DET_CONSTANT): (
        "MacKinnon (2010) QED WP 1227, Table 2 tau_c, N=1, declared weights"
    ),
    int(codes.DET_CONSTANT_TREND): (
        "MacKinnon (2010) QED WP 1227, Table 3 tau_ct, N=1, declared weights"
    ),
}
