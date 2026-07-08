# src/riskquant/basel/irb/correlation.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant.basel.constants.irb import (
    # IRB CORRELATION CONSTANTS
    IRB_CORRELATION_DECAY_FACTOR,
    IRB_CORRELATION_DENOMINATOR,
    IRB_CORRELATION_LOWER_BOUND,
    IRB_CORRELATION_UPPER_BOUND,
    # HVCRE CORRELATION CONSTANTS
    IRB_HVCRE_CORRELATION_UPPER_BOUND,
    # RETAIL CORRELATION CONSTANTS
    IRB_RETAIL_MORTGAGE_CORRELATION,
    IRB_RETAIL_OTHER_DECAY_FACTOR,
    IRB_RETAIL_OTHER_DENOMINATOR,
    IRB_RETAIL_OTHER_LOWER_BOUND,
    IRB_RETAIL_OTHER_UPPER_BOUND,
    IRB_RETAIL_QRRE_CORRELATION,
    # SME CORRELATION CONSTANTS
    IRB_SME_ADJ_DIVISOR,
    IRB_SME_ADJ_MAX_MULTIPLIER,
    IRB_SME_SALES_MAX_EUR_M,
    IRB_SME_SALES_MIN_EUR_M,
)

_F64 = NDArray[np.float64]


def basel_corporate_correlation(pd_vals: _F64) -> _F64:
    """
    Calculate the Basel IRB asset correlation for Corporate, Sovereign, and Bank exposures.

    Ref: Basel Consolidated Framework CRE31.9

    The correlation R(PD) is a decreasing function of PD, reflecting that higher
    credit quality (lower PD) is more sensitive to systemic risk, while lower
    credit quality (higher PD) is driven more by idiosyncratic risk.
    """
    w = (
        1.0 - np.exp(IRB_CORRELATION_DECAY_FACTOR * pd_vals)
    ) / IRB_CORRELATION_DENOMINATOR
    return IRB_CORRELATION_LOWER_BOUND * w + IRB_CORRELATION_UPPER_BOUND * (1.0 - w)


def basel_hvcre_correlation(pd_vals: _F64) -> _F64:
    """
    Calculate the Basel IRB asset correlation for High-Volatility Commercial Real Estate.

    Ref: Basel Consolidated Framework CRE31.11

    Functions identically to standard Corporate correlation, but with a higher upper
    bound (0.30 instead of 0.24) to reflect the heightened systemic sensitivity of HVCRE.
    """
    w = (
        1.0 - np.exp(IRB_CORRELATION_DECAY_FACTOR * pd_vals)
    ) / IRB_CORRELATION_DENOMINATOR
    return IRB_CORRELATION_LOWER_BOUND * w + IRB_HVCRE_CORRELATION_UPPER_BOUND * (
        1.0 - w
    )


def basel_sme_correlation(
    *,
    pd_vals: _F64,
    sales_eur_m: _F64,
) -> _F64:
    """
    Calculate the Basel IRB asset correlation with the SME firm-size adjustment.

    Ref: Basel Consolidated Framework CRE31.10

    Applies a downward adjustment to the standard corporate correlation for Small and
    Medium Enterprises (SMEs) with annual sales between €5M and €50M.
    """
    s = np.clip(
        a=sales_eur_m,
        a_min=IRB_SME_SALES_MIN_EUR_M,
        a_max=IRB_SME_SALES_MAX_EUR_M,
    )
    size_adj = IRB_SME_ADJ_MAX_MULTIPLIER * (
        1.0 - (s - IRB_SME_SALES_MIN_EUR_M) / IRB_SME_ADJ_DIVISOR
    )
    return basel_corporate_correlation(pd_vals) - size_adj


def basel_retail_mortgage_correlation(pd_vals: _F64) -> _F64:
    """
    Provide the Basel IRB asset correlation for Retail Residential Mortgages.

    Ref: Basel Consolidated Framework CRE31.14

    Mortgages possess a fixed asset correlation of 0.15, regardless of PD.
    Returns an array of shape matched to `pd_vals` to maintain kernel signature.
    """
    return np.full_like(pd_vals, IRB_RETAIL_MORTGAGE_CORRELATION)


def basel_retail_qrre_correlation(pd_vals: _F64) -> _F64:
    """
    Provide the Basel IRB asset correlation for Qualifying Revolving Retail Exposures.

    Ref: Basel Consolidated Framework CRE31.14

    QRRE (e.g., credit cards) possess a highly idiosyncratic risk profile, yielding
    a fixed, low asset correlation of 0.04, regardless of PD.
    Returns an array of shape matched to `pd_vals` to maintain kernel signature.
    """
    return np.full_like(pd_vals, IRB_RETAIL_QRRE_CORRELATION)


def basel_retail_other_correlation(pd_vals: _F64) -> _F64:
    """
    Calculate the Basel IRB asset correlation for Other Retail exposures.

    Ref: Basel Consolidated Framework CRE31.14

    Utilizes a dynamic correlation function similar to corporates, but with a
    steeper decay factor (-35 instead of -50) and lower bounds (0.03 to 0.16).
    """
    w = (
        1.0 - np.exp(IRB_RETAIL_OTHER_DECAY_FACTOR * pd_vals)
    ) / IRB_RETAIL_OTHER_DENOMINATOR
    return IRB_RETAIL_OTHER_LOWER_BOUND * w + IRB_RETAIL_OTHER_UPPER_BOUND * (1.0 - w)
