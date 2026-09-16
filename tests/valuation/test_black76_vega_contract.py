# tests/valuation/test_black76_vega_contract.py
"""
VAL-01 regression: `_black76_fv_vega`'s documented domain contract.

    vega == 0.0   iff   vol <= 0  or  sqrt_tte <= 0

The contract is the function's own docstring, which also names RQ-AUD-002 as
the remediation that introduced the guard and describes the exact regression it
prevents. D045 established that the guard is absent from the shipped source.

These tests assert the DOCUMENTED CONTRACT, not the current behaviour. They are
expected to fail until the prepared VAL-01 patch is applied. Do not relax them
to accommodate the defect -- the failure is the evidence.
"""

from __future__ import annotations

import math

import pytest

from riskquant.constants import IV_SOLVER_FAILED_SENTINEL
from riskquant.valuation.pricing_models.black76.model import _black76_fv_vega

F = 100.0
T = 1.0
SQRT_T = math.sqrt(T)
HALF_T = 0.5 * T


@pytest.mark.parametrize(
    "label,strike,sqrt_tte,half_tte,vol",
    [
        (
            "IV solver failure sentinel",
            100.0,
            SQRT_T,
            HALF_T,
            IV_SOLVER_FAILED_SENTINEL,
        ),
        ("zero vol, at the money", 100.0, SQRT_T, HALF_T, 0.0),
        ("zero vol, out of the money", 120.0, SQRT_T, HALF_T, 0.0),
        ("zero vol, in the money", 80.0, SQRT_T, HALF_T, 0.0),
        ("zero sqrt_tte", 100.0, 0.0, 0.0, 0.20),
        ("negative sqrt_tte", 100.0, -1.0, HALF_T, 0.20),
        ("negative vol, not the sentinel", 100.0, SQRT_T, HALF_T, -0.35),
    ],
)
def test_invalid_domain_returns_exactly_zero(label, strike, sqrt_tte, half_tte, vol):
    assert (
        _black76_fv_vega(
            fwd=F, strike=strike, sqrt_tte=sqrt_tte, half_tte=half_tte, vol=vol
        )
        == 0.0
    ), label


@pytest.mark.parametrize(
    "strike,tte,vol",
    [
        (100.0, 1.0, 0.20),
        (120.0, 0.5, 0.35),
        (3900.0, 2.0, 0.18),
        (0.90, 0.25, 0.09),
    ],
)
def test_valid_domain_matches_the_analytic_formula(strike, tte, vol):
    """vega = F * phi(d1) * sqrt(T), with an independent scipy phi."""
    from scipy.stats import norm

    fwd = {100.0: 100.0, 120.0: 100.0, 3900.0: 4200.0, 0.90: 0.85}[strike]
    vol_sqrt_tte = vol * math.sqrt(tte)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / vol_sqrt_tte
    want = fwd * float(norm.pdf(d1)) * math.sqrt(tte)
    got = _black76_fv_vega(
        fwd=fwd, strike=strike, sqrt_tte=math.sqrt(tte), half_tte=0.5 * tte, vol=vol
    )
    assert got == pytest.approx(want, rel=1e-12)


def test_valid_domain_vega_is_strictly_positive():
    assert (
        _black76_fv_vega(
            fwd=F, strike=100.0, sqrt_tte=SQRT_T, half_tte=HALF_T, vol=0.20
        )
        > 0.0
    )
