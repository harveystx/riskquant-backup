# tests/credit/exposure/test_cem_direction_maturity_regression.py
"""
P2-11 CEM regression: CEM must NOT gain SA-CCR's delta / maturity semantics.

The SA-CCR trade-level effective notional became d * MF * delta (P2-11A/B). CEM's
effective notional must remain the bare adjusted notional volume * current_price,
with supervisory delta and maturity factor both fixed at 1 and NOT read from any
trade field. This guards the CEM/SA-CCR separation.
"""

from __future__ import annotations

import math

from riskquant.solutions.credit_exposure.run import run_credit_exposure
from riskquant.solutions.credit_exposure.schemas import CreditExposureRequest

FLOOR_DEFAULT, COEFF_DEFAULT = 0.4, 0.6  # CEM NGR-compression defaults


def _cem_ead(volume, price, mtm, *, extra_trade_fields=None):
    fields = {
        "trade_uid": 1,
        "netting_set_uid": 1,
        "mtm": mtm,
        "volume": volume,
        "current_price": price,
        "asset_subclass_uix": 0,
    }
    if extra_trade_fields:
        fields.update(extra_trade_fields)
    req = CreditExposureRequest(
        model="current_exposure_method",
        trades=[fields],
        netting_sets=[{"netting_set_uid": 1, "obligor_uid": 9, "net_collateral": 0.0}],
        add_on_factors=[{"subclass_uix": 0, "factor": 0.10}],
    )
    return float(run_credit_exposure(req).netting_set_ead[0].ead)


def test_cem_effective_notional_is_volume_times_price_no_mf_no_delta() -> None:
    """In-the-money single trade: NRC = V, GRC = V, NGR = 1, so gross PFE enters
    at full weight. Gross PFE = volume * price * factor = 1000 * 1.0 * 0.10 = 100.
    EAD = NRC + PFE = 100 + 100 = 200, with no maturity or direction scaling."""
    ead = _cem_ead(1000.0, 1.0, mtm=100.0)
    grc = 100.0
    ngr = 1.0
    gross_pfe = 1000.0 * 1.0 * 0.10
    expected = grc + gross_pfe * (FLOOR_DEFAULT + COEFF_DEFAULT * ngr)
    assert math.isclose(ead, expected, rel_tol=1e-12)


def test_cem_ignores_saccr_direction_and_maturity_fields_if_supplied() -> None:
    """Even if a caller attaches SA-CCR fields to a CEM trade, CEM must produce
    the identical result — it does not consume them. A short + short-dated trade
    must give the same CEM EAD as a bare trade."""
    bare = _cem_ead(1000.0, 1.0, mtm=100.0)
    with_saccr_fields = _cem_ead(
        1000.0,
        1.0,
        mtm=100.0,
        extra_trade_fields={"direction": "short", "remaining_maturity_years": 0.02},
    )
    assert math.isclose(bare, with_saccr_fields, rel_tol=1e-12)


def test_cem_does_not_require_direction_or_maturity() -> None:
    """CEM accepts a trade with neither field — they are SA-CCR-only requirements."""
    ead = _cem_ead(500.0, 2.0, mtm=50.0)
    assert ead > 0.0
