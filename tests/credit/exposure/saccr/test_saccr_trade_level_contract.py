# tests/credit/exposure/saccr/test_saccr_trade_level_contract.py
"""
P2-11A / P2-11B: SA-CCR trade-level effective notional D_i = d_i * MF_i * delta_i.

Owner decision A1 + B1 (locked): the public SA-CCR path represents explicit
long/short direction (supervisory delta = +-1) and explicit remaining maturity
(unmargined CRE52 maturity factor). These are STRUCTURAL CRE52 terms, not house
parameters.

The oracle below computes d, delta, MF and D independently — it imports nothing
from `add_on.resolve` and never calls the production SA-CCR trade-level
derivation. The hedging-set aggregation / EAD it reuses is the CRE52 closed form
(already independently verified in Phase 2 and by the P2-03 sign-contract suite);
these tests prove REACHABILITY of that verified core from the new public fields,
not a re-audit of it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.saccr.aggregate import compute_saccr_add_on
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig
from riskquant.solutions.credit_exposure.run import run_credit_exposure
from riskquant.solutions.credit_exposure.schemas import CreditExposureRequest

SF, RHO, ALPHA, FLOOR = 0.18, 0.40, 1.4, 0.05
FLOOR_M = 10.0 / 250.0  # CRE52.48-49 ten-business-day floor, in years


# --------------------------------------------------------------------------
# Independent oracle — no production trade-level import
# --------------------------------------------------------------------------
def oracle_mf(m: float) -> float:
    """MF = sqrt(clamp(M, 10/250, 1)), CRE52.48-49, for a valid positive M."""
    return math.sqrt(min(max(m, FLOOR_M), 1.0))


def oracle_delta(direction: str) -> float:
    return 1.0 if direction == "long" else -1.0


def oracle_d(volume: float, price: float) -> float:
    return volume * price


def oracle_effective_notional(volume, price, direction, m) -> float:
    return oracle_d(volume, price) * oracle_mf(m) * oracle_delta(direction)


def oracle_add_on(effective_notionals) -> float:
    """Single subclass, single hedging set: A = SF * sum(D_i) (signed), then the
    CRE52 single-factor aggregation."""
    a = SF * sum(effective_notionals)
    return math.sqrt((RHO * a) ** 2 + (1.0 - RHO**2) * a * a)


def oracle_ead(effective_notionals, v: float = 0.0, c: float = 0.0) -> float:
    add_on = oracle_add_on(effective_notionals)
    rc = max(v - c, 0.0)
    if add_on <= 0.0:
        mult = 1.0
    else:
        mult = min(
            1.0,
            FLOOR + (1.0 - FLOOR) * math.exp((v - c) / (2.0 * (1.0 - FLOOR) * add_on)),
        )
    return ALPHA * (rc + mult * add_on)


def _request(trades):
    """trades: list of (volume, price, direction, maturity), all one subclass/ns."""
    return CreditExposureRequest(
        model="saccr",
        trades=[
            {
                "trade_uid": i + 1,
                "netting_set_uid": 1,
                "mtm": 0.0,
                "volume": v,
                "current_price": p,
                "asset_subclass_uix": 0,
                "direction": d,
                "remaining_maturity_years": m,
            }
            for i, (v, p, d, m) in enumerate(trades)
        ],
        netting_sets=[{"netting_set_uid": 1, "obligor_uid": 9, "net_collateral": 0.0}],
        add_on_factors=[{"subclass_uix": 0, "factor": SF}],
        subclass_hedging_sets=[{"subclass_uix": 0, "hedging_set_uix": 0}],
        hedging_set_correlations=[{"hedging_set_uix": 0, "rho": RHO}],
    )


def _public_ead(trades) -> float:
    return float(run_credit_exposure(_request(trades)).netting_set_ead[0].ead)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


# ==========================================================================
# 1. INDEPENDENT NUMERICAL CASE (the §19 table)
# ==========================================================================
@pytest.mark.parametrize(
    "units,price,direction,m,exp_d,exp_delta,exp_mf,exp_dd",
    [
        (1000.0, 1.0, "long", 5.0, 1000.0, 1.0, 1.0, 1000.0),
        (1000.0, 1.0, "short", 5.0, 1000.0, -1.0, 1.0, -1000.0),
        (1000.0, 1.0, "long", 0.25, 1000.0, 1.0, 0.5, 500.0),
        (2500.0, 1.0, "long", 5.0, 2500.0, 1.0, 1.0, 2500.0),
    ],
)
def test_trade_level_effective_notional_table(
    units, price, direction, m, exp_d, exp_delta, exp_mf, exp_dd
) -> None:
    """d, delta, MF, D each verified against the hand table, then D recovered
    from the public EAD of a single-trade book (EAD = alpha*SF*|D| at M>=... )."""
    assert _close(oracle_d(units, price), exp_d)
    assert _close(oracle_delta(direction), exp_delta)
    assert _close(oracle_mf(m), exp_mf)
    assert _close(oracle_effective_notional(units, price, direction, m), exp_dd)

    # A single trade's public EAD is alpha * SF * |D| (one-entry hedging set,
    # V-C=0 so mult=1). Recover |D| and check it equals |exp_dd|.
    ead = _public_ead([(units, price, direction, m)])
    recovered_abs_d = ead / (ALPHA * SF)
    assert _close(recovered_abs_d, abs(exp_dd))


# ==========================================================================
# 2. DIRECTION (§13)
# ==========================================================================
def test_single_long_has_positive_delta_and_reaches_core() -> None:
    ead = _public_ead([(1000.0, 1.0, "long", 5.0)])
    assert _close(
        ead, oracle_ead([oracle_effective_notional(1000.0, 1.0, "long", 5.0)])
    )


def test_single_short_has_negative_delta() -> None:
    """A single short's add-on is SF*|-D|, so its EAD equals the long's — the
    sign shows up only in netting. Verified against the oracle either way."""
    short = _public_ead([(1000.0, 1.0, "short", 5.0)])
    long_ = _public_ead([(1000.0, 1.0, "long", 5.0)])
    assert _close(short, long_)  # single-trade magnitude identical
    assert _close(
        short, oracle_ead([oracle_effective_notional(1000.0, 1.0, "short", 5.0)])
    )


def test_long_plus_short_same_subclass_nets_signed() -> None:
    """long 1000 + short 600, M>=1y: signed D = +400, add-on = SF*400."""
    trades = [(1000.0, 1.0, "long", 5.0), (600.0, 1.0, "short", 5.0)]
    ead = _public_ead(trades)
    oracle = oracle_ead([oracle_effective_notional(*t) for t in trades])
    assert _close(ead, oracle)
    # sanity: this is the netted 400, not the summed 1600
    assert _close(ead, ALPHA * SF * 400.0)


def test_full_offset_gives_zero_add_on() -> None:
    """long 1000 + short 1000: signed D = 0, add-on = 0, EAD = 0. The case the
    old public path could not represent."""
    ead = _public_ead([(1000.0, 1.0, "long", 5.0), (1000.0, 1.0, "short", 5.0)])
    assert _close(ead, 0.0)


def test_signed_notionals_reach_the_p2_03_core() -> None:
    """Reachability proof: feed the same signed D the public path produces into
    compute_saccr_add_on directly and confirm the add-on matches. The core is
    NOT re-implemented here — it is the production function, exercised with the
    signed input the new public fields now supply."""
    trades = [(1000.0, 1.0, "long", 5.0), (600.0, 1.0, "short", 5.0)]
    signed_d = np.array(
        [oracle_effective_notional(*t) for t in trades], dtype=np.float64
    )
    factors = CompiledAddOnFactors(
        subclass_uix=np.array([0], dtype=np.int32), factor=np.array([SF])
    )
    config = CompiledSaccrConfig(
        subclass_uix=np.array([0], dtype=np.int32),
        subclass_hedging_set=np.array([0], dtype=np.int32),
        hedging_set_uix=np.array([0], dtype=np.int32),
        hedging_set_rho=np.array([RHO]),
        alpha=ALPHA,
        multiplier_floor=FLOOR,
    )
    core_add_on = float(
        compute_saccr_add_on(
            netting_set_uids=np.array([1], dtype=np.int64),
            trade_netting_set_id=np.array([1, 1], dtype=np.int64),
            trade_effective_notional=signed_d,
            trade_subclass_uix=np.array([0, 0], dtype=np.int32),
            add_on_factors=factors,
            saccr_config=config,
        )[0]
    )
    assert _close(core_add_on, oracle_add_on(signed_d.tolist()))
    assert _close(core_add_on, SF * 400.0)  # signed net, not summed


@pytest.mark.parametrize(
    "bad",
    [
        {
            "volume": 1000.0,
            "current_price": 1.0,
            "asset_subclass_uix": 0,
            "remaining_maturity_years": 5.0,
        },  # direction missing
        {
            "volume": 1000.0,
            "current_price": 1.0,
            "asset_subclass_uix": 0,
            "direction": "flat",
            "remaining_maturity_years": 5.0,
        },  # bad direction
    ],
)
def test_missing_or_invalid_direction_fails_fast(bad) -> None:
    with pytest.raises(ValidationError):
        CreditExposureRequest(
            model="saccr",
            trades=[{"trade_uid": 1, "netting_set_uid": 1, "mtm": 0.0, **bad}],
            netting_sets=[
                {"netting_set_uid": 1, "obligor_uid": 9, "net_collateral": 0.0}
            ],
            add_on_factors=[{"subclass_uix": 0, "factor": SF}],
            subclass_hedging_sets=[{"subclass_uix": 0, "hedging_set_uix": 0}],
            hedging_set_correlations=[{"hedging_set_uix": 0, "rho": RHO}],
        )


# ==========================================================================
# 3. MATURITY (§14)
# ==========================================================================
@pytest.mark.parametrize(
    "m,expected_mf",
    [
        (FLOOR_M / 2.0, math.sqrt(FLOOR_M)),  # below floor -> clamped up to floor
        (FLOOR_M, math.sqrt(FLOOR_M)),  # exactly 10 business days
        (1.0 / 12.0, math.sqrt(1.0 / 12.0)),  # 1 month
        (0.25, 0.5),  # 3 months
        (0.5, math.sqrt(0.5)),  # 6 months
        (1.0, 1.0),  # one-year cap boundary
        (5.0, 1.0),  # above cap
    ],
)
def test_maturity_factor_matches_cre52_through_public_path(m, expected_mf) -> None:
    assert _close(oracle_mf(m), expected_mf)
    ead = _public_ead([(1000.0, 1.0, "long", m)])
    # single long: EAD = alpha * SF * (1000 * MF)
    assert _close(ead, ALPHA * SF * 1000.0 * expected_mf)


@pytest.mark.parametrize("bad_m", [0.0, -0.5, -1e-9])
def test_non_positive_maturity_fails_fast_not_clamped(bad_m) -> None:
    """An expired / negative maturity is a boundary error, never clamped into
    the 10-business-day floor."""
    with pytest.raises(ValidationError):
        CreditExposureRequest(
            model="saccr",
            trades=[
                {
                    "trade_uid": 1,
                    "netting_set_uid": 1,
                    "mtm": 0.0,
                    "volume": 1000.0,
                    "current_price": 1.0,
                    "asset_subclass_uix": 0,
                    "direction": "long",
                    "remaining_maturity_years": bad_m,
                }
            ],
            netting_sets=[
                {"netting_set_uid": 1, "obligor_uid": 9, "net_collateral": 0.0}
            ],
            add_on_factors=[{"subclass_uix": 0, "factor": SF}],
            subclass_hedging_sets=[{"subclass_uix": 0, "hedging_set_uix": 0}],
            hedging_set_correlations=[{"hedging_set_uix": 0, "rho": RHO}],
        )


# ==========================================================================
# 4. COMPOSITION (§15) + PUBLIC-DOMAIN REGRESSION (§18)
# ==========================================================================
def test_end_to_end_new_fields_affect_the_result() -> None:
    """The same book at two maturities must give different EAD — proving the new
    fields are mathematically wired, not merely stored."""
    short_dated = _public_ead([(1000.0, 1.0, "long", 0.25)])
    long_dated = _public_ead([(1000.0, 1.0, "long", 5.0)])
    assert not _close(short_dated, long_dated)
    assert _close(short_dated, ALPHA * SF * 500.0)  # MF = 0.5
    assert _close(long_dated, ALPHA * SF * 1000.0)  # MF = 1


def test_long_dated_long_reproduces_pre_a1b1_behaviour() -> None:
    """§18 regression: a positive-direction trade at M >= 1y has delta=+1, MF=1,
    so its effective notional is exactly volume*price — the pre-A1/B1 value. This
    is the one case that must be numerically unchanged."""
    ead = _public_ead([(1000.0, 1.0, "long", 5.0)])
    pre_a1b1_effective_notional = 1000.0 * 1.0  # old volume*price, no MF, no delta
    assert _close(ead, ALPHA * SF * pre_a1b1_effective_notional)


# ==========================================================================
# 5. DIRECTION FAIL-FAST HARDENING (P2-11 closure)
# ==========================================================================
def test_recipe_direction_mapping_rejects_unexpected_values() -> None:
    """Defence in depth: the recipe-side direction->sign mapping must reject any
    value that is not exactly 'long' or 'short', rather than folding it to short.
    Exercised directly on the recipe helper, bypassing the Pydantic Literal that
    guards the normal public path."""
    import polars as pl

    from riskquant.solutions.credit_exposure.recipes import _saccr_effective_notional

    base = {
        "volume": [1000.0],
        "current_price": [1.0],
        "remaining_maturity_years": [5.0],
    }

    # legal values map to the right sign
    long_d = _saccr_effective_notional(pl.DataFrame({**base, "direction": ["long"]}))
    short_d = _saccr_effective_notional(pl.DataFrame({**base, "direction": ["short"]}))
    assert long_d[0] == 1000.0
    assert short_d[0] == -1000.0

    # anything else fails fast, and NOT as a silent short
    for bad in ("SHORT", "sideways", "l", ""):
        with pytest.raises(ValueError, match="must be 'long' or 'short'"):
            _saccr_effective_notional(pl.DataFrame({**base, "direction": [bad]}))


def test_public_boundary_still_rejects_unexpected_direction() -> None:
    """The Pydantic Literal remains the first gate; the recipe hardening is a
    second one. Neither silently defaults an unexpected value to short."""
    with pytest.raises(ValidationError):
        _request([(1000.0, 1.0, "buy", 5.0)])  # "buy" is not a Literal member


# ==========================================================================
# 6. from_dataframe threads the SA-CCR fields (P2-11 closure)
# ==========================================================================
def test_from_dataframe_carries_direction_and_maturity() -> None:
    """The desk/notebook ergonomics path must forward the new SA-CCR columns, or
    every SA-CCR request built via from_dataframe fails validation."""
    import polars as pl

    from riskquant.solutions.credit_exposure.run import run_credit_exposure

    trades = pl.DataFrame({
        "trade_uid": [1, 2],
        "netting_set_uid": [1, 1],
        "mtm": [0.0, 0.0],
        "volume": [1000.0, 600.0],
        "current_price": [1.0, 1.0],
        "asset_subclass_uix": [0, 0],
        "direction": ["long", "short"],
        "remaining_maturity_years": [5.0, 5.0],
    })
    netting_sets = pl.DataFrame({
        "netting_set_uid": [1],
        "obligor_uid": [9],
        "net_collateral": [0.0],
    })
    add_on = pl.DataFrame({"subclass_uix": [0], "factor": [SF]})
    hedging = pl.DataFrame({"subclass_uix": [0], "hedging_set_uix": [0]})
    rho = pl.DataFrame({"hedging_set_uix": [0], "rho": [RHO]})

    req = CreditExposureRequest.from_dataframe(
        trades,
        netting_sets,
        model="saccr",
        add_on_factors=add_on,
        subclass_hedging_sets=hedging,
        hedging_set_correlations=rho,
    )
    ead = float(run_credit_exposure(req).netting_set_ead[0].ead)
    # long 1000 + short 600 @5y -> signed net 400 -> EAD = alpha*SF*400
    assert _close(ead, ALPHA * SF * 400.0)
