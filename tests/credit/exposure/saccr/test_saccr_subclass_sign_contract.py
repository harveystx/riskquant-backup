# tests/credit/exposure/saccr/test_saccr_subclass_sign_contract.py
"""
P2-03: signed vs absolute subclass quantity in the SA-CCR add-on.

The quantity under audit is the subclass-level add-on in
`credit/exposure/saccr/aggregate.py::compute_saccr_add_on`:

    D_jk = sum_i D_i                      subclass effective notional (SIGNED sum)
    A_jk = SF_k * |D_jk|                  <- old (absolute)
    A_jk = SF_k *  D_jk                   <- Basel CRE52 / P2-03 decision (signed)
    A_hs = sqrt( (rho * sum_k A_jk)^2 + (1 - rho^2) * sum_k A_jk^2 )

Basel permits FULL offset within a subclass and PARTIAL offset between
subclasses of one hedging set. The partial offset lives entirely in the
`rho * sum_k A_jk` term, and only exists if A_jk keeps its sign.

What these tests establish, separately:

  1. The PUBLIC boundary cannot carry a sign. `TradeRow.volume` and
     `TradeRow.current_price` are both `ge=0.0`, and there is no supervisory
     delta anywhere (`resolve.py`: "CEM: d=MF=1"). Every D_i >= 0, so every
     D_jk >= 0, so `abs` is the identity and production equals Basel exactly.
  2. Within a subclass both conventions net identically (signed sum first).
  3. Across subclasses of one hedging set, with signed input, the signed Basel
     form and the old absolute form DIVERGE. The owner decision (P2-03) is to
     KEEP THE SIGN: `A_jk = SF_k * D_jk`. Production now matches the signed
     Basel oracle on those inputs; the old absolute form is retained here only
     as `signed_type_add_on=False`, to make the behavioural change visible.

The oracle is a dict/loop written here. It imports nothing from production.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.evaluators.kernels.saccr import _kernel_saccr
from riskquant.credit.exposure.saccr.aggregate import compute_saccr_add_on
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig
from riskquant.solutions.credit_exposure.schemas import TradeRow

SF = {0: 0.18, 1: 0.18, 2: 0.40}  # subclass -> supervisory factor
HS = {0: 0, 1: 0, 2: 1}  # subclass -> hedging set (0 and 1 share one)
RHO = {0: 0.40, 1: 0.60}  # hedging set -> rho
ALPHA, FLOOR = 1.4, 0.05
X, Y = 1000.0, 600.0


# --------------------------------------------------------------------------
# Independent oracle — no production import, no numpy vectorisation
# --------------------------------------------------------------------------
def oracle_add_on(notionals, subclasses, *, signed_type_add_on: bool) -> float:
    """`signed_type_add_on=True` is the Basel form; False reproduces SF*|D|."""
    d: dict[tuple[int, int], float] = defaultdict(float)
    for q, k in zip(notionals, subclasses, strict=True):
        d[(HS[k], k)] += q  # full offset within subclass
    by_hs: dict[int, list[float]] = defaultdict(list)
    for (hs, k), d_jk in d.items():
        by_hs[hs].append(SF[k] * (d_jk if signed_type_add_on else abs(d_jk)))
    total = 0.0
    for hs, a in by_hs.items():
        r = RHO[hs]
        total += math.sqrt((r * sum(a)) ** 2 + (1.0 - r * r) * sum(v * v for v in a))
    return total


def production_add_on(notionals, subclasses) -> float:
    n = len(notionals)
    factors = CompiledAddOnFactors(
        subclass_uix=np.array(sorted(SF), dtype=np.int32),
        factor=np.array([SF[k] for k in sorted(SF)]),
    )
    config = CompiledSaccrConfig(
        subclass_uix=np.array(sorted(HS), dtype=np.int32),
        subclass_hedging_set=np.array([HS[k] for k in sorted(HS)], dtype=np.int32),
        hedging_set_uix=np.array(sorted(RHO), dtype=np.int32),
        hedging_set_rho=np.array([RHO[k] for k in sorted(RHO)]),
        alpha=ALPHA,
        multiplier_floor=FLOOR,
    )
    out = compute_saccr_add_on(
        netting_set_uids=np.array([1], dtype=np.int64),
        trade_netting_set_id=np.ones(n, dtype=np.int64),
        trade_effective_notional=np.array(notionals, dtype=np.float64),
        trade_subclass_uix=np.array(subclasses, dtype=np.int32),
        add_on_factors=factors,
        saccr_config=config,
    )
    return float(out[0])


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


# ==========================================================================
# 1. THE PUBLIC BOUNDARY CANNOT CARRY A SIGN
# ==========================================================================
@pytest.mark.parametrize("field,value", [("volume", -100.0), ("current_price", -5.0)])
def test_public_trade_row_rejects_a_signed_notional_input(field, value) -> None:
    """effective_notional = volume * current_price, both ge=0: the sign never
    enters. This is the precondition that makes `abs` harmless in production."""
    kwargs = {"volume": 100.0, "current_price": 5.0, field: value}
    with pytest.raises(ValidationError, match="greater than or equal"):
        TradeRow(
            trade_uid=1, netting_set_uid=1, mtm=0.0, asset_subclass_uix=0, **kwargs
        )


def test_zero_notional_is_legal_at_the_public_boundary() -> None:
    row = TradeRow(
        trade_uid=1,
        netting_set_uid=1,
        mtm=0.0,
        volume=0.0,
        current_price=5.0,
        asset_subclass_uix=0,
    )
    assert row.volume == 0.0


@pytest.mark.parametrize(
    "notionals,subclasses",
    [
        ([X], [0]),
        ([0.0], [0]),
        ([X, Y], [0, 0]),  # same subclass
        ([X, Y], [0, 1]),  # two subclasses, one hedging set
        ([X, Y, 250.0], [0, 1, 2]),  # two hedging sets
    ],
)
def test_on_the_public_domain_production_equals_the_basel_form(
    notionals, subclasses
) -> None:
    """Every D_jk >= 0 here, so SF*|D| and SF*D coincide exactly."""
    basel = oracle_add_on(notionals, subclasses, signed_type_add_on=True)
    assert _close(production_add_on(notionals, subclasses), basel)


def test_public_domain_equivalence_holds_over_random_non_negative_books() -> None:
    rng = np.random.default_rng(20260921)
    for _ in range(200):
        n = int(rng.integers(1, 8))
        notionals = rng.uniform(0.0, 5_000.0, n).tolist()
        subclasses = rng.integers(0, 3, n).tolist()
        basel = oracle_add_on(notionals, subclasses, signed_type_add_on=True)
        assert _close(production_add_on(notionals, subclasses), basel)


# ==========================================================================
# 2. WITHIN A SUBCLASS — signed netting, both conventions agree
# ==========================================================================
@pytest.mark.parametrize(
    "label,notionals,expected",
    [
        ("positive", [X], 0.18 * X),
        ("negative", [-X], 0.18 * X),  # single entry: |A| either way
        ("zero", [0.0], 0.0),
        ("same_sign", [X, Y], 0.18 * (X + Y)),
        ("opposite_unequal", [X, -Y], 0.18 * (X - Y)),  # FULL offset, not X + Y
        ("exact_cancellation", [X, -X], 0.0),
    ],
)
def test_within_subclass_quantities_net_before_any_sign_is_removed(
    label, notionals, expected
) -> None:
    """A single-subclass hedging set collapses to SF*|D_jk| for any rho, so the
    hand value is exact. Opposite-sign trades must NET — the result is
    SF*|x-y|, never SF*(|x|+|y|)."""
    subclasses = [0] * len(notionals)
    got = production_add_on(notionals, subclasses)
    assert _close(got, expected), (label, got, expected)
    assert _close(got, oracle_add_on(notionals, subclasses, signed_type_add_on=True))


# ==========================================================================
# 3. ACROSS SUBCLASSES — the conventions diverge on signed input
# ==========================================================================
def test_closed_forms_for_an_exactly_offsetting_cross_subclass_pair() -> None:
    """+x in subclass 0, -x in subclass 1, same hedging set, A = SF*x = 180.

    Basel (signed A):   sqrt((rho*(A - A))^2 + (1-rho^2)*2A^2) = A*sqrt(2(1-rho^2))
    Absolute (|A|):     sqrt((rho*(A + A))^2 + (1-rho^2)*2A^2) = A*sqrt(2(1+rho^2))
    """
    a, r = SF[0] * X, RHO[0]
    signed_basel_closed = a * math.sqrt(2.0 * (1.0 - r * r))
    absolute_closed = a * math.sqrt(2.0 * (1.0 + r * r))

    # Both closed forms against the independent oracle, then production pinned to
    # the signed Basel form (the implemented P2-03 decision).
    assert _close(
        oracle_add_on([X, -X], [0, 1], signed_type_add_on=True), signed_basel_closed
    )
    assert _close(
        oracle_add_on([X, -X], [0, 1], signed_type_add_on=False), absolute_closed
    )
    assert absolute_closed > signed_basel_closed  # dropping the sign can only add
    assert _close(production_add_on([X, -X], [0, 1]), signed_basel_closed)


@pytest.mark.parametrize(
    "notionals,subclasses",
    [
        ([X, -Y], [0, 1]),
        ([X, -X], [0, 1]),
        ([X, -Y, 250.0], [0, 1, 2]),
    ],
)
def test_signed_cross_subclass_input_matches_the_basel_form(
    notionals, subclasses
) -> None:
    """P2-03 CLOSED. Production preserves the subclass add-on sign, so an
    opposite-sign pair in two subclasses of one hedging set now partially
    offsets through the rho-weighted term — matching the signed Basel oracle and
    NO LONGER matching the absolute form."""
    basel = oracle_add_on(notionals, subclasses, signed_type_add_on=True)
    absolute = oracle_add_on(notionals, subclasses, signed_type_add_on=False)
    assert _close(production_add_on(notionals, subclasses), basel)
    assert not _close(basel, absolute)  # these inputs genuinely distinguish the two


# ==========================================================================
# 4. DOWNSTREAM — the add-on reaches EAD through the verified kernel
# ==========================================================================
def _ead_via_kernel(v: float, c: float, add_on: float) -> float:
    out = np.empty((1, 1), dtype=np.float64, order="C")
    _kernel_saccr(
        out,
        np.ascontiguousarray(np.array([[v]])),
        np.array([0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([c]),
        np.array([add_on]),
        ALPHA,
        FLOOR,
    )
    return float(out[0, 0])


def _ead_closed_form(v: float, c: float, add_on: float) -> float:
    rc = max(v - c, 0.0)
    mult = min(
        1.0, FLOOR + (1.0 - FLOOR) * math.exp((v - c) / (2.0 * (1.0 - FLOOR) * add_on))
    )
    return ALPHA * (rc + mult * add_on)


@pytest.mark.parametrize("v_minus_c", [500.0, 0.0, -100.0, -1000.0])
def test_public_domain_ead_equals_the_basel_ead(v_minus_c) -> None:
    """On the public domain the add-on is Basel-exact, so EAD is too."""
    notionals, subclasses = [X, Y, 250.0], [0, 1, 2]
    add_on = production_add_on(notionals, subclasses)
    basel = oracle_add_on(notionals, subclasses, signed_type_add_on=True)
    assert math.isclose(
        _ead_via_kernel(v_minus_c, 0.0, add_on),
        _ead_closed_form(v_minus_c, 0.0, basel),
        rel_tol=1e-12,
    )


def test_an_add_on_overstatement_is_amplified_when_out_of_the_money() -> None:
    """Why P2-03 matters if signed input is ever admitted. AddOn enters EAD twice
    — directly in PFE and inside the multiplier exponent — and both push the
    same way, so for V - C < 0 the EAD overstatement exceeds the add-on's.

    Pure propagation mathematics between the two candidate contracts, computed
    from the oracle alone, so it holds whichever way P2-03 is decided."""
    notionals, subclasses = [X, -X], [0, 1]
    absolute = oracle_add_on(notionals, subclasses, signed_type_add_on=False)
    basel = oracle_add_on(notionals, subclasses, signed_type_add_on=True)
    add_on_ratio = absolute / basel

    itm = _ead_closed_form(0.0, 0.0, absolute) / _ead_closed_form(0.0, 0.0, basel)
    otm = _ead_closed_form(-1000.0, 0.0, absolute) / _ead_closed_form(
        -1000.0, 0.0, basel
    )
    assert math.isclose(
        itm, add_on_ratio, rel_tol=1e-12
    )  # mult = 1: ratio passes through
    assert otm > add_on_ratio  # multiplier channel amplifies
