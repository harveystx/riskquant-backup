# tests/sensitivities/test_vanilla_forward_full_revaluation_reconciliation.py
"""
WS-C (D103), C12 and C17: emitted forward sensitivities against production full
revaluation, in float64.

Repricing is the production valuation path (`assemble_valuation_state` +
`run_valuation`) on a market grid whose columns are the base and the shocked
levels. The Taylor polynomial is evaluated here, from the EMITTED rows only.
Shocked levels come from the three charts (LOG_RETURN, SIMPLE_RETURN,
DIFFERENCE) of each factor's configured shift type.

In DiscountingMode.NONE with an ABSOLUTE book the second-order Taylor
polynomial is exact (PV is bilinear in (F, X_a) and linear in X_k), so

    |dV_reprice - Taylor| <= 2^-48 (sum tau_bar + T_s + T0)          (§5.12)

and without the (F, X_a) pair the residual is Q DF_a dF dX_a, within the same
tolerance. C17 checks the RELATIVE and LOG book conventions produced by
`build_sensitivity_book_spec`: transforms within 2^-50 |value|; RELATIVE is
exact; LOG is not exact, and its residual equals the analytic remainder of the
test-owned closed form within 2^-48 * scale.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_vanilla_forward_sensitizer as tv

from riskquant.market_data.compile import assemble_valuation_state
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.book_adapter import build_sensitivity_book_spec
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

S = 24
TOL = 2.0**-48
CHART = {
    **tv.fx.shift_type_by_name(),
    "TEST_EXACT_SIMPLE[3M]": "SIMPLE_RETURN",
    "TEST_EXACT_EUR[1M]": "LOG_RETURN",
    "USDGBP": "LOG_RETURN",
    "EURGBP": "LOG_RETURN",
}


def _shocked() -> list[dict[str, float]]:
    rng = np.random.default_rng(103)
    columns = []
    for _ in range(S):
        col = {}
        for name, x0 in tv.LEVELS.items():
            chart = CHART.get(name)
            if chart == "LOG_RETURN":
                col[name] = x0 * math.exp(rng.normal(0.0, 0.03))
            elif chart == "SIMPLE_RETURN":
                col[name] = x0 * (1.0 + rng.normal(0.0, 0.03))
            elif chart == "DIFFERENCE":
                col[name] = x0 + rng.normal(0.0, 1.0)
        columns.append(col)
    return columns


@pytest.fixture(scope="module")
def world():
    shocked = _shocked()
    w = tv._build([row for row, _ in tv.TOPOLOGY.values()], columns=[{}, *shocked])
    w.shocked = shocked
    w.sens = tv._sens(w, 0)
    state = assemble_valuation_state(matrix=w.matrix, uix_mapper=w.mapper)
    pv = np.empty((w.portfolio.total_trades, S + 1), dtype=np.float64)
    run_valuation(portfolio=w.portfolio, valuation_state=state, out=pv)
    w.dv = {int(u): pv[i, 1:] - pv[i, 0] for i, u in enumerate(w.portfolio.trade_uids)}
    w.v0 = {int(u): pv[i, 0] for i, u in enumerate(w.portfolio.trade_uids)}
    assert set(w.dv) == set(tv.TOPOLOGY)  # one row per uid
    return w


def _delta(w, name: str) -> np.ndarray:
    return np.array([col[name] for col in w.shocked]) - tv.LEVELS[name]


def _scale(row, view, lv) -> float:
    fname, xa, xk = view
    q, k = row["quantity"], row["strike_price"]
    x_a = 1.0 if xa is None else lv[xa]
    x_k = 1.0 if xk is None else lv[xk]
    return abs(q * lv[fname] * x_a) + abs(q * k * x_k)


def _taylor(w, uid: int, coeffs=None, delta=None, drop_pairs: bool = False):
    """(polynomial, sum tau_bar) per scenario from emitted rows and the oracle A_bar."""
    row, view = tv.TOPOLOGY[uid]
    diag, cross = coeffs if coeffs is not None else tv._emitted(w.sens, uid)
    o_diag, o_cross = tv._oracle(row["quantity"], row["strike_price"], view, tv.LEVELS)
    d = delta or {rf: _delta(w, tv.NAME[rf]) for rf in diag}
    value = np.zeros(S)
    tau = np.zeros(S)
    for rf, (d1, d2) in diag.items():
        value += d1 * d[rf] + 0.5 * d2 * d[rf] ** 2
        tau += o_diag[rf][2] * np.abs(d[rf]) + 0.5 * abs(d2) * d[rf] ** 2
    if not drop_pairs:
        for (i, j), c in cross.items():
            value += c * d[i] * d[j]
            tau += o_cross[(i, j)][1] * np.abs(d[i] * d[j])
    return value, tau


def _t_scales(w, uid: int) -> tuple[np.ndarray, float]:
    row, view = tv.TOPOLOGY[uid]
    ts = np.array([_scale(row, view, {**tv.LEVELS, **col}) for col in w.shocked])
    return ts, _scale(row, view, tv.LEVELS)


# ---- C12 -----------------------------------------------------------------------------


def test_c12_book_covers_all_three_charts() -> None:
    used = {CHART[n] for _, view in tv.TOPOLOGY.values() for n in view if n is not None}
    assert used == {"LOG_RETURN", "SIMPLE_RETURN", "DIFFERENCE"}


@pytest.mark.parametrize("uid", sorted(tv.TOPOLOGY))
def test_c12_taylor_from_emitted_rows_reprices_exactly(world, uid: int) -> None:
    value, tau = _taylor(world, uid)
    ts, t0 = _t_scales(world, uid)
    assert np.all(np.abs(world.dv[uid] - value) <= TOL * (tau + ts + t0))


@pytest.mark.parametrize(
    "uid", [u for u, (_, v) in tv.TOPOLOGY.items() if v[1] is not None]
)
def test_c12_without_the_pair_the_residual_is_the_bilinear_term(
    world, uid: int
) -> None:
    row, (fname, xa, _) = tv.TOPOLOGY[uid]
    value, _ = _taylor(world, uid, drop_pairs=True)
    _, full_tau = _taylor(world, uid)
    ts, t0 = _t_scales(world, uid)
    expected = row["quantity"] * _delta(world, fname) * _delta(world, xa)
    assert np.all(expected != 0.0)
    tol = TOL * (full_tau + ts + t0)
    assert np.all(np.abs((world.dv[uid] - value) - expected) <= tol)


# ---- C17 -----------------------------------------------------------------------------


def _converted(world, convention: DisplacementConvention):
    x0 = {
        int(rf): tv.LEVELS[tv.NAME[int(rf)]]
        for rf in np.unique(world.sens.factor_rf_ids)
    }
    fids = np.asarray(world.layout.required_factor_ids, dtype=np.int32)
    uids = np.array(sorted(world.v0), dtype=np.int64)
    binding = TaylorBaseBinding(
        as_of_ord32=int(date_to_ord32(world.as_of)),
        base_levels_digest=base_levels_digest(
            factor_ids=fids,
            base_levels=np.array([tv.LEVELS[tv.NAME[int(f)]] for f in fids]),
        ),
        discounting_mode=tv.NONE.value,
        static_config_digest=static_config_content_digest(tv.RSC),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=uids, values=np.array([world.v0[int(u)] for u in uids])
        ),
    )
    spec = build_sensitivity_book_spec(
        sensitivities=world.sens,
        required_factor_ids=fids,
        displacement_convention=convention,
        base_levels_by_factor=x0,
        base_binding=binding,
    )
    diag = {
        (int(p), int(f)): (float(a), float(b))
        for p, f, a, b in spec.sensitivities.iter_rows()
    }
    cross = {
        (int(p), int(i), int(j)): float(v)
        for p, i, j, v in spec.cross_sensitivities.iter_rows()
    }
    return x0, diag, cross


@pytest.mark.parametrize(
    "convention", [DisplacementConvention.RELATIVE, DisplacementConvention.LOG]
)
def test_c17_convention_transforms(world, convention) -> None:
    x0, diag, cross = _converted(world, convention)
    s = world.sens
    for p, f, d1, d2 in zip(
        s.position_uids.tolist(),
        s.factor_rf_ids.tolist(),
        s.first_order.tolist(),
        s.second_order.tolist(),
        strict=True,
    ):
        e1 = d1 * x0[f]
        e2 = d2 * x0[f] ** 2 + (
            d1 * x0[f] if convention is DisplacementConvention.LOG else 0.0
        )
        assert abs(diag[(p, f)][0] - e1) <= 2.0**-50 * abs(e1)
        assert abs(diag[(p, f)][1] - e2) <= 2.0**-50 * abs(e2)
    for p, i, j, c in zip(
        s.cross_position_uids.tolist(),
        s.cross_factor_rf_ids_i.tolist(),
        s.cross_factor_rf_ids_j.tolist(),
        s.cross_second_order.tolist(),
        strict=True,
    ):
        e = c * x0[i] * x0[j]
        assert abs(cross[(p, i, j)] - e) <= 2.0**-50 * abs(e)


def _book(diag, cross, uid):
    d = {f: v for (p, f), v in diag.items() if p == uid}
    c = {(i, j): v for (p, i, j), v in cross.items() if p == uid}
    return d, c


@pytest.mark.parametrize("uid", sorted(tv.TOPOLOGY))
def test_c17_relative_book_reprices_exactly(world, uid: int) -> None:
    x0, diag, cross = _converted(world, DisplacementConvention.RELATIVE)
    coeffs = _book(diag, cross, uid)
    r = {f: _delta(world, tv.NAME[f]) / x0[f] for f in coeffs[0]}
    value, _ = _taylor(world, uid, coeffs=coeffs, delta=r)
    _, tau = _taylor(world, uid)  # tolerance scale in the level coordinate
    ts, t0 = _t_scales(world, uid)
    assert np.all(np.abs(world.dv[uid] - value) <= TOL * (tau + ts + t0))


def _oracle_pv(uid: int, lv: dict[str, float]) -> float:
    row, (fname, xa, xk) = tv.TOPOLOGY[uid]
    x_a = 1.0 if xa is None else lv[xa]
    x_k = 1.0 if xk is None else lv[xk]
    return row["quantity"] * (lv[fname] * x_a - row["strike_price"] * x_k)


@pytest.mark.parametrize("uid", sorted(tv.TOPOLOGY))
def test_c17_log_book_residual_is_the_analytic_remainder(world, uid: int) -> None:
    """LOG is not exact: production repricing minus the production LOG
    polynomial equals the closed-form remainder of the test-owned oracle."""
    x0, diag, cross = _converted(world, DisplacementConvention.LOG)
    coeffs = _book(diag, cross, uid)
    y = {f: np.log((_delta(world, tv.NAME[f]) + x0[f]) / x0[f]) for f in coeffs[0]}
    value, _ = _taylor(world, uid, coeffs=coeffs, delta=y)
    _, tau = _taylor(world, uid)  # tolerance scale in the level coordinate
    row, view = tv.TOPOLOGY[uid]
    o_diag, o_cross = tv._oracle(row["quantity"], row["strike_price"], view, tv.LEVELS)
    o_value = np.zeros(S)
    for rf, (d1, d2, _) in o_diag.items():
        o_value += (
            d1 * x0[rf] * y[rf] + 0.5 * (d2 * x0[rf] ** 2 + d1 * x0[rf]) * y[rf] ** 2
        )
    for (i, j), (c, _) in o_cross.items():
        o_value += c * x0[i] * x0[j] * y[i] * y[j]
    exact = np.array([_oracle_pv(uid, {**tv.LEVELS, **col}) for col in world.shocked])
    remainder = exact - _oracle_pv(uid, tv.LEVELS) - o_value
    ts, t0 = _t_scales(world, uid)
    residual = world.dv[uid] - value
    assert np.any(np.abs(remainder) > TOL * (tau + ts + t0))  # LOG is not exact
    assert np.all(np.abs(residual - remainder) <= TOL * (tau + ts + t0))
