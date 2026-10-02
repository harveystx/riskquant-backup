# tests/solutions/risk_run/test_vanilla_forward_node_production.py
"""
WS-B: VANILLA_FORWARD_NODE FULL_REVAL production evidence (D102 evidence).

Every run is `run_scenario_var_es` end to end on the `_fwdfix` world (raw frame
-> ingest -> compile -> market data -> valuation state -> VanillaForwardBlock
-> `_write_full_reval_block` -> PnlGrid -> D081 conversion -> `compute_var_es`
-> RiskRun). Spies call through and record what production hands on: the
shocked levels and the consumed shock columns (`apply_linear_terms_to_base_levels`),
and the conversion inputs and outputs (`convert_scenario_pnl_to_reporting_currency`).

The oracle is `_fwdoracle` (numpy, math, decimal only; checked statically
below). P&L comparisons feed it the CAPTURED production shocked levels, so
level generation and valuation are certified separately (revision 8, §5.2).
Every comparison uses the frozen tolerance of revision 8, §5:

    V0      2^-48 T0_o
    dV      2u|dV_o| + 2^-48 (T_s,o + T0_o) + phi
    dW      |r_s| tol_dV + |r_s - r_0| tol_V0 + 2^-48 (|dV_o r_s| + |V0_o| (|r_s| + |r_0|))
    VaR     eps_in;  ES eps_in + 2 gamma_{f+2} (||P_o|| + eps_in)
    levels  chart-specific (LOG / SIMPLE / DIFFERENCE)

Scope: DiscountingMode.NONE, FULL_REVAL only. F5 (inverted strike-leg FX) is
FULL_REVAL valuation evidence; it says nothing about Taylor admission. Each
detector applies one semantic defect to a copy of the captured production data
(no production source is mutated) and requires the comparison to fail.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numba
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fwdfix as fx
import _fwdoracle as orc

import riskquant.solutions.risk_run.api as api
from riskquant.config.namespace import GlobalIdNamespace
from riskquant.config.proxy_model.schema_objects import ContainerProxyTerm
from riskquant.constants import INVALID_UIX
from riskquant.jit import parallel_njit, stable_math_parallel_njit
from riskquant.market_data.risk_factors.shift_types import ShiftType
from riskquant.portfolio.instruments import VANILLA_FORWARD_NODE_ID
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

S = fx.HISTORY_DAYS - 1
CONFIDENCES = (0.9, 0.97)  # 0.9: snapped integral m = 8; 0.97: m = 2.4, w > 0
TARGET, DRIVER = "WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]"


# ---- production run with call-through spies -------------------------------------------


def _capture(w: fx.World, **over: Any) -> SimpleNamespace:
    seen: dict[str, Any] = {"out": [], "shocks": []}
    real_apply = api.apply_linear_terms_to_base_levels
    real_convert = api.convert_scenario_pnl_to_reporting_currency

    def apply(**kw: Any) -> None:
        real_apply(**kw)
        seen["out"].append(np.array(kw["out"], copy=True))
        seen["shocks"].append(np.array(kw["shocks"], copy=True))
        seen["offsets"] = np.array(kw["row_term_offsets"], copy=True)
        seen["columns"] = np.array(kw["term_column_uix"], copy=True)
        seen["loadings"] = np.array(kw["term_loadings"], copy=True)

    def convert(**kw: Any):
        seen["convert_in"] = {
            k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for k, v in kw.items()
        }
        seen["convert_out"] = real_convert(**kw)
        return seen["convert_out"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api, "apply_linear_terms_to_base_levels", apply)
        mp.setattr(api, "convert_scenario_pnl_to_reporting_currency", convert)
        run = run_scenario_var_es(**w.inputs(**over))
    levels = np.concatenate(seen["out"], axis=1)
    shocks = np.concatenate(seen["shocks"], axis=0)
    return SimpleNamespace(
        run=run,
        levels=levels,
        shocks=shocks,
        offsets=seen["offsets"],
        columns=seen["columns"],
        loadings=seen["loadings"],
        v0=seen["convert_in"]["base_settlement_values"].astype(np.float64),
        dv=seen["convert_in"]["settlement_pnl"].astype(np.float64),
        dw=np.asarray(seen["convert_out"].converted_pnl, dtype=np.float64),
        portfolio=np.asarray(seen["convert_out"].portfolio_pnl, dtype=np.float64),
    )


def _proxy(beta: float) -> tuple[ContainerProxyTerm, ...]:
    """The canonical CONTAINER term, WTI <- beta * Y(BRENT pillar, LOG_RETURN)."""
    return (
        ContainerProxyTerm(
            target_container_name="WTI_CRUDE_FWD",
            driver_container_name="BRENT_CRUDE_FWD",
            driver_chart=ShiftType.LOG_RETURN,
            loading=beta,
        ),
    )


@pytest.fixture(scope="module")
def world() -> fx.World:
    return fx.World()


@pytest.fixture(scope="module")
def runs(world: fx.World) -> dict[str, SimpleNamespace]:
    return {
        "base": _capture(world),
        "frac": _capture(world, confidence_level=0.97),
        "A": _capture(world, proxy_terms=_proxy(0.95)),
        "B": _capture(world, proxy_terms=_proxy(1.00)),
    }


# ---- independent oracle on captured levels --------------------------------------------


def _by_name(w: fx.World, levels: np.ndarray) -> dict[str, np.ndarray]:
    return {n: np.asarray(levels[w.row(n)], dtype=np.float64) for n in w.names}


def _value(uid: int, lv: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    kind, name, q, k, fx_a, fx_k, _ = fx.ORACLE_ROWS[uid]
    x = lv[fx.FX_NAME]
    if kind == "forward":
        return orc.forward_value(
            q, lv[name], k, orc.effective_rate(x, fx_a), orc.effective_rate(x, fx_k)
        )
    return orc.spot_value(q, lv[name], orc.effective_rate(x, fx_a))


def _rate(uid: int, lv: dict[str, np.ndarray]) -> np.ndarray:
    kind = fx.REPORTING_RATE[fx.ORACLE_ROWS[uid][6]]
    x = lv[fx.FX_NAME]
    r = orc.effective_rate(x, kind)
    return np.ones_like(x) if r is None else r


def _oracle(w: fx.World, levels: np.ndarray) -> SimpleNamespace:
    """Per UID (rows = sorted U): V0_o, T0_o, dV_o, T_s,o, r_0, r_s, dW_o, tolerances."""
    lv0 = {n: np.array([w.base[w.row(n)]]) for n in w.names}
    lvs = _by_name(w, levels)
    v0, t0, dv, ts, r0, rs = [], [], [], [], [], []
    for uid in w.uids.tolist():
        a0, b0 = _value(uid, lv0)
        a_s, b_s = _value(uid, lvs)
        v0.append(a0[0])
        t0.append(b0[0])
        dv.append(a_s - a0[0])
        ts.append(b_s)
        r0.append(_rate(uid, lv0)[0])
        rs.append(_rate(uid, lvs))
    o = SimpleNamespace(
        v0=np.array(v0),
        t0=np.array(t0),
        dv=np.array(dv),
        ts=np.array(ts),
        r0=np.array(r0),
        rs=np.array(rs),
    )
    o.tol_v0 = orc.tol_v0(o.t0)
    o.tol_dv = orc.tol_dv_full(o.dv, o.ts, o.t0[:, None])
    o.dw = orc.reporting_dw(o.dv, o.v0[:, None], o.rs, o.r0[:, None])
    o.tol_dw = orc.tol_dw(
        o.rs, o.r0[:, None], o.tol_dv, o.tol_v0[:, None], o.dv, o.v0[:, None]
    )
    o.portfolio = o.dw.sum(axis=0)
    o.eps_p = orc.portfolio_eps(o.tol_dw, o.dw)
    return o


def _within(prod: np.ndarray, ref: np.ndarray, tol: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(prod, dtype=np.float64) - ref) <= tol


def _row(w: fx.World, uid: int) -> int:
    return int(np.searchsorted(w.uids, uid))


# ---- Phase 1: preconditions, A-L1, hostility, oracle independence ------------------------


@stable_math_parallel_njit
def _numba_log(x: np.ndarray, out: np.ndarray) -> None:
    for i in numba.prange(x.shape[0]):
        out[i] = math.log(x[i])


@parallel_njit
def _numba_exp(x: np.ndarray, out: np.ndarray) -> None:
    for i in numba.prange(x.shape[0]):
        out[i] = math.exp(x[i])


def _log_chart_arguments(w: fx.World) -> tuple[np.ndarray, np.ndarray]:
    """Fixture arguments of log (ratios) and exp (X_o, beta 1.0 and 0.95)."""
    charts = fx.shift_type_by_name()
    ratios, xs = [], []
    for name in w.names:
        if charts[name] != orc.LOG:
            continue
        h = w.hist[name]
        ratios.append(h[1:] / h[:-1])
        for beta in (1.0, 0.95):
            xs.append(orc.level_oracle(h, 1.0, beta, orc.LOG)[1] * beta)
    return np.concatenate(ratios), np.concatenate(xs)


def test_a_l1_libm_contract_on_fixture_arguments(world: fx.World) -> None:
    """A-L1 (fixture-scoped, empirical): Numba and CPython log/exp lie within 4
    binary64 ulps of the precision-40 Decimal reference on these arguments."""
    ratios, xs = _log_chart_arguments(world)
    nb_log, nb_exp = np.empty_like(ratios), np.empty_like(xs)
    _numba_log(ratios, nb_log)
    _numba_exp(xs, nb_exp)
    py_log = np.array([math.log(v) for v in ratios.tolist()])
    py_exp = np.array([math.exp(v) for v in xs.tolist()])
    assert orc.ulps_from_decimal_reference(ratios, nb_log, "ln") <= 4.0
    assert orc.ulps_from_decimal_reference(ratios, py_log, "ln") <= 4.0
    assert orc.ulps_from_decimal_reference(xs, nb_exp, "exp") <= 4.0
    assert orc.ulps_from_decimal_reference(xs, py_exp, "exp") <= 4.0
    one, out = np.array([1.0]), np.empty(1)
    _numba_log(one, out)
    assert out[0] == 0.0 and math.log(1.0) == 0.0


def test_numerical_preconditions(world: fx.World, runs) -> None:
    """P-N, P-O, P-Y, P-L1 to P-L3, P-T (revision 8, §5.0). P-Delta, P-C and P-n
    concern TAYLOR/DERIVED positions; this book declares none."""
    w, base = world, runs["base"]
    o = _oracle(w, base.levels)
    lo, hi = 2.0**-1000, 2.0**1000
    for arr in (
        w.base,
        base.levels,
        o.v0,
        o.t0,
        o.ts,
        o.dv,
        o.dw,
        o.rs,
        np.concatenate([w.hist[n] for n in w.names]),
        np.array([r["quantity"] for r in fx.ROWS]),
        np.array([r.get("strike_price", 1.0) for r in fx.ROWS]),
    ):
        nz = np.abs(arr[arr != 0.0])
        assert np.all((nz >= lo) & (nz <= hi))  # P-N
    assert np.all(np.abs(o.dv) + o.tol_dv <= 2.0**127)  # P-O
    charts = fx.shift_type_by_name()
    for name in w.names:
        y64 = orc.chart_return(w.hist[name][1:], w.hist[name][:-1], charts[name])
        nz = np.abs(y64[y64 != 0.0])
        assert np.all((nz >= 2.0**-126) & (nz <= 2.0**126))  # P-Y
    scale = np.asarray(w.rsc.rf_id_to_unit_scale_factor)
    assert all(scale[int(f)] == 1.0 for f in w.axis.factor_ids)  # P-L1
    assert np.all(np.diff(base.offsets) == 1)  # P-L2
    assert base.levels.shape == (len(w.names), S)  # P-L3: horizon 1, S = T - 1
    for c in CONFIDENCES:
        assert orc.tail_geometry(c, S)[3] > 0.0  # P-T
    m = w.inputs()["methodology"]
    assert m.taylor_derived_uids.size == 0 and m.taylor_supplied_uids.size == 0


def test_fixture_is_hostile_and_node_sourced(world: fx.World) -> None:
    b = world.batch
    assert b.uids.tolist() == [r["uid"] for r in fx.ROWS]
    assert b.uids.tolist() != sorted(b.uids.tolist())
    fwd_rows = np.flatnonzero(b.instrument_ids == VANILLA_FORWARD_NODE_ID)
    assert np.any(np.diff(fwd_rows) > 1)  # forward rows not all adjacent
    assert set(b.uids[fwd_rows].tolist()) == set(fx.FORWARDS)
    nodes = GlobalIdNamespace.NODES
    rf_ids = [world.rf[fx.ORACLE_ROWS[u][1]] for u in fx.FORWARDS]
    assert all(nodes.min_id <= r <= nodes.max_id for r in rf_ids)
    assert len(set(np.diff(np.sort(rf_ids)).tolist())) > 1  # sparse rf ids


def test_oracle_imports_no_production_code() -> None:
    """Static proof: the oracle module imports only numpy, math and decimal."""
    tree = ast.parse(Path(orc.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "math", "decimal", "numpy"}, imported


# ---- Phase 2: semantic bindings --------------------------------------------------------


def test_forward_rows_bind_to_the_named_factors_and_fx_legs(world: fx.World) -> None:
    """row -> effective forward uix -> price uix -> market rf id; FX legs by name."""
    blk = world.context.portfolio.vanilla_forward
    m = world.mapper
    fx_rf = world.rf[fx.FX_NAME]
    for r, uid in enumerate(blk.trade_uids.tolist()):
        _, name, _, _, fx_a, fx_k, _ = fx.ORACLE_ROWS[uid]
        price_uix, _, _, fa = m.efp_effective_quads[
            blk.effective_forward_price_uix[r]
        ].tolist()
        assert int(m.market_id_universe[price_uix]) == world.rf[name]
        assert price_uix != world.rf[name]  # uix is not the rf id
        _, _, fk = m.edf_effective_triples[
            blk.effective_discount_factor_uix[r]
        ].tolist()
        for leg_uix, kind in ((fa, fx_a), (fk, fx_k)):
            if kind == "identity":
                assert leg_uix == INVALID_UIX
                continue
            pair_uix, invert = m.fx_effective_pairs[leg_uix].tolist()
            assert int(m.market_id_universe[pair_uix]) == fx_rf
            assert invert == (1 if kind == "inverted" else 0)


# ---- Phase 2/3: historical levels by name (base run) ----------------------------------


def test_historical_levels_by_factor_name(world: fx.World, runs) -> None:
    """Every axis row, all three charts, against the oracle from raw history."""
    charts = fx.shift_type_by_name()
    assert set(charts.values()) == {orc.LOG, orc.SIMPLE, orc.DIFF}
    for name in world.names:
        _, _, x_hat, tol = orc.level_oracle(
            world.hist[name], world.base[world.row(name)], 1.0, charts[name]
        )
        assert _within(runs["base"].levels[world.row(name)], x_hat, tol).all(), name


# ---- Phase 3: valuation, reporting and tail functionals ---------------------------------


def test_axis_of_the_conversion_is_sorted_u(world: fx.World, runs) -> None:
    assert runs["base"].run.position_uids.tolist() == world.uids.tolist()


def test_v0_matches_the_oracle(world: fx.World, runs) -> None:
    o = _oracle(world, runs["base"].levels)
    assert _within(runs["base"].v0, o.v0, o.tol_v0).all()


def test_full_reval_dv_matches_the_oracle(world: fx.World, runs) -> None:
    o = _oracle(world, runs["base"].levels)
    assert _within(runs["base"].dv, o.dv, o.tol_dv).all()


def test_reporting_dw_and_portfolio_match_the_oracle(world: fx.World, runs) -> None:
    base = runs["base"]
    o = _oracle(world, base.levels)
    assert _within(base.dw, o.dw, o.tol_dw).all()
    assert _within(base.portfolio, o.portfolio, o.eps_p).all()


@pytest.mark.parametrize("key,confidence", [("base", 0.9), ("frac", 0.97)])
def test_var_es_per_uid_and_portfolio(world: fx.World, runs, key, confidence) -> None:
    cap = runs[key]
    o = _oracle(world, cap.levels)
    geometry = orc.tail_geometry(confidence, S)
    f = geometry[1]
    assert (geometry[2] == 0.0) == (key == "base")  # snapped integral / fractional
    for i in range(world.uids.size):
        eps = float(o.tol_dw[i].max())
        var_o, es_o = orc.var_es(o.dw[i], geometry)
        assert abs(cap.run.var[i] - var_o) <= eps
        assert abs(cap.run.es[i] - es_o) <= orc.tol_es(
            eps, float(np.abs(o.dw[i]).max()), f
        )
    eps_p = float(o.eps_p.max())
    var_o, es_o = orc.var_es(o.portfolio, geometry)
    assert abs(cap.run.portfolio_var - var_o) <= eps_p
    assert abs(cap.run.portfolio_es - es_o) <= orc.tol_es(
        eps_p, float(np.abs(o.portfolio).max()), f
    )


def test_batch_size_and_repetition_do_not_move_the_result(
    world: fx.World, runs
) -> None:
    ref = runs["base"]
    for batch_size in (1, 13, S):
        cap = _capture(world, batch_size=batch_size)
        assert np.array_equal(cap.dw, ref.dw) and np.array_equal(cap.dv, ref.dv)
        assert [v.hex() for v in cap.run.var] == [v.hex() for v in ref.run.var]
        assert [v.hex() for v in cap.run.es] == [v.hex() for v in ref.run.es]
    again = _capture(world)
    assert again.run.result_digest == ref.run.result_digest


# ---- Phase 4: F4 Proxy V1 level controls (Run A beta 0.95, Run B beta 1.00) -------------


def _y32(w: fx.World, cap: SimpleNamespace) -> tuple[np.ndarray, float]:
    """The shock column consumed by the target row, and its loading."""
    t = int(cap.offsets[w.row(TARGET)])
    return cap.shocks[:, int(cap.columns[t])].copy(), float(cap.loadings[t])


def _target_oracle(w: fx.World, beta: float, source: str = DRIVER):
    return orc.level_oracle(w.hist[source], w.base[w.row(TARGET)], beta, orc.LOG)


def _s_star(w: fx.World) -> int:
    y64 = orc.chart_return(w.hist[DRIVER][1:], w.hist[DRIVER][:-1], orc.LOG)
    s = int(np.argmax(np.abs(y64)))
    assert abs(y64[s]) >= 1e-3
    return s


def test_f4_captured_driver_shock_is_identical_in_both_runs(
    world: fx.World, runs
) -> None:
    """B-beta0: the control is the captured Y32 column, not the seed."""
    y_a, beta_a = _y32(world, runs["A"])
    y_b, beta_b = _y32(world, runs["B"])
    assert (beta_a, beta_b) == (0.95, 1.00)
    assert y_a.dtype == np.float32 and np.array_equal(
        y_a.view(np.uint32), y_b.view(np.uint32)
    )
    y64, y32_o, _, _ = _target_oracle(world, 1.0)
    # The consumed column is the DRIVER's return (LOG, A-L1): |Y32 - Y32_o| <= (2u + 2^-48)|Y64|.
    assert np.all(
        np.abs(y_a.astype(np.float64) - y32_o) <= (2.0 * orc.U + 2.0**-48) * np.abs(y64)
    )


@pytest.mark.parametrize("key,beta", [("A", 0.95), ("B", 1.00)])
def test_f4_target_levels_follow_beta_times_y32(
    world: fx.World, runs, key, beta
) -> None:
    """B-beta1 / B-beta2: x_A,s = T^-1(x_A,0, fl64(beta * Y32_o)) within tol_level."""
    _, _, x_hat, tol = _target_oracle(world, beta)
    assert _within(runs[key].levels[world.row(TARGET)], x_hat, tol).all()


def test_f4_runs_differ_only_on_the_target(world: fx.World, runs) -> None:
    a, b = runs["A"].levels, runs["B"].levels
    s = _s_star(world)
    assert a[world.row(TARGET), s] != b[world.row(TARGET), s]
    for name in world.names:
        if name != TARGET:
            assert np.array_equal(a[world.row(name)], b[world.row(name)]), name


def test_f4_beta_controls_are_non_degenerate(world: fx.World) -> None:
    """D7 and D6 preconditions at s*: beta omitted, beta squared, driver base
    level and target's own history all move the level beyond 2 tol_level."""
    s = _s_star(world)
    _, y32_o, x_hat, tol = _target_oracle(world, 0.95)
    x0_a, x0_d = world.base[world.row(TARGET)], world.base[world.row(DRIVER)]
    wrong = {
        "beta omitted": orc.reconstruct(x0_a, np.array([y32_o[s]]), orc.LOG)[0],
        "beta squared": orc.reconstruct(x0_a, np.array([0.9025 * y32_o[s]]), orc.LOG)[
            0
        ],
        "driver base": orc.reconstruct(x0_d, np.array([0.95 * y32_o[s]]), orc.LOG)[0],
        "own history": _target_oracle(world, 0.95, source=TARGET)[2][s],
    }
    for label, value in wrong.items():
        assert abs(value - x_hat[s]) > 2.0 * tol[s], label


# ---- detectors: each semantic defect must fail the comparison ---------------------------


def test_d1_omitted_forward_contribution_is_detected(world: fx.World, runs) -> None:
    base = runs["base"]
    o = _oracle(world, base.levels)
    for uid in (fx.F3, fx.F4):
        dv = base.dv.copy()
        dv[_row(world, uid)] = 0.0
        assert not _within(dv, o.dv, o.tol_dv).all(), uid


def test_d2_unshocked_node_is_detected(world: fx.World, runs) -> None:
    """On D2_DAY only TEST_EXACT_DIFF[3M] moves: F2's dV is nonzero, every other
    UID's dV is exactly 0, and an unshocked node fails the comparison."""
    base, s = runs["base"], fx.D2_DAY
    o = _oracle(world, base.levels)
    f2 = _row(world, fx.F2)
    assert base.dv[f2, s] != 0.0
    assert np.all(np.delete(base.dv[:, s], f2) == 0.0)
    frozen = base.levels.copy()
    frozen[world.row("TEST_EXACT_DIFF[3M]"), s] = world.base[
        world.row("TEST_EXACT_DIFF[3M]")
    ]
    assert not _within(
        base.dv[:, s], _oracle(world, frozen).dv[:, s], o.tol_dv[:, s]
    ).all()


def test_d3_swapped_forward_factors_are_detected(world: fx.World, runs) -> None:
    base = runs["base"]
    swapped = base.levels.copy()
    i, j = world.row(DRIVER), world.row(TARGET)
    swapped[[i, j]] = swapped[[j, i]]
    rebased = SimpleNamespace(
        base=world.base.copy(), names=world.names, row=world.row, uids=world.uids
    )
    rebased.base[[i, j]] = rebased.base[[j, i]]
    o = _oracle(rebased, swapped)
    assert not _within(base.dv, o.dv, o.tol_dv).all()


def test_d4_missing_reporting_conversion_is_detected(world: fx.World, runs) -> None:
    base = runs["base"]
    o = _oracle(world, base.levels)
    usd = [_row(world, u) for u in fx.ORACLE_ROWS if fx.ORACLE_ROWS[u][6] == "USD"]
    moved = o.rs[usd] != o.r0[usd, None]  # USDEUR is frozen only on D2_DAY
    assert np.all(np.delete(moved, fx.D2_DAY, axis=1))
    assert not _within(base.dv, o.dw, o.tol_dw).all()


def test_d5_missing_uid_in_aggregation_is_detected(world: fx.World, runs) -> None:
    base = runs["base"]
    o = _oracle(world, base.levels)
    for uid in fx.FORWARDS:
        partial = base.portfolio - base.dw[_row(world, uid)]
        assert not _within(partial, o.portfolio, o.eps_p).all(), uid


def test_d6_target_level_is_not_from_its_own_history(world: fx.World, runs) -> None:
    """B-beta3: Run A's target level is not T^-1(x_A,0, 0.95 * Y32_o(WTI own))."""
    s = _s_star(world)
    _, _, own, _ = _target_oracle(world, 0.95, source=TARGET)
    _, _, _, tol = _target_oracle(world, 0.95)
    assert abs(runs["A"].levels[world.row(TARGET), s] - own[s]) > 2.0 * tol[s]


def test_d7_wrong_beta_application_is_detected(world: fx.World, runs) -> None:
    _, _, x_hat, tol = _target_oracle(world, 0.95)
    for wrong in (1.0, 0.9025):
        _, _, x_wrong, _ = _target_oracle(world, wrong)
        assert not _within(x_wrong, x_hat, tol).all(), wrong


# ---- other admitted scenario models and a sorted world ----------------------------------


@pytest.mark.parametrize("model", ["sbb", "filtered_bootstrap"])
def test_pnl_reporting_and_tail_match_the_oracle_for_other_models(
    world: fx.World, model: str
) -> None:
    """The P&L oracle consumes the captured levels of any scenario model."""
    config = {"model": model, "horizon": 1, "total_scenarios": 40, "seed": 11}
    if model == "sbb":
        config["expected_block_size"] = 5.0
    cap = _capture(world, scenario_config=parse_scenario_model_config(config))
    o = _oracle(world, cap.levels)
    assert _within(cap.v0, o.v0, o.tol_v0).all()
    assert _within(cap.dv, o.dv, o.tol_dv).all()
    assert _within(cap.dw, o.dw, o.tol_dw).all()
    assert _within(cap.portfolio, o.portfolio, o.eps_p).all()
    geometry = orc.tail_geometry(0.9, 40)
    var_o, es_o = orc.var_es(o.portfolio, geometry)
    eps_p = float(o.eps_p.max())
    assert abs(cap.run.portfolio_var - var_o) <= eps_p
    assert abs(cap.run.portfolio_es - es_o) <= orc.tol_es(
        eps_p, float(np.abs(o.portfolio).max()), geometry[1]
    )


def test_hostile_and_sorted_worlds_give_the_same_numbers(world: fx.World, runs) -> None:
    """Same economics with ascending, contiguous UIDs in ascending ingest order:
    per economic slot, P&L, VaR and ES are bitwise equal to the hostile world."""
    new_uid = {r["uid"]: 101 + i for i, r in enumerate(fx.ROWS)}
    rows = tuple(
        sorted(
            ({**r, "uid": new_uid[r["uid"]]} for r in fx.ROWS), key=lambda r: r["uid"]
        )
    )
    sorted_world = fx.World(rows=rows)
    cap, ref = _capture(sorted_world), runs["base"]
    for uid in world.uids.tolist():
        i, j = _row(world, uid), _row(sorted_world, new_uid[uid])
        assert np.array_equal(cap.dw[j], ref.dw[i]), uid
        assert cap.run.var[j].hex() == ref.run.var[i].hex(), uid
        assert cap.run.es[j].hex() == ref.run.es[i].hex(), uid
    assert cap.run.portfolio_var.hex() == ref.run.portfolio_var.hex()
