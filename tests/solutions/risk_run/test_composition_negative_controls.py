# tests/solutions/risk_run/test_composition_negative_controls.py
"""
Negative controls for the composition evidence.

Each test injects ONE deliberate corruption into the production path
(monkeypatch, undone after the test) and proves that the independent check used
in `test_methodology_composition.py` detects it, or that the fail-closed gate is
the ONLY barrier (without it, a plausible wrong number completes silently). A
detector that cannot see its defect is theatre; these make the claim executable.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf
from test_methodology_composition import ATOL, _oracle, _rows, spy  # noqa: F401

import riskquant.solutions.risk_run.api as api
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator

DETECT = 10 * ATOL


def _worst(run, seen, *, taylor_uids=(), data=None, uids=mf.ALL_UIDS):
    rows = _rows(run, seen)
    return max(
        float(
            np.abs(rows[u][0] - _oracle(u, taylor=u in taylor_uids, data=data)[2]).max()
        )
        for u in uids
    )


def test_nc_full_calling_taylor_trips_the_independence_trap(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("FULL_REVAL entered a Taylor path")

    monkeypatch.setattr(api.FactorAxis, "from_keys", boom)
    monkeypatch.setattr(SensitivityPnlCalculator, "run_into_grid", boom)
    real = api._value_positions

    def corrupt(**kw):
        api.FactorAxis.from_keys(None)  # FULL now builds a Taylor market
        return real(**kw)

    monkeypatch.setattr(api, "_value_positions", corrupt)
    with pytest.raises(AssertionError, match="Taylor path"):
        run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=()))


def test_nc_full_values_in_derived_rows_are_caught(monkeypatch, spy) -> None:
    current = SensitivityPnlCalculator.run_into_grid
    w = mf.world()

    def corrupt(self, market, **kw):
        prov = current(self, market, **kw)
        uids = kw["grid"].block_uids(kw["block_index"])
        plan = api._row_aggregation(
            portfolio=w.portfolio, universe=np.array(mf.ALL_UIDS, np.int64)
        )
        route = api._ValuationRoute(
            portfolio=w.portfolio, uix_mapper=w.mapper, universe_map=w.universe_map
        )
        pos = api._positions(plan.economic_uids, uids)
        value = lambda lv: api._value_positions(
            route=route, aggregation=plan, flat_levels=np.ascontiguousarray(lv)
        )[pos]
        v0 = value(market.base_levels[:, None])[:, 0]
        kw["grid"].block_view(kw["block_index"])[:] = (
            value(market.shocked_levels) - v0[:, None]
        )
        return prov

    monkeypatch.setattr(SensitivityPnlCalculator, "run_into_grid", corrupt)
    run = run_scenario_var_es(**mf.inputs())
    assert _worst(run, spy, taylor_uids=mf.DERIVED_UIDS) > DETECT


def test_nc_aggregation_that_keeps_one_row_per_uid_is_caught() -> None:
    """At the valuation helpers (N1: a multi-row SPOT UID cannot reach the run)."""
    from test_methodology_composition import _decomposed

    w = mf.world()
    split = _decomposed(w.portfolio, 2_003, [25.0, 30.0, 20.0])
    plan = api._row_aggregation(
        portfolio=split, universe=np.array(mf.ALL_UIDS, np.int64)
    )
    first = plan.row_order[plan.segment_starts]
    corrupt = dataclasses.replace(
        plan, row_order=first, segment_starts=np.arange(first.shape[0], dtype=np.intp)
    )
    route = api._ValuationRoute(
        portfolio=split, uix_mapper=w.mapper, universe_map=w.universe_map
    )
    base = np.ascontiguousarray(w.su.base_levels[:, None])
    row = list(plan.economic_uids).index(2_003)
    wti0 = float(w.su.base_levels[mf._row(w, "WTI_CRUDE")])
    good = api._value_positions(route=route, aggregation=plan, flat_levels=base)[row, 0]
    bad = api._value_positions(route=route, aggregation=corrupt, flat_levels=base)[
        row, 0
    ]
    assert good == pytest.approx(75.0 * wti0, abs=1e-9) and abs(bad - 75.0 * wti0) > 1.0


@pytest.mark.parametrize(
    "defect", ["q0_for_qs", "translation_omitted", "fx_applied_twice"]
)
def test_nc_reporting_defects_are_caught_by_the_l0_rows(
    monkeypatch, spy, defect
) -> None:
    current = api.convert_scenario_pnl_to_reporting_currency

    def corrupt(**kw):
        if defect == "q0_for_qs":
            kw["fx_levels"] = np.ascontiguousarray(
                np.repeat(
                    kw["base_fx_levels"][:, None], kw["fx_levels"].shape[1], axis=1
                )
            )
        elif defect == "translation_omitted":
            kw["base_settlement_values"] = np.zeros_like(kw["base_settlement_values"])
        else:
            first = current(**kw)
            kw["settlement_pnl"] = np.ascontiguousarray(first.converted_pnl)
        return current(**kw)

    monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", corrupt)
    run = run_scenario_var_es(**mf.inputs())
    assert _worst(run, spy, taylor_uids=mf.DERIVED_UIDS) > DETECT


def test_nc_factor_axis_gate_is_load_bearing(monkeypatch, spy) -> None:
    """Without the ordered-axis gate a permuted axis completes with finite,
    plausible, WRONG numbers: the gate is the only barrier. The reporting FX
    rows are run-derived (D085), so the caller's permuted arrays are the only
    contradiction."""
    from test_methodology_composition import _axis_case

    monkeypatch.setattr(api, "_require_factor_axis", lambda **kw: None)
    run = run_scenario_var_es(
        **_axis_case("permuted", mf.inputs(full=mf.ALL_UIDS, derived=()))
    )
    assert np.isfinite(run.var).all()
    assert _worst(run, spy) > DETECT


@pytest.mark.parametrize(
    "defect", ["v0_from_other_state", "scenarios_from_stale_state"]
)
def test_nc_valuation_route_defects_are_caught(monkeypatch, spy, defect) -> None:
    real = api._value_positions
    shocked = mf.shocked_levels()

    def corrupt(**kw):
        lv = kw["flat_levels"]
        if defect == "v0_from_other_state" and lv.shape[1] == 1:
            kw["flat_levels"] = np.ascontiguousarray(shocked[:, :1])
        if defect == "scenarios_from_stale_state" and lv.shape[1] > 1:
            kw["flat_levels"] = np.ascontiguousarray(
                np.repeat(mf.world().su.base_levels[:, None], lv.shape[1], axis=1)
            )
        return real(**kw)

    monkeypatch.setattr(api, "_value_positions", corrupt)
    run = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=(), batch_size=4))
    rows = _rows(run, spy)
    v0_err = max(abs(rows[u][2] - _oracle(u, taylor=False)[0]) for u in mf.ALL_UIDS)
    dv_err = max(
        float(np.abs(rows[u][1] - _oracle(u, taylor=False)[1]).max())
        for u in mf.ALL_UIDS
    )
    assert max(v0_err, dv_err) > DETECT


def test_nc_without_gate_a_an_undeclared_valued_uid_vanishes_silently(
    monkeypatch,
) -> None:
    real = api._row_aggregation
    monkeypatch.setattr(
        api,
        "_row_aggregation",
        lambda *, portfolio, universe: real(
            portfolio=portfolio, universe=np.unique(portfolio.trade_uids)
        ),
    )
    universe = tuple(u for u in mf.ALL_UIDS if u != 2_006)
    run = run_scenario_var_es(**mf.inputs(full=universe, derived=(), universe=universe))
    assert 2_006 not in run.position_uids.tolist() and np.isfinite(run.var).all()


def test_nc_without_gate_b_a_missing_uid_borrows_another_uids_value(
    monkeypatch, spy
) -> None:
    real = api._row_aggregation
    monkeypatch.setattr(
        api,
        "_row_aggregation",
        lambda *, portfolio, universe: real(
            portfolio=portfolio, universe=np.unique(portfolio.trade_uids)
        ),
    )
    monkeypatch.setattr(
        api,
        "_positions",
        lambda econ, uids: np.minimum(
            np.searchsorted(econ, uids), econ.shape[0] - 1
        ).astype(np.intp),
    )
    universe = mf.ALL_UIDS + (9_999,)
    run = run_scenario_var_es(**mf.inputs(full=universe, derived=(), universe=universe))
    rows = _rows(run, spy)
    assert np.array_equal(rows[9_999][1], rows[2_006][1])  # plausible, wrong, silent
