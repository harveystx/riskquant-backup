# tests/solutions/risk_run/test_parametric_run_world.py
"""
PR-06 at the production entry point: the parametric scenario world belongs to
the run; a batch is only an execution partition of it (D029, D041, D087).

EXACT throughout: the canonical shocked-level buffer, identity, VaR and ES are
compared bitwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reporting_axis import RF, World, capture, spot  # noqa: F401  (fixture)
from test_taylor_order_semantics import FO, Q_BASE, Q_HIST, Q_ROWS, SOF, certified

import riskquant.solutions.risk_run.api as api
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

_S = 20


def _cfg(seed: int = 3, total: int = _S, kind: str = "student_t") -> Any:
    dist = {"kind": kind, "df": 5.0} if kind == "student_t" else {"kind": kind}
    return parse_scenario_model_config({
        "model": "parametric",
        "total_scenarios": total,
        "distribution": dist,
        "seed": seed,
    })


def _data(ids: list[int]) -> ParametricRiskData:
    ids_ = np.array(sorted(ids), dtype=np.int32)
    k = ids_.size
    corr = np.full((k, k), 0.35) + np.eye(k) * 0.65
    return ParametricRiskData(
        factor_ids=ids_,
        cholesky_matrix=np.ascontiguousarray(np.linalg.cholesky(corr)),
        volatilities=np.linspace(0.01, 0.02, k),
    )


def _world(ccy: str = "EUR") -> World:
    return World(
        [spot(1, "EQ_USD", "USD", 3.0), spot(2, "EQ_USD", "USD", -1.0)],
        ccy,
        {"EQ_USD": 100.0, "USDEUR": 0.90},
        {"EQ_USD": [100.0, 101.0], "USDEUR": [0.90, 0.91]},
    )


# ---- 1. the batches must tile [0, S) exactly once, in order (EXACT, fail fast) -------


def _hostile(mode: str):
    real = api.build_scenario_generator

    class Wrapped:
        def __init__(self, g: Any) -> None:
            self.g, self.total_scenarios = g, g.total_scenarios

        def iter_batches(self, **kw: Any):
            batches = [
                type(b)(
                    shocks=np.array(b.shocks, copy=True, order="F"),
                    required_to_shock_uix=b.required_to_shock_uix,
                    shock_scale=b.shock_scale,
                    required_shift_type_id=b.required_shift_type_id,
                    scenario_offset=b.scenario_offset,
                    scenario_count_total=b.scenario_count_total,
                )
                for b in self.g.iter_batches(**kw)
            ]
            if mode == "repeated":
                batches.insert(1, batches[0])
            elif mode == "skipped":
                batches.pop(1)
            elif mode == "reordered":
                batches[0], batches[1] = batches[1], batches[0]
            elif mode == "truncated":
                batches.pop()
            yield from batches

    return lambda config, data=None: Wrapped(real(config, data=data))


@pytest.mark.parametrize("mode", ["repeated", "skipped", "reordered", "truncated"])
def test_a_batch_stream_that_does_not_tile_the_world_is_refused(
    monkeypatch, mode
) -> None:
    """Before PR-06 each of these completed; `skipped` changed VaR under the
    unchanged identity. Refused before any P&L is allocated."""
    w = _world()
    kw = w.inputs(
        scenario_config=_cfg(), data=_data(list(w.axis.factor_ids)), batch_size=7
    )
    reached: list[str] = []
    real_grid = api.allocate_pnl_grid

    def grid(**k: Any):
        reached.append("grid")
        return real_grid(**k)

    monkeypatch.setattr(api, "build_scenario_generator", _hostile(mode))
    monkeypatch.setattr(api, "allocate_pnl_grid", grid)
    with pytest.raises(RuntimeError, match="scenario"):
        run_scenario_var_es(**kw)
    assert reached == []


# ---- 2. one world across every partition (EXACT) -------------------------------------


@pytest.mark.parametrize("kind", ["standard_normal", "student_t"])
def test_every_partition_yields_the_same_world_and_result(capture, kind) -> None:
    w = _world()
    data = _data(list(w.axis.factor_ids))
    ref = run_scenario_var_es(
        **w.inputs(scenario_config=_cfg(kind=kind), data=data, batch_size=_S)
    )
    ref_levels = np.hstack(capture["shocked"]).tobytes()
    for b in (1, 2, 3, 7, _S - 1, _S + 1):
        capture["shocked"].clear()
        r = run_scenario_var_es(
            **w.inputs(scenario_config=_cfg(kind=kind), data=data, batch_size=b)
        )
        assert np.hstack(capture["shocked"]).tobytes() == ref_levels, b
        assert r.identity == ref.identity and r.result_digest == ref.result_digest
        assert (
            r.var.tobytes() == ref.var.tobytes() and r.es.tobytes() == ref.es.tobytes()
        )


# ---- 3. neither the reporting suffix nor the Taylor order touches the world ----------


def test_a_reporting_only_suffix_leaves_the_valuation_rows_bitwise(capture) -> None:
    usd, eur = _world("USD"), _world("EUR")
    assert usd.names == ["EQ_USD"] and eur.names == ["EQ_USD", "USDEUR"]
    data = _data([RF["EQ_USD"], RF["USDEUR"]])  # same scenario data both times
    run_scenario_var_es(**usd.inputs(scenario_config=_cfg(), data=data))
    usd_rows = np.hstack(capture["shocked"])
    capture["shocked"].clear()
    run_scenario_var_es(**eur.inputs(scenario_config=_cfg(), data=data))
    eur_rows = np.hstack(capture["shocked"])
    assert usd_rows[0].tobytes() == eur_rows[eur.row("EQ_USD")].tobytes()


def test_the_taylor_order_never_reaches_the_scenario_world(capture) -> None:
    w = World(Q_ROWS, "USD", Q_BASE, Q_HIST)
    data = _data(list(w.axis.factor_ids))
    runs, worlds = {}, {}
    for order in (FO, SOF):
        capture["shocked"].clear()
        runs[order] = run_scenario_var_es(
            **w.inputs(
                derived=(1, 2),
                block=certified(w, [1, 2], order),
                scenario_config=_cfg(),
                data=data,
            )
        )
        worlds[order] = np.hstack(capture["shocked"]).tobytes()
    assert worlds[FO] == worlds[SOF]
    assert runs[FO].identity != runs[SOF].identity  # D086: different stored terms


# ---- 4. seed, identity and reproduction (EXACT) --------------------------------------


def test_seed_defines_the_world_and_the_identity_batch_size_does_not(capture) -> None:
    w = _world()
    data = _data(list(w.axis.factor_ids))
    a = run_scenario_var_es(
        **w.inputs(scenario_config=_cfg(seed=3), data=data, batch_size=4)
    )
    world_a = np.hstack(capture["shocked"]).tobytes()
    capture["shocked"].clear()
    b = run_scenario_var_es(
        **w.inputs(scenario_config=_cfg(seed=4), data=data, batch_size=4)
    )
    assert np.hstack(capture["shocked"]).tobytes() != world_a
    assert a.identity != b.identity
    assert (
        a.manifest.scenario_config["seed"] == 3
        and b.manifest.scenario_config["seed"] == 4
    )
    again = run_scenario_var_es(
        **w.inputs(scenario_config=_cfg(seed=3), data=data, batch_size=9),
        reproduce=a.manifest,
    )
    assert again.identity == a.identity and again.var.tobytes() == a.var.tobytes()
