# tests/solutions/risk_run/test_proxy_v1_production.py
"""
D088 Stage D: the production run executes Proxy V1.

Every case drives `run_scenario_var_es` end to end on the hostile remediation
static configuration (via the reporting-axis `World`). Oracles are closed forms
evaluated in float64 on the float32 world values, which are themselves the
chart definitions of the historical level paths (D088 §11, §13.6: rtol 1e-12,
atol 1e-15); determinism checks are bitwise.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_reporting_axis as t  # noqa: E402
from test_parametric_run_world import _data as _param_data  # noqa: E402
from test_parametric_run_world import _manifest_spy  # noqa: E402
from test_reporting_axis import (  # noqa: E402
    AS_OF,
    RF,
    World,
    capture,  # noqa: F401  (pytest fixture)
    spot,
)

import riskquant.solutions.risk_run.api as api  # noqa: E402
from riskquant.config.proxy_model.compile import compile_proxy_model  # noqa: E402
from riskquant.config.static_config.compile import (  # noqa: E402
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.proxy_model.schema_objects import NodeProxyTerm  # noqa: E402
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix  # noqa: E402
from riskquant.market_data.risk_factors.shift_types import ShiftType  # noqa: E402
from riskquant.solutions.risk_run.api import run_scenario_var_es  # noqa: E402
from riskquant.solutions.run_identity.hashing import hash_proxy_model  # noqa: E402
from riskquant.solutions.scenario_models.api import (  # noqa: E402
    parse_scenario_model_config,
)
from riskquant.utils.business_days import ord_to_date  # noqa: E402

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1
RTOL, ATOL = 1e-12, 1e-15

# One USD position reported in EUR: USDEUR is a reporting-only suffix node.
ROWS = [spot(1, "EQ_USD", "USD", 1.0)]
BASE = {"EQ_USD": 100.0, "USDEUR": 0.90}
HIST = {"EQ_USD": [100.0, 110.0], "USDEUR": [0.90, 0.80]}


# ---- helpers ---------------------------------------------------------------------------


_SHIFT_NAME = {
    LOG: "LOG_RETURN",
    SIMPLE: "SIMPLE_RETURN",
    DIFF: "DIFFERENCE",
    ABS: "ABSOLUTE_BASE_RETURN",
    NO_SHIFT: "NO_SHIFT",
}


def _rsc(**shift: int):
    """The fixture static configuration recompiled with some nodes' configured
    shift types (their target charts kappa) changed at the source."""
    dfs = t.hf.static_config_dfs()
    node = dfs["node_meta"]
    for name, sid in shift.items():
        node = node.with_columns(
            pl.when(pl.col("node_name") == name)
            .then(pl.lit(_SHIFT_NAME[sid]))
            .otherwise(pl.col("shift_type"))
            .alias("shift_type")
        )
    dfs["node_meta"] = node
    rsc = compile_risk_static_config_from_dict_of_dfs(dfs)
    for name, sid in shift.items():
        assert int(rsc.rf_id_to_shift_type_id[RF[name]]) == sid
    return rsc


def _world(rows=ROWS, ccy="EUR", base=BASE, hist=HIST, rsc=None) -> World:
    if rsc is None:
        return World(rows, ccy, base, hist)
    with mock.patch.object(t, "RSC", rsc):
        return World(rows, ccy, base, hist)


def _term(target: str, driver: str, chart: int, beta: float) -> NodeProxyTerm:
    return NodeProxyTerm(
        target_name=target, driver_name=driver, driver_chart=ShiftType(chart), loading=beta
    )


def _compile(w: World, terms=(), axis=None):
    return compile_proxy_model(
        authored_terms=terms,
        static_config=w.context.static_config,
        as_of_date=ord_to_date(AS_OF),
        run_axis_rf_ids=w.axis.factor_ids if axis is None else axis,
    )


def _row_terms(model, rf_id: int) -> list[tuple[int, int, float]]:
    r = int(np.flatnonzero(model.row_target_rf_ids == rf_id)[0])
    lo, hi = model.row_term_offsets[r], model.row_term_offsets[r + 1]
    return list(zip(
        model.term_driver_rf_ids[lo:hi].tolist(),
        model.term_chart_ids[lo:hi].tolist(),
        model.term_loadings[lo:hi].tolist(),
        strict=True,
    ))


def _y(name: str, chart: int, hist=HIST) -> float:
    """The float32 world value of a one-step level-backed column (chart definition)."""
    v0, v1 = hist[name][0], hist[name][1]
    x = {LOG: math.log(v1 / v0), SIMPLE: v1 / v0 - 1.0, DIFF: v1 - v0,
         ABS: (v1 - v0) / abs(v0)}[chart]
    return float(np.float32(x))


def _shocked(capture) -> np.ndarray:
    return np.hstack(capture["shocked"])


# ---- A / B: identity rows for valuation and reporting-suffix nodes ------------------------


def test_the_run_compiles_the_full_axis_and_identifies_that_operator() -> None:
    w = _world()
    assert [t.NAME[int(f)] for f in w.axis.factor_ids] == ["EQ_USD", "USDEUR"]
    run = run_scenario_var_es(**w.inputs())
    full = _compile(w)
    prefix_only = _compile(w, axis=w.axis.factor_ids[: w.axis.num_valuation_factors])
    assert run.manifest.shocks.proxy_model_digest == hash_proxy_model(full)
    assert run.manifest.shocks.proxy_model_digest != hash_proxy_model(prefix_only)
    assert _row_terms(full, RF["USDEUR"]) == [(RF["USDEUR"], LOG, 1.0)]  # suffix identity row
    assert _row_terms(full, RF["EQ_USD"]) == [(RF["EQ_USD"], LOG, 1.0)]


def test_identity_rows_reconstruct_valuation_and_suffix_nodes(capture) -> None:
    """A: EQ_USD_s = 100 e^Y. B: q_s = 0.90 e^Y and the conversion reads q_s."""
    w = _world()
    run_scenario_var_es(**w.inputs())
    s = _shocked(capture)
    np.testing.assert_allclose(
        s[w.row("EQ_USD"), 0], 100.0 * math.exp(_y("EQ_USD", LOG)), rtol=RTOL, atol=ATOL
    )
    q_s = 0.90 * math.exp(_y("USDEUR", LOG))
    np.testing.assert_allclose(s[w.row("USDEUR"), 0], q_s, rtol=RTOL, atol=ATOL)
    fx = capture["convert_in"]["fx_levels"]
    assert fx.shape[0] == 1
    assert fx[0].tobytes() == s[w.row("USDEUR")].tobytes()


# ---- C: one FX node, prefix or suffix, one law ------------------------------------------------


def test_same_fx_node_has_one_law_in_prefix_and_suffix(capture) -> None:
    suffix = _world()
    run_scenario_var_es(**suffix.inputs())
    q_suffix = _shocked(capture)[suffix.row("USDEUR")].copy()
    capture["shocked"].clear()
    prefix = _world(rows=[*ROWS, spot(2, "EQ_USD", "EUR", 1.0)])
    assert prefix.axis.reporting_factor_ids.size == 0  # USDEUR now in the prefix
    run_scenario_var_es(**prefix.inputs())
    q_prefix = _shocked(capture)[prefix.row("USDEUR")]
    assert q_prefix.tobytes() == q_suffix.tobytes()
    assert _row_terms(_compile(prefix), RF["USDEUR"]) == _row_terms(
        _compile(suffix), RF["USDEUR"]
    )


# ---- D: authored suffix proxy without own history --------------------------------------------


def test_authored_suffix_proxy_without_history(capture) -> None:
    w = _world()
    hist = {"EQ_USD": HIST["EQ_USD"], "USDCHF": [0.88, 0.90]}
    ids = sorted([RF["EQ_USD"], RF["USDCHF"]])
    names = {RF["EQ_USD"]: "EQ_USD", RF["USDCHF"]: "USDCHF"}
    data = RiskFactorMatrix(
        dates=np.arange(AS_OF - 1, AS_OF + 1, dtype=np.int32),
        factor_ids=np.array(ids, dtype=np.int32),
        values=np.asfortranarray(np.column_stack([hist[names[i]] for i in ids])),
    )
    kw = w.inputs(data=data)
    with pytest.raises(ValueError, match="not in the matrix world"):
        run_scenario_var_es(**kw)  # identity row needs (USDEUR, LOG, h)
    kw["proxy_terms"] = (_term("USDEUR", "USDCHF", LOG, 1.0),)
    run_scenario_var_es(**kw)
    q_s = 0.90 * math.exp(_y("USDCHF", LOG, hist))
    np.testing.assert_allclose(_shocked(capture)[w.row("USDEUR"), 0], q_s, rtol=RTOL, atol=ATOL)


# ---- E / F: charts ------------------------------------------------------------------------------


def test_missing_driver_chart_fails_before_generation_and_manifest(monkeypatch) -> None:
    w = _world(hist={"EQ_USD": [100.0, 101.0], "USDEUR": [0.90, 0.91]})
    built = _manifest_spy(monkeypatch)
    gens: list[int] = []
    monkeypatch.setattr(api, "build_scenario_generator", lambda *a, **k: gens.append(1))
    cfg = parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": 20,
        "seed": 3,
        "distribution": {"kind": "standard_normal"},
    })
    kw = w.inputs(scenario_config=cfg, data=_param_data(list(w.axis.factor_ids)))
    kw["proxy_terms"] = (_term("EQ_USD", "EQ_USD", DIFF, 1.0),)
    with pytest.raises(ValueError, match="does not supply"):
        run_scenario_var_es(**kw)
    assert built == [] and gens == []


def _request_spy(monkeypatch) -> list[list[tuple[int, int]]]:
    """Record the (driver, driver chart) world request the run streams."""
    seen: list[list[tuple[int, int]]] = []
    real = api.build_scenario_generator

    class Recording:
        def __init__(self, g: Any) -> None:
            self._g = g
            self.total_scenarios = g.total_scenarios

        def iter_batches(self, **kw: Any):
            seen.append(list(zip(kw["driver_rf_ids"].tolist(), kw["chart_ids"].tolist(), strict=True)))
            return self._g.iter_batches(**kw)

    monkeypatch.setattr(api, "build_scenario_generator", lambda *a, **k: Recording(real(*a, **k)))
    return seen


def test_log_target_driven_by_difference_uses_the_difference_column(capture, monkeypatch) -> None:
    """kappa(EQ_USD) = LOG, tau = DIFF: the authored term requests (USDEUR, DIFF),
    the world supplies that column, and reconstruction is LOG over it. EQ_USD's
    own (EQ_USD, LOG) column is never requested (its row is authored), and
    USDEUR's identity row requests (USDEUR, LOG) as a separate column."""
    requests = _request_spy(monkeypatch)
    w = _world()
    kw = w.inputs()
    kw["proxy_terms"] = (_term("EQ_USD", "USDEUR", DIFF, 0.5),)
    run_scenario_var_es(**kw)
    assert requests == [[(RF["USDEUR"], DIFF), (RF["USDEUR"], LOG)]]
    got = _shocked(capture)[w.row("EQ_USD"), 0]
    np.testing.assert_allclose(got, 100.0 * math.exp(0.5 * _y("USDEUR", DIFF)), rtol=RTOL, atol=ATOL)
    assert abs(got - 100.0 * math.exp(0.5 * _y("USDEUR", LOG))) > 0.1


# ---- G / H: native suffix rows ------------------------------------------------------------------


def test_no_shift_suffix_needs_no_column_and_stays_outside_the_digest(capture) -> None:
    w = _world(rsc=_rsc(USDEUR=NO_SHIFT))
    run = run_scenario_var_es(**w.inputs(data=w.data(drop=("USDEUR",))))
    assert _shocked(capture)[w.row("USDEUR")].tolist() == [0.90]
    prefix_only = _compile(w, axis=w.axis.factor_ids[: w.axis.num_valuation_factors])
    assert run.manifest.shocks.proxy_model_digest == hash_proxy_model(prefix_only)


def test_abs_suffix_uses_its_native_column_outside_the_digest(capture) -> None:
    w = _world(rsc=_rsc(USDEUR=ABS))
    run = run_scenario_var_es(**w.inputs())
    q_s = 0.90 + _y("USDEUR", ABS) * 0.90
    np.testing.assert_allclose(_shocked(capture)[w.row("USDEUR"), 0], q_s, rtol=RTOL, atol=ATOL)
    prefix_only = _compile(w, axis=w.axis.factor_ids[: w.axis.num_valuation_factors])
    assert run.manifest.shocks.proxy_model_digest == hash_proxy_model(prefix_only)


# ---- §14 identity ------------------------------------------------------------------------------


def _ids(run) -> tuple[str, str]:
    return run.manifest.shocks.proxy_model_digest, run.identity


def test_suffix_added_changes_proxy_and_run_identity() -> None:
    eur, usd = _world(ccy="EUR"), _world(ccy="USD")
    assert usd.axis.reporting_factor_ids.size == 0
    a, b = _ids(run_scenario_var_es(**eur.inputs())), _ids(run_scenario_var_es(**usd.inputs()))
    assert a[0] != b[0] and a[1] != b[1]


def test_suffix_authored_coefficient_changes_proxy_and_run_identity() -> None:
    w = _world()
    base = _ids(run_scenario_var_es(**w.inputs()))
    kw = w.inputs()
    kw["proxy_terms"] = (_term("USDEUR", "USDEUR", LOG, 0.5),)
    other = _ids(run_scenario_var_es(**kw))
    assert base[0] != other[0] and base[1] != other[1]


def test_suffix_kappa_changes_proxy_and_run_identity() -> None:
    a = _ids(run_scenario_var_es(**_world().inputs()))
    b = _ids(run_scenario_var_es(**_world(rsc=_rsc(USDEUR=SIMPLE)).inputs()))
    assert a[0] != b[0] and a[1] != b[1]


def test_reporting_orientation_changes_run_identity_not_proxy_identity() -> None:
    """EUR and GBP positions on one axis [EQ_EUR, EQ_GBP | EURGBP]: reported in
    GBP the EURGBP row is read directly, in EUR reciprocally."""
    rows = [spot(1, "EQ_EUR", "EUR", 1.0), spot(2, "EQ_GBP", "GBP", 1.0)]
    base = {"EQ_EUR": 50.0, "EQ_GBP": 80.0, "EURGBP": 0.85}
    hist = {"EQ_EUR": [50.0, 51.0], "EQ_GBP": [80.0, 79.0], "EURGBP": [0.85, 0.86]}
    in_gbp, in_eur = _world(rows, "GBP", base, hist), _world(rows, "EUR", base, hist)
    assert np.array_equal(in_gbp.axis.factor_ids, in_eur.axis.factor_ids)
    ra, rb = run_scenario_var_es(**in_gbp.inputs()), run_scenario_var_es(**in_eur.inputs())
    assert ra.manifest.shocks.proxy_model_digest == rb.manifest.shocks.proxy_model_digest
    assert ra.manifest.reporting != rb.manifest.reporting
    assert ra.identity != rb.identity


def test_suffix_base_level_changes_run_identity_not_proxy_identity() -> None:
    a = run_scenario_var_es(**_world().inputs())
    b = run_scenario_var_es(**_world(base={**BASE, "USDEUR": 0.9009}).inputs())
    assert a.manifest.shocks.proxy_model_digest == b.manifest.shocks.proxy_model_digest
    assert a.manifest.pnl.base_levels_digest != b.manifest.pnl.base_levels_digest
    assert a.identity != b.identity


def test_world_horizon_changes_run_identity_not_proxy_identity() -> None:
    hist = {"EQ_USD": [100.0, 105.0, 110.0, 108.0], "USDEUR": [0.90, 0.85, 0.80, 0.82]}
    w = _world(hist=hist)
    runs = [
        run_scenario_var_es(
            **w.inputs(scenario_config=parse_scenario_model_config({"model": "historical", "horizon": h}))
        )
        for h in (1, 2)
    ]
    assert runs[0].manifest.shocks.proxy_model_digest == runs[1].manifest.shocks.proxy_model_digest
    assert runs[0].identity != runs[1].identity
    # The proxy digest is the operator's identity only, never the run's.
    assert all(r.identity != r.manifest.shocks.proxy_model_digest for r in runs)


# ---- §15 reproduction ------------------------------------------------------------------------------


def test_reproduce_with_another_batch_size_is_bitwise_identical() -> None:
    hist = {n: list(np.array(t.B_HIST[n], dtype=np.float64)) for n in t.B_HIST}
    w = _world(t.B_ROWS, "EUR", t.B_BASE, hist)
    kw = w.inputs()
    kw["proxy_terms"] = (_term("EQ_GBP", "EQ_USD", LOG, 0.7),)
    first = run_scenario_var_es(**kw)
    again = run_scenario_var_es(**{**kw, "batch_size": 2}, reproduce=first.manifest)
    assert again.identity == first.identity
    assert first.var.tobytes() == again.var.tobytes() and first.es.tobytes() == again.es.tobytes()


# ---- §16 closed-form oracles through the production seam ------------------------------------------


@pytest.mark.parametrize(
    ("kappa", "terms", "expected"),
    [
        (LOG, [("USDEUR", DIFF, 0.5)], lambda y: 100.0 * math.exp(0.5 * y[("USDEUR", DIFF)])),
        (LOG, [("USDEUR", LOG, 0.95)], lambda y: 100.0 * math.exp(0.95 * y[("USDEUR", LOG)])),
        (SIMPLE, [("EQ_USD", SIMPLE, 0.5)], lambda y: 100.0 * (1.0 + 0.5 * y[("EQ_USD", SIMPLE)])),
        (DIFF, [("USDEUR", LOG, 2.0)], lambda y: 100.0 + 2.0 * y[("USDEUR", LOG)]),
        (DIFF, [("USDEUR", DIFF, 3.0)], lambda y: 100.0 + 3.0 * y[("USDEUR", DIFF)]),
        (
            LOG,
            [("EQ_USD", LOG, 0.9), ("USDEUR", DIFF, 0.02)],
            lambda y: 100.0 * math.exp(0.9 * y[("EQ_USD", LOG)] + 0.02 * y[("USDEUR", DIFF)]),
        ),
    ],
    ids=["LOG<-DIFF", "LOG<-LOG", "SIMPLE", "DIFF<-LOG", "DIFF<-DIFF", "mixed"],
)
def test_closed_form_oracles_through_the_production_run(capture, kappa, terms, expected) -> None:
    w = _world(rsc=_rsc(EQ_USD=kappa))
    kw = w.inputs()
    kw["proxy_terms"] = tuple(_term("EQ_USD", d, c, b) for d, c, b in terms)
    run_scenario_var_es(**kw)
    y = {(n, c): _y(n, c) for n in ("EQ_USD", "USDEUR") for c in (LOG, SIMPLE, DIFF)}
    np.testing.assert_allclose(
        _shocked(capture)[w.row("EQ_USD"), 0], expected(y), rtol=RTOL, atol=ATOL
    )


# ---- the D013 path is off the production execution path -------------------------------------------


def test_the_run_has_no_d013_inputs_or_calls() -> None:
    """Stage E: the D013 architecture does not exist. The run has no D013
    inputs, and none of the D013 modules can be imported."""
    import importlib

    params = inspect.signature(run_scenario_var_es).parameters
    assert "proxy_terms" in params
    assert not {"source_factor_ids", "shock_scale"} & set(params)
    for name in ("apply_shocks_to_base_levels", "_require_shock_sources"):
        assert not hasattr(api, name)
    for module in (
        "riskquant.proxy_resolver",
        "riskquant.proxy_resolver.proxy",
        "riskquant.config.proxy_config",
        "riskquant.config.proxy_config.compile",
        "riskquant.simlab.shocks.apply",
        "riskquant.simlab.shocks.shock_kernel",
        "riskquant.simlab.builder.shock_universe",
        "riskquant.simlab.scenarios.world_column_stream",
        "riskquant.ingest_fixtures.config.risk_proxy_config",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    run = run_scenario_var_es(**_world().inputs())
    assert np.isfinite(run.var).all()



# ---- D089 / D090 horizon equality at the production run --------------------------------------------


def _generator_spy(monkeypatch) -> list[int]:
    built: list[int] = []
    real = api.build_scenario_generator

    def spy(*a: Any, **k: Any):
        built.append(1)
        return real(*a, **k)

    monkeypatch.setattr(api, "build_scenario_generator", spy)
    return built


def _period_data(w: World, count: int):
    base = _param_data(list(w.axis.factor_ids))
    return dataclasses.replace(base, effective_period_count=count)


def _period_cfg(count: int):
    return parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": count, "unit": "BUSINESS_DAY"},
        "total_scenarios": 16,
        "seed": 3,
        "distribution": {"kind": "standard_normal"},
    })


@pytest.mark.parametrize(("requested", "effective"), [(1, 5), (5, 1), (2, 3)])
def test_a_horizon_mismatch_fails_before_manifest_and_generation(
    monkeypatch, requested, effective
) -> None:
    """Requested (config) and effective (data) periods differ: the production
    run refuses before any manifest or generator exists. No tolerance, no
    scaling, no conversion (D089, D090)."""
    manifests, generators = _manifest_spy(monkeypatch), _generator_spy(monkeypatch)
    w = _world(hist={"EQ_USD": [100.0, 101.0], "USDEUR": [0.90, 0.91]})
    kw = w.inputs(scenario_config=_period_cfg(requested), data=_period_data(w, effective))
    with pytest.raises(ValueError, match="D090"):
        run_scenario_var_es(**kw)
    assert manifests == [] and generators == []


def test_an_equal_horizon_runs_once_through_the_world() -> None:
    w = _world(hist={"EQ_USD": [100.0, 101.0], "USDEUR": [0.90, 0.91]})
    runs = [
        run_scenario_var_es(**w.inputs(scenario_config=_period_cfg(h), data=_period_data(w, h)))
        for h in (1, 5)
    ]
    assert all(np.isfinite(r.var).all() for r in runs)
    # Same proxy operator at both horizons; the world (and so the run) differs.
    assert runs[0].manifest.shocks.proxy_model_digest == runs[1].manifest.shocks.proxy_model_digest
    assert runs[0].identity != runs[1].identity


# ---- one run: V1 prefix, V1 suffix, ABS suffix, NO_SHIFT suffix --------------------------------------


def _rsc_with_usdjpy(**shift: int):
    """The fixture configuration plus one FX pair USDJPY (JPY per USD), so a
    reporting-USD book can carry three reporting-only suffix nodes."""
    dfs = t.hf.static_config_dfs()
    node = dfs["node_meta"]
    row = node.filter(pl.col("node_name") == "USDCHF").with_columns(
        pl.lit(5009, dtype=node.schema["node_id"]).alias("node_id"),
        pl.lit("USDJPY").alias("node_name"),
        pl.lit("JPY").alias("quote_ccy"),
    )
    assert 5009 not in node["node_id"].to_list()
    node = pl.concat([node, row])
    for name, sid in shift.items():
        node = node.with_columns(
            pl.when(pl.col("node_name") == name)
            .then(pl.lit(_SHIFT_NAME[sid]))
            .otherwise(pl.col("shift_type"))
            .alias("shift_type")
        )
    dfs["node_meta"] = node
    return compile_risk_static_config_from_dict_of_dfs(dfs)


def test_one_run_mixes_v1_prefix_v1_suffix_and_native_suffix_rows(capture, monkeypatch) -> None:
    rsc = _rsc_with_usdjpy(USDCHF=ABS, USDJPY=NO_SHIFT)
    jpy = int(rsc.rel_name_to_rf_id["USDJPY"])
    rows = [
        spot(1, "EQ_USD", "USD", 1.0),
        spot(2, "EQ_EUR", "EUR", 1.0),
        spot(3, "EQ_CHF", "CHF", 1.0),
        spot(4, "EQ_JPY", "JPY", 1.0),
    ]
    base = {"EQ_USD": 100.0, "EQ_EUR": 50.0, "EQ_CHF": 40.0, "EQ_JPY": 2000.0,
            "USDEUR": 0.90, "USDCHF": 0.88, "USDJPY": 150.0}
    hist = {"EQ_USD": [100.0, 102.0], "EQ_EUR": [50.0, 49.0], "EQ_CHF": [40.0, 41.0],
            "EQ_JPY": [2000.0, 2010.0], "USDEUR": [0.90, 0.92], "USDCHF": [0.88, 0.87],
            "USDJPY": [150.0, 151.0]}
    requests = _request_spy(monkeypatch)
    with mock.patch.dict(t.NAME, {jpy: "USDJPY"}), mock.patch.object(t, "RSC", rsc):
        w = World(rows, "USD", base, hist)
        suffix = [t.NAME[int(f)] for f in w.axis.factor_ids[w.axis.num_valuation_factors:]]
        assert sorted(suffix) == ["USDCHF", "USDEUR", "USDJPY"]
        names = [t.NAME[int(f)] for f in w.axis.factor_ids]
        assert len(names) == len(set(names))  # every FX factor once on the axis
        run = run_scenario_var_es(**w.inputs(data=w.data(drop=("USDJPY",))))
        full = compile_proxy_model(
            authored_terms=(), static_config=rsc, as_of_date=ord_to_date(AS_OF),
            run_axis_rf_ids=w.axis.factor_ids,
        )
    native = {RF["USDCHF"], jpy}
    v1_axis = np.array([f for f in w.axis.factor_ids.tolist() if f not in native], dtype=np.int32)
    # Compiled rows: the four V1 prefix nodes and the V1 suffix node, nothing native.
    assert sorted(full.row_target_rf_ids.tolist()) == sorted(v1_axis.tolist())
    assert run.manifest.shocks.proxy_model_digest == hash_proxy_model(full)
    # World request: every V1 row's own LOG column, the ABS node's native
    # column; nothing for NO_SHIFT.
    expect = {(int(f), LOG) for f in v1_axis.tolist()} | {(RF["USDCHF"], ABS)}
    assert len(requests) == 1 and set(requests[0]) == expect and len(requests[0]) == len(expect)
    # Levels, from the chart definitions of the two-point history.
    s = _shocked(capture)[:, 0]
    row = {n: i for i, n in enumerate(names)}
    np.testing.assert_allclose(
        s[row["USDEUR"]], 0.90 * math.exp(_y("USDEUR", LOG, hist)), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        s[row["USDCHF"]], 0.88 + _y("USDCHF", ABS, hist) * 0.88, rtol=RTOL, atol=ATOL
    )
    assert s[row["USDJPY"]] == 150.0
    np.testing.assert_allclose(
        s[row["EQ_JPY"]], 2000.0 * math.exp(_y("EQ_JPY", LOG, hist)), rtol=RTOL, atol=ATOL
    )
    # The reporting conversion reads each suffix FX level from the shocked axis.
    fx = capture["convert_in"]["fx_levels"]
    shocked = _shocked(capture)
    assert sorted(
        next(n for n in ("USDEUR", "USDCHF", "USDJPY") if fx[i].tobytes() == shocked[row[n]].tobytes())
        for i in range(fx.shape[0])
    ) == ["USDCHF", "USDEUR", "USDJPY"]


# ---- container terms and factor-state admissibility on the production run ---------------------


def test_a_container_term_runs_end_to_end_and_equals_its_node_expansion(capture) -> None:
    """ContainerProxyTerm BRENT_FWD <- 0.5 * Y(BRENT_FWD pillar, DIFFERENCE),
    EXACT pairing (each pillar with itself). kappa LOG != tau DIFF:
    A_s = A_0 e^(0.5 * dY). The container form does not survive compilation:
    its digest equals the digest of the explicit node terms."""
    from riskquant.config.proxy_model.schema_objects import ContainerProxyTerm

    rows = [spot(1, "BRENT_FWD[M+0]", "USD", 1.0), spot(2, "BRENT_FWD[M+1]", "USD", 1.0)]
    base = {"BRENT_FWD[M+0]": 80.0, "BRENT_FWD[M+1]": 81.0}
    hist = {"BRENT_FWD[M+0]": [80.0, 82.0], "BRENT_FWD[M+1]": [81.0, 80.5]}
    w = World(rows, "USD", base, hist)
    kw = w.inputs()
    kw["proxy_terms"] = (ContainerProxyTerm(
        target_container_name="BRENT_FWD", driver_container_name="BRENT_FWD",
        driver_chart=ShiftType.DIFFERENCE, loading=0.5),)
    run = run_scenario_var_es(**kw)
    s = _shocked(capture)
    for name, (v0, v1) in hist.items():
        y = float(np.float32(v1 - v0))
        np.testing.assert_allclose(s[w.row(name), 0], base[name] * math.exp(0.5 * y), rtol=RTOL, atol=ATOL)
    nodes = tuple(_term(n, n, DIFF, 0.5) for n in ("BRENT_FWD[M+0]", "BRENT_FWD[M+1]"))
    assert run.manifest.shocks.proxy_model_digest == hash_proxy_model(_compile(w, nodes))


@pytest.mark.parametrize(("beta", "admissible"), [(1.0, False), (0.25, True)])
def test_chart_reconstruction_is_checked_by_the_factor_state_guard(beta, admissible) -> None:
    """USDEUR (valuation prefix, kappa SIMPLE, A_0 = 1) <- beta * Y(EQ_USD, DIFF)
    with dY = -2. beta = 1: SIMPLE reconstruction gives 1 * (1 - 2) = -1, which
    the market-data factor-state guard rejects for an FX node. beta = 0.25 gives
    0.5 and runs. The kernel reconstructs; admissibility is not its concern."""
    rows = [spot(1, "EQ_USD", "USD", 1.0), spot(2, "EQ_USD", "EUR", 1.0)]
    w = _world(rows=rows, base={"EQ_USD": 100.0, "USDEUR": 1.0},
               hist={"EQ_USD": [100.0, 98.0], "USDEUR": [1.0, 1.0]}, rsc=_rsc(USDEUR=SIMPLE))
    assert w.axis.reporting_factor_ids.size == 0  # USDEUR is a valuation factor
    kw = w.inputs()
    kw["proxy_terms"] = (_term("USDEUR", "EQ_USD", DIFF, beta),)
    if admissible:
        assert np.isfinite(run_scenario_var_es(**kw).var).all()
    else:
        with pytest.raises(ValueError):
            run_scenario_var_es(**kw)


# ---- D093: frame authoring is a lossless adapter into the typed production path -------------


def test_frame_authoring_equals_typed_authoring_through_the_production_run(capture) -> None:
    from riskquant.config.proxy_model.ingest import PROXY_MODEL_TERM_SCHEMA, proxy_terms_from_df

    typed = (_term("EQ_USD", "USDEUR", DIFF, 0.3), _term("EQ_USD", "USDEUR", DIFF, 0.2),
             _term("EQ_USD", "USDEUR", LOG, -0.1))
    frame = pl.DataFrame({
        "term_kind": ["NODE"] * 3, "target_name": ["EQ_USD"] * 3, "driver_name": ["USDEUR"] * 3,
        "driver_chart": ["LOG_RETURN", "DIFFERENCE", "DIFFERENCE"], "loading": [-0.1, 0.2, 0.3],
    }, schema=PROXY_MODEL_TERM_SCHEMA)  # the same terms, authored in another row order
    results = []
    for terms in (typed, proxy_terms_from_df(frame)):
        capture["shocked"].clear()
        kw = _world().inputs()
        kw["proxy_terms"] = terms
        run = run_scenario_var_es(**kw)
        results.append((run, _shocked(capture).copy()))
    (a, sa), (b, sb) = results
    assert a.manifest.shocks.proxy_model_digest == b.manifest.shocks.proxy_model_digest
    assert a.identity == b.identity and a.manifest.to_json() == b.manifest.to_json()
    assert sa.tobytes() == sb.tobytes()
    assert a.var.tobytes() == b.var.tobytes() and a.es.tobytes() == b.es.tobytes()


# ---- ScenarioSetBatch lifetime: the run keeps no view of a batch --------------------------


def test_the_run_consumes_each_batch_before_requesting_the_next(monkeypatch) -> None:
    """Every batch is handed out in private storage that is overwritten with NaN
    as soon as the next batch is requested (and after the stream ends). A run
    that kept a view of `shocks` past its iteration would change or fail; the
    run is bitwise identical to the undisturbed run."""
    w = _world(hist={"EQ_USD": [100.0, 101.0, 99.5, 102.0, 100.5, 103.0],
                     "USDEUR": [0.90, 0.91, 0.905, 0.92, 0.915, 0.93]})
    kw = w.inputs()
    kw["proxy_terms"] = (_term("EQ_USD", "USDEUR", DIFF, 0.5),)
    kw["batch_size"] = 1  # five scenarios, five batches
    clean = run_scenario_var_es(**kw)
    assert clean.var.size and np.isfinite(clean.var).all()
    real = api.build_scenario_generator

    class Poisoning:
        def __init__(self, g: Any) -> None:
            self._g = g
            self.total_scenarios = g.total_scenarios

        def iter_batches(self, **k: Any):
            previous = None
            for b in self._g.iter_batches(**k):
                if previous is not None:
                    previous.fill(np.nan)
                private = np.asfortranarray(np.array(b.shocks, copy=True))
                previous = private
                yield dataclasses.replace(b, shocks=private)
            if previous is not None:
                previous.fill(np.nan)

    monkeypatch.setattr(api, "build_scenario_generator", lambda *a, **k: Poisoning(real(*a, **k)))
    poisoned = run_scenario_var_es(**kw)
    assert poisoned.var.tobytes() == clean.var.tobytes()
    assert poisoned.es.tobytes() == clean.es.tobytes()
