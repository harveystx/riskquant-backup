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


def test_log_target_driven_by_difference_uses_the_difference_column(capture) -> None:
    w = _world()
    kw = w.inputs()
    kw["proxy_terms"] = (_term("EQ_USD", "USDEUR", DIFF, 0.5),)
    run_scenario_var_es(**kw)
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


def test_the_run_has_no_d013_inputs_or_calls(monkeypatch) -> None:
    params = inspect.signature(run_scenario_var_es).parameters
    assert "proxy_terms" in params
    assert not {"source_factor_ids", "shock_scale"} & set(params)
    for name in ("apply_shocks_to_base_levels", "_require_shock_sources"):
        assert not hasattr(api, name)

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("D013 path reached")

    import riskquant.proxy_resolver.proxy as legacy_resolver
    import riskquant.simlab.shocks.apply as legacy_apply

    monkeypatch.setattr(legacy_apply, "apply_shocks_to_base_levels", boom)
    monkeypatch.setattr(
        legacy_resolver, "resolve_required_factor_ids_to_market_data_factor_ids", boom
    )
    run = run_scenario_var_es(**_world().inputs())
    assert np.isfinite(run.var).all()
