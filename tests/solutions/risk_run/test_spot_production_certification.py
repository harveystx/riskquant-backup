# tests/solutions/risk_run/test_spot_production_certification.py
"""
SPOT production slice: end-to-end market-risk certification, ingest -> VaR/ES.

Primary fixture (real, transparent): the canonical demo static configuration
with HOSTILE node ids (USDEUR 9007, AAPL_EQ 12, BRENT_CRUDE 4002, WTI_CRUDE
101). The run builds the axis [9007, 12, 101, 4002]; the source order
[9007, 12, 4002, 4002] is a non-involutive permutation of the history-matrix
order [12, 4002, 9007] (the SimLab P0 pattern). Six SPOT positions with hostile,
unsorted UIDs are ingested from the raw frame and compiled BY THE RUN (N1),
split FULL_REVAL / TAYLOR_DERIVED, long and short, settled in EUR (reciprocal
USDEUR quote) and USD, reported in USD.

Independent oracle (plain Python/NumPy, keyed by factor NAME and economic UID):
it reads the raw demo sources (snapshot levels, history table, proxy rule) and
the raw positions frame. It never calls scenario generation, valuation, the
Taylor calculator, the PnlGrid, D081 or VaR/ES.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

import riskquant.solutions.risk_run.api as api
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.run_identity.hashing import hash_compiled_portfolio
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    static_config_content_digest,
    valuation_contract_fingerprint,
)

FX = mf.FX
UIDS, RF = mf.HOSTILE, mf.HOSTILE_RF
CONF = 0.75  # m = 0.25 * S: 2.25 for S = 9, 16 for S = 64 (exact)
ATOL = 1e-3  # float32 PnlGrid (the established grid tolerance)
ATOL_V0 = 1e-9
ATOL_LEVEL = 1e-5  # float32 shocks, |shock| < 0.05, levels < 250
SOURCES = tuple(FX._SCENARIO_SOURCE_NAMES)
PROXY = {
    "USDEUR": ("USDEUR", 1.0),
    "AAPL_EQ": ("AAPL_EQ", 1.0),
    "BRENT_CRUDE": ("BRENT_CRUDE", 1.0),
    "WTI_CRUDE": ("BRENT_CRUDE", 0.95),
}  # _PROXY_CONFIG_DF
_frame = mf.positions(UIDS)
POS = {
    int(u): (float(q), str(n), str(c))
    for u, q, n, c in zip(
        _frame["uid"],
        _frame["quantity"],
        _frame["underlying"],
        _frame["settlement_ccy"],
        strict=True,
    )
}
DERIVED = set(mf.slots(UIDS, mf.DERIVED_SLOTS))
MODELS = {
    "historical": {"model": "historical", "horizon": 1},
    "filtered_historical": {"model": "filtered_historical", "horizon": 1},
    "sbb": {
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": 64,
        "expected_block_size": 3.0,
        "seed": 7,
    },
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": 64,
        "seed": 7,
    },
    "parametric": {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": 64,
        "distribution": {"kind": "student_t", "df": 5.0},
        "seed": 7,
    },
}
VOL = {"USDEUR": 0.01, "AAPL_EQ": 0.02, "BRENT_CRUDE": 0.03}


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------
def oracle_base():
    return {n: float(FX._RISK_FACTOR_LEVELS[n]) for n in PROXY}


def oracle_historical_levels():
    """Historical, horizon 1, LOG_RETURN: x_s = x_0 * exp(scale * ln(H_t / H_t-1))."""
    hist = np.asarray(FX._HISTORY_LEVELS, dtype=np.float64)
    col = {n: hist[:, j] for j, n in enumerate(SOURCES)}
    base = oracle_base()
    return {
        n: base[n] * np.exp(s * np.log(col[src][1:] / col[src][:-1]))
        for n, (src, s) in PROXY.items()
    }


def value(uid, lv):
    qty, und, ccy = POS[uid]
    return qty * lv[und] * (lv["USDEUR"] if ccy == "EUR" else 1.0)


def rate(uid, lv):  # USD per unit of settlement currency
    return 1.0 / lv["USDEUR"] if POS[uid][2] == "EUR" else 1.0


def delta_v(uid, x0, xs):
    if uid not in DERIVED:  # FULL_REVAL: exact revaluation
        return value(uid, xs) - value(uid, x0)
    qty, und, ccy = POS[uid]  # SECOND_ORDER_TRUNCATED: first order (spot)
    dv = qty * (xs[und] - x0[und]) * (x0["USDEUR"] if ccy == "EUR" else 1.0)
    if ccy == "EUR":
        dv = dv + qty * x0[und] * (xs["USDEUR"] - x0["USDEUR"])
    return dv


def oracle_rows(x0, xs):
    rows = {}
    for uid in POS:
        v0, dv, q0, qs = (
            value(uid, x0),
            delta_v(uid, x0, xs),
            rate(uid, x0),
            rate(uid, xs),
        )
        dw_a = (v0 + dv) * qs - v0 * q0  # D081: V_s q_s - V_0 q_0
        dw_b = dv * qs + v0 * (qs - q0)  # D081: dV q_s + V_0 (q_s - q_0)
        np.testing.assert_allclose(dw_a, dw_b, rtol=1e-12, atol=1e-9)
        rows[uid] = (v0, dv, dw_b)
    return rows


def oracle_var_es(p, c):
    """Documented convention: m = (1 - c) S; VaR = ceil(m)-th lower order statistic;
    ES integrates exactly m observations' mass (fractional boundary weight)."""
    s = np.sort(np.asarray(p, dtype=np.float64))
    m = (1.0 - c) * s.shape[0]
    k, f = math.ceil(m), math.floor(m)
    es = s[:f].sum() / m if k == f else (s[:f].sum() + (m - f) * s[f]) / m
    return float(s[k - 1]), float(es)


# ---------------------------------------------------------------------------
# Production capture
# ---------------------------------------------------------------------------
@pytest.fixture
def cap(monkeypatch):
    seen = {}
    real_write, real_convert = (
        api._write_full_reval_block,
        api.convert_scenario_pnl_to_reporting_currency,
    )

    def write(**kw):
        seen["shocked"] = np.array(kw["shocked_levels"], copy=True)
        return real_write(**kw)

    def convert(**kw):
        seen["convert"] = {
            k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for k, v in kw.items()
        }
        out = real_convert(**kw)
        seen["reported"] = np.array(
            out.converted_pnl, dtype=np.float64, copy=True
        )  # production D081 output
        return out

    monkeypatch.setattr(api, "_write_full_reval_block", write)
    monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", convert)
    return seen


def _inputs(model="historical", rf=RF, batch_size=4):
    inp = mf.inputs(uids=UIDS, rf=rf, batch_size=batch_size)
    inp["confidence_level"] = CONF
    inp["scenario_config"] = parse_scenario_model_config(MODELS[model])
    if model == "parametric":
        w = mf.world(UIDS, rf)
        ids = sorted(int(w.rf[n]) for n in SOURCES)
        name = {int(w.rf[n]): n for n in SOURCES}
        inp["data"] = ParametricRiskData(
            factor_ids=np.array(ids, np.int32),
            cholesky_matrix=np.ascontiguousarray(np.eye(3)),
            volatilities=np.ascontiguousarray([VOL[name[i]] for i in ids]),
            effective_period_count=1,
            effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
            chart_ids=np.full(len(ids), SHIFT_LOG_RETURN_ID, dtype=np.int8),
        )
    return inp


def _by_name(w, arr):
    return {n: np.asarray(arr[mf._row(w, n)], dtype=np.float64) for n in PROXY}


def _production_rows(run, seen):
    c = seen["convert"]
    rows = {}
    for i, uid in enumerate(run.position_uids.tolist()):
        rows[uid] = (
            float(c["base_settlement_values"][i]),
            np.asarray(c["settlement_pnl"][i], np.float64),
        )
    return rows


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------
def test_run_axes_are_authoritative_and_hostile(cap) -> None:
    w = mf.world(UIDS, RF)
    inp = _inputs()
    ctx = inp["valuation_context"]
    assert _frame["uid"].tolist() == list(UIDS) != sorted(UIDS)  # unsorted source order
    assert ctx.portfolio.trade_uids.tolist() == list(UIDS)  # compiled by the run (N1)
    assert inp["methodology"].production_position_uids.tolist() == sorted(UIDS)
    assert w.su.required_factor_ids.tolist() == [
        9007,
        12,
        101,
        4002,
    ]  # from the universe map
    # The fixture's one authored Proxy V1 term (D019): WTI_CRUDE (rf 101, no
    # history) <- 0.95 * BRENT_CRUDE (rf 4002) log return; every other row is
    # its identity row.
    assert [(w.rf[t.target_name], w.rf[t.driver_name], int(t.driver_chart), t.loading)
            for t in w.su.proxy_terms] == [(101, 4002, 1, 0.95)]
    assert w.historical.factor_ids.tolist() == [12, 4002, 9007]
    run = run_scenario_var_es(**inp)
    assert run.position_uids.tolist() == sorted(UIDS)  # final economic axis = U
    assert cap["shocked"].shape == (4, 9)


def test_canonical_scenario_levels_equal_the_oracle_by_factor_name(cap) -> None:
    w = mf.world(UIDS, RF)
    hist = np.asarray(FX._HISTORY_LEVELS)
    for j, n in enumerate(SOURCES):  # source table reading
        assert np.array_equal(
            np.asarray(w.historical.values)[
                :, w.historical.factor_ids.tolist().index(w.rf[n])
            ],
            hist[:, j],
        )
    run_scenario_var_es(**_inputs())
    got, want = _by_name(w, cap["shocked"]), oracle_historical_levels()
    for n in PROXY:
        np.testing.assert_allclose(got[n], want[n], rtol=0, atol=ATOL_LEVEL, err_msg=n)


def test_v0_dv_and_reporting_pnl_reconcile_per_uid(cap) -> None:
    run = run_scenario_var_es(**_inputs())
    prod, want = (
        _production_rows(run, cap),
        oracle_rows(oracle_base(), oracle_historical_levels()),
    )
    reported = cap["reported"]
    for i, uid in enumerate(run.position_uids.tolist()):
        v0, dv, dw = want[uid]
        assert prod[uid][0] == pytest.approx(v0, abs=ATOL_V0), uid
        np.testing.assert_allclose(
            prod[uid][1], dv, rtol=0, atol=ATOL, err_msg=f"dV {uid}"
        )
        np.testing.assert_allclose(
            reported[i], dw, rtol=0, atol=ATOL, err_msg=f"dW {uid}"
        )
    assert any(want[u][0] < 0 for u in want) and any(POS[u][2] == "EUR" for u in POS)


def test_portfolio_var_es_equal_the_independent_reference(cap) -> None:
    run = run_scenario_var_es(**_inputs())
    want = oracle_rows(oracle_base(), oracle_historical_levels())
    portfolio = sum(want[u][2] for u in want)
    var, es = oracle_var_es(portfolio, CONF)
    assert float(run.var[-1]) == pytest.approx(var, abs=6 * ATOL)
    assert float(run.es[-1]) == pytest.approx(es, abs=6 * ATOL)
    for i, uid in enumerate(run.position_uids.tolist()):  # per-position groups
        pv, pe = oracle_var_es(want[uid][2], CONF)
        assert float(run.var[i]) == pytest.approx(pv, abs=ATOL) and float(
            run.es[i]
        ) == pytest.approx(pe, abs=ATOL)


@pytest.mark.parametrize("model", list(MODELS))
def test_every_admitted_scenario_model_feeds_one_downstream_contract(
    cap, model
) -> None:
    """The downstream path does not care how scenarios were generated: from the
    run's own canonical levels, the oracle reproduces every UID row and VaR/ES."""
    w = mf.world(UIDS, RF)
    run = run_scenario_var_es(**_inputs(model))
    xs = _by_name(w, cap["shocked"])
    want = oracle_rows(oracle_base(), xs)
    reported = cap["reported"]
    for i, uid in enumerate(run.position_uids.tolist()):
        np.testing.assert_allclose(
            reported[i], want[uid][2], rtol=0, atol=ATOL, err_msg=f"{model} {uid}"
        )
    var, es = oracle_var_es(sum(want[u][2] for u in want), CONF)
    assert float(run.var[-1]) == pytest.approx(var, abs=6 * ATOL)
    assert float(run.es[-1]) == pytest.approx(es, abs=6 * ATOL)


@pytest.mark.parametrize(
    "model", ["historical", "filtered_historical", "sbb", "filtered_bootstrap"]
)
def test_hostile_and_sorted_worlds_are_economically_identical(cap, model) -> None:
    """Same economics, different factor-id assignment and source order: equal
    per-name scenario levels, per-UID P&L and VaR/ES."""
    out = {}
    for tag, rf in (("hostile", RF), ("sorted", None)):
        w = mf.world(UIDS, rf)
        run = run_scenario_var_es(**_inputs(model, rf=rf))
        out[tag] = (_by_name(w, cap["shocked"]), run)
    for n in PROXY:
        np.testing.assert_allclose(
            out["hostile"][0][n], out["sorted"][0][n], rtol=0, atol=1e-9, err_msg=n
        )
    a, b = out["hostile"][1], out["sorted"][1]
    assert a.position_uids.tolist() == b.position_uids.tolist()
    np.testing.assert_allclose(a.var, b.var, rtol=0, atol=ATOL)
    np.testing.assert_allclose(a.es, b.es, rtol=0, atol=ATOL)
    assert a.identity != b.identity  # different static config


def test_batch_size_and_repetition_do_not_move_the_certified_numbers() -> None:
    runs = [run_scenario_var_es(**_inputs(batch_size=b)) for b in (1, 3, 64, 3)]
    for r in runs[1:]:
        assert r.identity == runs[0].identity
        assert [v.hex() for v in r.var.tolist()] == [
            v.hex() for v in runs[0].var.tolist()
        ]
        assert [v.hex() for v in r.es.tolist()] == [
            v.hex() for v in runs[0].es.tolist()
        ]


def test_certified_manifest_carries_the_run_owned_context() -> None:
    inp = _inputs()
    run = run_scenario_var_es(**inp)
    ctx = inp["valuation_context"]
    c = run.manifest.pnl.composition.context
    assert c.as_of_ord32 == ctx.uix_mapper.as_of_ord32
    assert c.static_config_digest == static_config_content_digest(ctx.static_config)
    assert c.valuation_contract_fingerprint == valuation_contract_fingerprint()
    assert (
        c.discounting_mode == "NONE" or c.discounting_mode == ctx.discounting_mode.value
    )
    assert c.portfolio_digest == hash_compiled_portfolio(ctx.portfolio)
    assert (
        run.manifest.pnl.composition.methodology.partition_digest
        == inp["methodology"].digest()
    )
