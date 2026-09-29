# tests/solutions/risk_run/test_reporting_axis.py
"""
D085 / D082 (amended): run-derived settlement currencies, run-derived reporting
FX, and ONE canonical run axis = valuation prefix ++ reporting-only FX suffix.

Every numerical oracle here is independent of `riskquant.reporting`: settlement
values are `q * S` (and `q * S * x` for a direct instrument FX leg), the
settlement -> reporting rate is chosen by hand from the pair orientation, and
the D081 identity dW = V_s q_s - V_0 q_0 is evaluated on the run's own
canonical shocked levels (captured at `apply_linear_terms_to_base_levels`). Two
repository representation contracts are applied to the oracle explicitly,
never as a tolerance: scenario shocks are float32, and the PnlGrid stores
settlement dV in float32.

Static configuration: the hostile remediation fixture (pairs USDEUR direct,
EURGBP and USDCHF; JPY configured without a pair).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REMEDIATION = Path(__file__).resolve().parents[2] / "market_data" / "remediation"
sys.path.insert(0, str(_REMEDIATION))
import hostile_fixture as hf

import riskquant.solutions.risk_run.api as api
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.api import derive_run_factor_axis, run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
    RunFactorAxis,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.book_adapter import (
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

RSC = compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())
RF = dict(RSC.rel_name_to_rf_id)
NAME = {v: k for k, v in RF.items()}
AS_OF = int(date_to_ord32(hf.AS_OF))
NONE = DiscountingMode.NONE
HISTORICAL = {"model": "historical", "horizon": 1}
_EXPENSIVE = (
    "build_run_manifest",
    "_value_positions",
    "build_scenario_generator",
    "allocate_pnl_grid",
    "compute_var_es",
)


def spot(uid: int, underlying: str, settle: str, qty: float) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "SPOT",
        "quantity": qty,
        "underlying": underlying,
        "settlement_ccy": settle,
    }


class World:
    """One book on the real production path, bound to its canonical run axis."""

    def __init__(self, rows, reporting_ccy, base, hist) -> None:
        self.book = pd.DataFrame(rows)
        self.qty = dict(zip(self.book.uid, self.book.quantity, strict=True))
        self.batch = compile_raw_df_into_ingest_batch(
            raw_df=self.book, static_config=RSC, copy=True
        )
        self.mapper = build_uix_mapper([self.batch], as_of_ord32=AS_OF)
        self.context = ValuationContext(
            static_config=RSC,
            ingest_batch=self.batch,
            uix_mapper=self.mapper,
            discounting_mode=NONE,
        )
        self.reporting_ccy = reporting_ccy
        self.axis = derive_run_factor_axis(
            valuation_context=self.context, reporting_ccy=reporting_ccy
        )
        self.names = [NAME[int(f)] for f in self.axis.factor_ids]
        self.base = np.array([base[n] for n in self.names], dtype=np.float64)
        self.hist = hist
        self.uids = np.array(sorted(self.book.uid), dtype=np.int64)

    def row(self, name: str) -> int:
        return self.names.index(name)

    def data(self, drop: tuple[str, ...] = ()) -> RiskFactorMatrix:
        keep = [f for f in np.sort(self.axis.factor_ids) if NAME[int(f)] not in drop]
        values = np.column_stack([self.hist[NAME[int(f)]] for f in keep])
        t = values.shape[0]
        return RiskFactorMatrix(
            dates=np.arange(AS_OF - t + 1, AS_OF + 1, dtype=np.int32),
            factor_ids=np.array(keep, dtype=np.int32),
            values=np.asfortranarray(values),
        )

    def inputs(self, *, derived=(), block=None, **over: Any) -> dict[str, Any]:
        u_d = np.array(sorted(derived), dtype=np.int64)
        kw: dict[str, Any] = {
            "scenario_config": parse_scenario_model_config(HISTORICAL),
            "data": self.data(),
            "required_factor_ids": self.axis.factor_ids.copy(),
            "required_shift_type_id": self.axis.shift_type_ids.copy(),
            "proxy_terms": (),
            "base_levels": self.base.copy(),
            "methodology": MethodologyDeclaration(
                production_position_uids=self.uids,
                full_reval_uids=np.setdiff1d(self.uids, u_d),
                taylor_supplied_uids=np.empty(0, dtype=np.int64),
                taylor_derived_uids=u_d,
            ),
            "valuation_context": self.context,
            "reporting": ReportingSpec(reporting_ccy=self.reporting_ccy),
            "confidence_level": 0.5,
            "batch_size": 7,
            "taylor_derived": block,
        }
        kw.update(over)
        return kw

    def derived_block(self, derived, *, axis_ids=None) -> TaylorBlock:
        """U_D-scoped production sensitisation, certified on the valuation prefix."""
        k_val = self.axis.num_valuation_factors
        axis_ids = self.axis.factor_ids[:k_val] if axis_ids is None else axis_ids
        sub = self.book[self.book.uid.isin(derived)].reset_index(drop=True)
        b = compile_raw_df_into_ingest_batch(raw_df=sub, static_config=RSC, copy=True)
        m = build_uix_mapper([b], as_of_ord32=AS_OF)
        p = compile_portfolio_from_ingest_batch(ingest_batch=b, uix_mapper=m)
        layout = build_universe_mapping(
            static_config=RSC, uix_mapper=m, discounting_mode=NONE
        )
        levels = np.array([
            self.base[self.row(NAME[int(f)])] for f in layout.required_factor_ids
        ])
        mx = build_market_data_matrix_from_flat_levels(
            discounting_mode=NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=m,
            universe_map=layout,
            flat_levels=np.ascontiguousarray(levels[:, None]),
            num_scenarios=1,
        )
        out = np.empty((p.total_trades, 1))
        run_valuation(
            portfolio=p,
            valuation_state=assemble_valuation_state(matrix=mx, uix_mapper=m),
            out=out,
        )
        sens = run_sensitivities(portfolio=p, uix_mapper=m, market_data=mx)
        book = np.unique(sens.position_uids)
        binding = TaylorBaseBinding(
            as_of_ord32=AS_OF,
            base_levels_digest=base_levels_digest(
                factor_ids=self.axis.factor_ids[:k_val], base_levels=self.base[:k_val]
            ),
            discounting_mode=NONE.value,
            static_config_digest=static_config_content_digest(RSC),
            valuation_contract_fingerprint=valuation_contract_fingerprint(),
            base_settlement_values_digest=base_settlement_values_digest(
                position_uids=book, values=out[:, 0]
            ),
        )
        spec = build_sensitivity_book_spec(
            sensitivities=sens, required_factor_ids=axis_ids
        )
        spec = dataclasses.replace(
            spec,
            taylor_declaration=derive_taylor_declaration(
                sensitivities=sens,
                order=TaylorOrder.SECOND_ORDER_FULL,
                base_binding=binding,
            ),
        )
        return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


@pytest.fixture
def capture(monkeypatch):
    """Record the canonical shocked levels and the conversion's inputs/outputs."""
    seen: dict[str, Any] = {"shocked": []}
    real_apply = api.apply_linear_terms_to_base_levels
    real_convert = api.convert_scenario_pnl_to_reporting_currency

    def apply(**kw: Any) -> None:
        real_apply(**kw)
        seen["shocked"].append(kw["out"].copy())

    def convert(**kw: Any):
        seen["convert_in"] = kw
        seen["convert_out"] = real_convert(**kw)
        return seen["convert_out"]

    monkeypatch.setattr(api, "apply_linear_terms_to_base_levels", apply)
    monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", convert)
    return seen


@pytest.fixture
def no_numerics(monkeypatch):
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the run reached identity or numerics")

    for stage in _EXPENSIVE:
        monkeypatch.setattr(api, stage, boom)


def _oracle_dw(w: World, levels: np.ndarray, value, rate) -> dict[int, np.ndarray]:
    """dW per UID: float32 grid dV, then the D081 identity by hand."""
    at = {n: levels[w.row(n)] for n in w.names}
    at0 = {n: float(w.base[w.row(n)]) for n in w.names}
    out = {}
    for uid in w.uids.tolist():
        v0, vs = value(uid, at0), value(uid, at)
        q0, qs = rate(uid, at0), rate(uid, at)
        dv32 = np.asarray(vs - v0, dtype=np.float32).astype(np.float64)
        out[uid] = dv32 * qs + v0 * (qs - q0)
    return out


def _shocked(seen) -> np.ndarray:
    return np.hstack(seen["shocked"])


# ---- A: single USD position, EUR reporting: the motivating capability ---------------


A_ROWS = [spot(1, "EQ_USD", "USD", 1.0)]
A_BASE = {"EQ_USD": 100.0, "USDEUR": 0.90}
A_HIST = {"EQ_USD": [100.0, 110.0], "USDEUR": [0.90, 0.80]}


def test_a_reporting_only_fx_enters_the_run_axis_as_a_suffix() -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    assert w.names == ["EQ_USD", "USDEUR"]
    assert w.axis.num_valuation_factors == 1
    assert w.axis.reporting_factor_ids.tolist() == [RF["USDEUR"]]
    assert w.axis.shift_type_ids[1] == RSC.rf_id_to_shift_type_id[RF["USDEUR"]]


def test_a_single_usd_position_reports_in_eur_exactly(capture) -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    run = run_scenario_var_es(**w.inputs())
    levels = _shocked(capture)
    oracle = _oracle_dw(
        w, levels, lambda u, x: 1.0 * x["EQ_USD"], lambda u, x: x["USDEUR"]
    )[1]
    got = capture["convert_out"].converted_pnl[0]
    assert got.tobytes() == oracle.tobytes()
    # the analytic example: 110 * 0.80 - 100 * 0.90 = -2; the residual is the
    # float32 shock contract on ln(1.1) and ln(0.8/0.9), plus the float32 grid dV
    assert abs(float(got[0]) + 2.0) < 1e-6
    assert run.portfolio_var.hex() == float(oracle[0]).hex()  # S = 1, q = 0.5


# ---- B / E / G: mixed currencies, reciprocal quote, shared factor --------------------


B_BASE = {"EQ_USD": 212.37, "EQ_GBP": 148.91, "USDEUR": 0.9259, "EURGBP": 0.8561}
_RNG = np.random.default_rng(11)
B_HIST = {
    n: v * np.exp(np.cumsum(np.r_[0.0, _RNG.normal(0.0, 0.02, 39)]))
    for n, v in B_BASE.items()
}
B_ROWS = [
    spot(1, "EQ_USD", "USD", 100.0),
    spot(2, "EQ_GBP", "GBP", -250.0),
    spot(3, "EQ_USD", "USD", 7.0),
]
B_VALUE = {1: ("EQ_USD", 100.0), 2: ("EQ_GBP", -250.0), 3: ("EQ_USD", 7.0)}


def _b_rate(uid: int, x: dict[str, Any]):
    # USD -> EUR: USDEUR direct.  GBP -> EUR: only EURGBP is quoted, q = 1 / x.
    return x["USDEUR"] if B_VALUE[uid][0] == "EQ_USD" else 1.0 / x["EURGBP"]


def test_b_mixed_currencies_each_reporting_factor_exactly_once(capture) -> None:
    w = World(B_ROWS, "EUR", B_BASE, B_HIST)
    assert w.axis.reporting_factor_ids.tolist() == sorted([RF["USDEUR"], RF["EURGBP"]])
    run = run_scenario_var_es(**w.inputs())
    levels = _shocked(capture)
    oracle = _oracle_dw(
        w, levels, lambda u, x: B_VALUE[u][1] * x[B_VALUE[u][0]], _b_rate
    )
    got = capture["convert_out"].converted_pnl
    for i, uid in enumerate(w.uids.tolist()):
        assert got[i].tobytes() == oracle[uid].tobytes(), uid
    rows = np.vstack([oracle[u] for u in w.uids.tolist()])
    ref = compute_var_es(np.vstack([rows, rows.sum(axis=0)[None, :]]), 0.5)
    assert [v.hex() for v in run.var] == [float(v).hex() for v in ref.var]
    assert [v.hex() for v in run.es] == [float(v).hex() for v in ref.es]


def test_e_reciprocal_reporting_quote_is_inverted_exactly_once(capture) -> None:
    w = World(B_ROWS, "EUR", B_BASE, B_HIST)
    run_scenario_var_es(**w.inputs())
    plan = capture["convert_in"]["plan"]
    fx_levels = capture["convert_in"]["fx_levels"]
    gbp = list(w.uids).index(2)
    assert int(plan.invert[gbp]) == 1
    row = int(plan.fx_row_index[gbp])
    np.testing.assert_array_equal(fx_levels[row], _shocked(capture)[w.row("EURGBP")])


def test_g_positions_sharing_a_currency_share_one_suffix_row(capture) -> None:
    w = World(B_ROWS, "EUR", B_BASE, B_HIST)
    run_scenario_var_es(**w.inputs())
    plan = capture["convert_in"]["plan"]
    i1, i3 = list(w.uids).index(1), list(w.uids).index(3)
    assert plan.fx_row_index[i1] == plan.fx_row_index[i3]
    assert capture["convert_in"]["fx_levels"].shape[0] == 2


# ---- C / D: identity conversion; factor already on the valuation prefix --------------


def test_c_source_equal_to_reporting_needs_no_factor(capture) -> None:
    rows = [spot(1, "EQ_EUR", "EUR", 3.0)]
    w = World(rows, "EUR", {"EQ_EUR": 53.12}, {"EQ_EUR": B_HIST["EQ_USD"]})
    assert w.axis.reporting_factor_ids.size == 0
    run_scenario_var_es(**w.inputs())
    assert capture["convert_in"]["fx_levels"].shape[0] == 0
    np.testing.assert_array_equal(
        capture["convert_out"].converted_pnl,
        capture["convert_in"]["settlement_pnl"],
    )


def test_d_reporting_factor_already_in_the_prefix_is_not_duplicated(capture) -> None:
    rows = [spot(1, "EQ_USD", "USD", 100.0), spot(2, "EQ_USD", "EUR", 40.0)]
    w = World(rows, "EUR", B_BASE, B_HIST)
    assert w.axis.reporting_factor_ids.size == 0
    assert w.names.count("USDEUR") == 1
    run_scenario_var_es(**w.inputs())
    levels = _shocked(capture)

    def value(uid, x):
        return 100.0 * x["EQ_USD"] if uid == 1 else 40.0 * x["EQ_USD"] * x["USDEUR"]

    oracle = _oracle_dw(w, levels, value, lambda u, x: x["USDEUR"] if u == 1 else 1.0)
    got = capture["convert_out"].converted_pnl
    for i, uid in enumerate(w.uids.tolist()):
        np.testing.assert_allclose(got[i], oracle[uid], rtol=0, atol=0)


# ---- F: reporting FX absent from the scenario source ---------------------------------


def test_f_missing_reporting_fx_history_fails_at_composition(no_numerics) -> None:
    """A reporting-only suffix node is a Proxy V1 row (D088 §13.8(a)); with no
    authored term its identity row requests (USDEUR, LOG, h). A world without
    that column is refused before identity and numerics; nothing substitutes."""
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    kw = w.inputs(data=w.data(drop=("USDEUR",)))
    with pytest.raises(ValueError, match="not in the matrix world"):
        run_scenario_var_es(**kw)


# ---- H / I: no caller-owned source currency or FX routing remains --------------------


def test_h_i_reporting_spec_carries_only_the_reporting_currency() -> None:
    assert [f.name for f in dataclasses.fields(ReportingSpec)] == ["reporting_ccy"]
    for stale in ("settlement_ccys", "fx_factor_names", "fx_factor_rows"):
        with pytest.raises(TypeError):
            ReportingSpec(reporting_ccy="EUR", **{stale: ()})


def test_h_the_trade_record_alone_defines_the_source_currency(capture) -> None:
    """The same asset settled in EUR instead of USD is a different economic
    position: its source currency, and so its conversion, follow the trade."""
    usd = World(A_ROWS, "USD", A_BASE, A_HIST)
    eur = World([spot(1, "EQ_USD", "EUR", 1.0)], "USD", A_BASE, A_HIST)
    assert usd.axis.reporting_factor_ids.size == 0  # USD value, USD reporting
    assert eur.names.count("USDEUR") == 1  # EUR value: valuation fold on the prefix
    run_scenario_var_es(**eur.inputs())
    plan = capture["convert_in"]["plan"]
    assert plan.settlement_ccys == ("EUR",) and int(plan.invert[0]) == 1


# ---- J / K: one canonical scenario state ---------------------------------------------


def test_j_parametric_reporting_fx_shares_the_joint_draw(capture) -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    rho, vols = 0.8, np.array([0.01, 0.01])
    ids = np.sort(w.axis.factor_ids).astype(np.int32)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    data = ParametricRiskData(
        factor_ids=ids,
        cholesky_matrix=np.ascontiguousarray(chol),
        volatilities=np.ascontiguousarray(vols),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(ids.shape[0], SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )
    cfg = parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": 20_000,
        "distribution": {"kind": "standard_normal"},
        "seed": 3,
    })
    run_scenario_var_es(**w.inputs(scenario_config=cfg, data=data, batch_size=4096))
    levels = _shocked(capture)
    r = np.log(levels / w.base[:, None])
    corr = float(np.corrcoef(r[w.row("EQ_USD")], r[w.row("USDEUR")])[0, 1])
    # 20_000 draws: the sampling s.e. of rho = 0.8 is (1 - rho^2) / sqrt(S) ~ 0.0025
    assert abs(corr - rho) < 0.02


def test_k_bootstrap_uses_one_time_index_for_prefix_and_suffix(capture) -> None:
    w = World(B_ROWS, "EUR", B_BASE, B_HIST)
    cfg = parse_scenario_model_config({
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": 400,
        "expected_block_size": 5.0,
        "seed": 9,
    })
    run_scenario_var_es(**w.inputs(scenario_config=cfg))
    levels = _shocked(capture)
    shocks = np.log(levels / w.base[:, None])
    returns = {
        n: np
        .log(np.asarray(B_HIST[n][1:]) / np.asarray(B_HIST[n][:-1]))
        .astype(np.float32)
        .astype(np.float64)
        for n in w.names
    }
    t = np.argmin(np.abs(returns["EQ_USD"][:, None] - shocks[w.row("EQ_USD")]), axis=0)
    for name in ("EQ_GBP", "USDEUR", "EURGBP"):
        np.testing.assert_allclose(
            shocks[w.row(name)], returns[name][t], rtol=0, atol=1e-12
        )


# ---- L: Taylor / DERIVED on the valuation prefix only --------------------------------


def test_l_derived_uses_the_valuation_prefix_and_matches_full(capture) -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    full = run_scenario_var_es(**w.inputs())
    full_dw = capture["convert_out"].converted_pnl.copy()
    block = w.derived_block([1])
    assert block.calculator.axis.num_factors == w.axis.num_valuation_factors
    derived = run_scenario_var_es(**w.inputs(derived=[1], block=block))
    assert derived.pnl_provenance is not None
    assert capture["convert_out"].converted_pnl.tobytes() == full_dw.tobytes()
    assert derived.portfolio_var.hex() == full.portfolio_var.hex()


def test_l_a_book_on_the_full_run_axis_is_refused(no_numerics) -> None:
    """Reporting FX cannot become a Taylor axis row: a book compiled on the
    expanded axis is not the run's Taylor (valuation-prefix) axis."""
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    block = w.derived_block([1], axis_ids=w.axis.factor_ids)
    with pytest.raises(CompositionContractError, match="factor axis"):
        run_scenario_var_es(**w.inputs(derived=[1], block=block))


# ---- composition negatives: every one fails before identity and numerics -------------


def _swap(a: np.ndarray, i: int, j: int) -> np.ndarray:
    b = np.array(a, copy=True)
    b[[i, j]] = b[[j, i]]
    return b


def test_valuation_only_axis_is_refused(no_numerics) -> None:
    """The caller guessed the valuation universe and omitted the suffix."""
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    k = w.axis.num_valuation_factors
    with pytest.raises(CompositionContractError, match="canonical axis"):
        run_scenario_var_es(
            **w.inputs(
                required_factor_ids=w.axis.factor_ids[:k].copy(),
                required_shift_type_id=w.axis.shift_type_ids[:k].copy(),
                base_levels=w.base[:k].copy(),
            )
        )


@pytest.mark.parametrize(
    "rows", [(0, 1), (2, 3)], ids=["prefix_permuted", "suffix_unsorted"]
)
def test_permuted_prefix_or_unsorted_suffix_is_refused(no_numerics, rows) -> None:
    w = World(B_ROWS, "EUR", B_BASE, B_HIST)
    assert w.axis.num_valuation_factors == 2
    kw = w.inputs()
    for key in (
        "required_factor_ids",
        "required_shift_type_id",
        "base_levels",
    ):
        kw[key] = _swap(kw[key], *rows)
    with pytest.raises(CompositionContractError, match="canonical axis"):
        run_scenario_var_es(**kw)


def test_scenario_data_on_another_axis_length_is_refused(no_numerics) -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    with pytest.raises(CompositionContractError, match="base_levels has shape"):
        run_scenario_var_es(**w.inputs(base_levels=w.base[:1].copy()))


def test_run_factor_axis_refuses_duplicates_and_unsorted_suffix() -> None:
    with pytest.raises(ValueError, match="repeat"):
        RunFactorAxis(
            factor_ids=np.array([7, 31, 31], np.int32),
            shift_type_ids=np.ones(3, np.int8),
            num_valuation_factors=1,
        )
    with pytest.raises(ValueError, match="ascending"):
        RunFactorAxis(
            factor_ids=np.array([7, 402, 31], np.int32),
            shift_type_ids=np.ones(3, np.int8),
            num_valuation_factors=1,
        )


def test_unroutable_reporting_currency_is_refused(no_numerics) -> None:
    w = World(A_ROWS, "EUR", A_BASE, A_HIST)
    with pytest.raises(CompositionContractError, match="no configured FX pair"):
        run_scenario_var_es(**w.inputs(reporting=ReportingSpec(reporting_ccy="JPY")))


def test_underivable_source_currency_is_refused() -> None:
    """An ingest FX leg that names no configured pair has no source currency."""
    w = World([spot(1, "EQ_USD", "EUR", 1.0)], "USD", A_BASE, A_HIST)
    bad = dataclasses.replace(
        w.batch, fx_u2s_ids=np.full_like(w.batch.fx_u2s_ids, RF["EQ_USD"])
    )
    context = type("Ctx", (), {"ingest_batch": bad, "static_config": RSC})()
    with pytest.raises(CompositionContractError, match="not a configured FX pair"):
        api._settlement_ccy_ids(context=context, economic_uids=w.uids)


def test_empty_suffix_keeps_the_valuation_universe_as_the_whole_axis() -> None:
    rows = [spot(1, "EQ_USD", "USD", 1.0), spot(2, "EQ_USD", "EUR", 1.0)]
    w = World(rows, "USD", B_BASE, B_HIST)
    layout = build_universe_mapping(
        static_config=RSC, uix_mapper=w.mapper, discounting_mode=NONE
    )
    assert w.axis.reporting_factor_ids.size == 0
    np.testing.assert_array_equal(w.axis.factor_ids, layout.required_factor_ids)


def test_source_currency_of_an_inverted_instrument_leg_is_the_pair_base() -> None:
    """EQ_GBP settled in EUR folds through EURGBP with invert = 1 (D060): the
    P&L currency is the pair's BASE side (EUR), not its quote side (GBP)."""
    w = World([spot(1, "EQ_GBP", "EUR", 1.0)], "USD", B_BASE, B_HIST)
    assert int(w.batch.fx_u2s_inv[0]) == 1
    settle = api._settlement_ccy_ids(context=w.context, economic_uids=w.uids)
    assert RSC.ccy_id_to_name[int(settle[0])] == "EUR"
    # valuation needs EURGBP (the fold); EUR -> USD reporting needs USDEUR (suffix)
    assert w.axis.reporting_factor_ids.tolist() == [RF["USDEUR"]]
