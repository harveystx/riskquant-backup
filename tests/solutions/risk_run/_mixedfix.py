# tests/solutions/risk_run/_mixedfix.py
"""
Mixed-methodology fixture on the REAL production path.

Six SPOT positions are ingested, compiled and valued through the same public
functions the D019 demo uses (static config, proxy terms, snapshot,
UixMapper, UniverseMapping). The economics of slot k never change; its UID
does. `BASE` UIDs are ascending and contiguous. `HOSTILE` UIDs are sparse and
are ingested in NON-ascending order, so the compiled `trade_uids` are unsorted,
the grid block order interleaves with sorted(U), and no positional coincidence
can hide an index-space defect.

TAYLOR/DERIVED books are certified from the production sensitizer output.
Default order SECOND_ORDER_TRUNCATED (cross terms omitted BY DECLARATION):
for the bilinear quanto payoff qty*P*X that makes dV_T differ from the full
revaluation by qty*dP*dX, so a FULL value leaking into a DERIVED row (or the
reverse) is numerically identifiable. SECOND_ORDER_FULL would be exact here and
would hide exactly that leak.

Independent references (never call production valuation or reporting):
  L1  `l1_value`     : qty * P[underlying] * (USDEUR if settled in EUR)
  L4  `l4_taylor`    : the declared Taylor polynomial of that form, term by term
  L0  `l0_reporting` : V_s*q_s - V_0*q_0, q = 1 (USD) or 1/USDEUR (EUR)
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl

from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world
from riskquant.utils.business_days import ord_to_date

_DEMO = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "demos" / "e2e"
_spec = importlib.util.spec_from_file_location("_mixedfix_demo", _DEMO / "_fixtures.py")
FX = importlib.util.module_from_spec(_spec)
sys.modules["_mixedfix_demo"] = FX  # @dataclass resolves types via sys.modules
_spec.loader.exec_module(FX)

#: slot -> (quantity, underlying, settlement currency)
_ECONOMICS = (
    (100.0, "AAPL_EQ", "USD"),
    (-50.0, "BRENT_CRUDE", "EUR"),
    (75.0, "WTI_CRUDE", "USD"),
    (30.0, "AAPL_EQ", "EUR"),
    (20.0, "BRENT_CRUDE", "EUR"),
    (40.0, "WTI_CRUDE", "USD"),
)
BASE = (2_001, 2_002, 2_003, 2_004, 2_005, 2_006)
HOSTILE = (9_007, 12, 4_002, 101, 73, 555)
#: Node ids for the static config. The real `build_universe_mapping` lays FX
#: spots first, then scalar nodes: with these ids the run axis it produces is
#: [9007, 12, 101, 4002], genuinely unsorted and non-contiguous.
HOSTILE_RF = (
    ("USDEUR", 9_007),
    ("AAPL_EQ", 12),
    ("BRENT_CRUDE", 4_002),
    ("WTI_CRUDE", 101),
)
FULL_SLOTS = (1, 2, 3)  # Brent/EUR, WTI/USD, AAPL/EUR
DERIVED_SLOTS = (0, 4, 5)  # AAPL/USD, Brent/EUR, WTI/USD: every run-axis factor

ALL_UIDS = BASE
FULL_UIDS = tuple(BASE[i] for i in FULL_SLOTS)
DERIVED_UIDS = tuple(BASE[i] for i in DERIVED_SLOTS)
HISTORICAL = parse_scenario_model_config({"model": "historical", "horizon": 1})


def slots(uids: tuple[int, ...], which: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(uids[i] for i in which)


def positions(uids: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": list(uids),
        "instrument": ["SPOT"] * len(uids),
        "quantity": [e[0] for e in _ECONOMICS],
        "underlying": [e[1] for e in _ECONOMICS],
        "settlement_ccy": [e[2] for e in _ECONOMICS],
    })


@functools.cache
def _shared(rf=None) -> SimpleNamespace:
    if rf is None:
        rsc = FX._run_risk_static_config(verbose=False)
    else:
        sheets = dict(FX._RISK_STATIC_CONFIG_DICT_OF_DFS)
        key = next(k for k, v in sheets.items() if v is FX._NODE_META_DF)
        ids = dict(rf)
        node = FX._NODE_META_DF
        sheets[key] = node.with_columns(
            pl.Series("node_id", [ids[n] for n in node["node_name"]]).cast(
                node.schema["node_id"]
            )
        )
        rsc = FX.compile_risk_static_config_from_dict_of_dfs(sheets)
    return SimpleNamespace(
        rsc=rsc,
        terms=FX._run_proxy_terms(verbose=False),
        snapshot=FX._run_risk_factor_snapshot(rsc, verbose=False),
        historical=FX._build_historical_matrix(rsc),
        as_of=int(FX.date_to_ord32(FX._SNAPSHOT_AS_OF_DATE)),
    )


@functools.cache
def world(uids: tuple[int, ...] = BASE, rf=None) -> SimpleNamespace:
    return build_world(positions(uids), rf=rf, uids=uids)


def build_world(frame: pd.DataFrame, *, rf=None, uids=None) -> SimpleNamespace:
    """The production objects of one positions frame (ingest -> compile -> price)."""
    sh = _shared(rf)
    uids = tuple(int(u) for u in frame["uid"]) if uids is None else uids
    batch = FX.compile_raw_df_into_ingest_batch(
        raw_df=frame.copy(), static_config=sh.rsc
    )
    mapper = FX.build_uix_mapper([batch], as_of_ord32=sh.as_of)
    portfolio = FX.compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    universe_map = FX.build_universe_mapping(
        discounting_mode=FX.DiscountingMode.NONE,
        static_config=sh.rsc,
        uix_mapper=mapper,
    )
    su = FX._run_shock_universe(
        rsc=sh.rsc,
        proxy_terms=sh.terms,
        run_axis=derive_run_factor_axis(
            valuation_context=ValuationContext(
                static_config=sh.rsc,
                ingest_batch=batch,
                uix_mapper=mapper,
                discounting_mode=FX.DiscountingMode.NONE,
            ),
            reporting_ccy="USD",
        ),
        snapshot=sh.snapshot,
        verbose=False,
    )
    market_data = FX._run_market_data(
        rsc=sh.rsc, uix_mapper=mapper, snapshot=sh.snapshot
    )
    pricing = FX._run_pricing(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data, verbose=False
    )
    sens = FX.run_sensitivities(
        portfolio=portfolio, market_data=market_data, uix_mapper=mapper
    )
    rf = {name: int(rid) for name, rid in sh.rsc.rel_name_to_rf_id.items()}
    return SimpleNamespace(
        uids=uids,
        batch=batch,
        rsc=sh.rsc,
        mapper=mapper,
        portfolio=portfolio,
        universe_map=universe_map,
        su=su,
        pricing=pricing,
        sens=sens,
        rf=rf,
        as_of=sh.as_of,
        historical=sh.historical,
        frame=frame.set_index("uid"),
    )


def _row(w, name: str) -> int:
    return int(np.flatnonzero(w.su.required_factor_ids == w.rf[name])[0])


@functools.cache
def derived_block(
    book: tuple[int, ...],
    *,
    uids: tuple[int, ...] = BASE,
    rf=None,
    order: str = "truncated",
    as_of_shift: int = 0,
    axis: str = "run",
    tamper: tuple = (),
) -> TaylorBlock:
    """Certify the production sensitivities of `book` as TAYLOR/DERIVED.

    The book is built by the adapter on the declared axis (F-1, D082).
    """
    w = world(uids, rf)
    s = w.sens
    keep = np.isin(s.position_uids, book)
    keep_x = np.isin(s.cross_position_uids, book) & (order == "full")
    sub = dataclasses.replace(
        s,
        **{
            f.name: (
                getattr(s, f.name)[keep_x]
                if f.name.startswith("cross_")
                else getattr(s, f.name)[keep]
            )
            for f in dataclasses.fields(s)
            if isinstance(getattr(s, f.name), np.ndarray)
        },
    )
    book_uids = np.array(sorted(book), dtype=np.int64)
    pv = dict(
        zip(w.pricing.trade_uids.tolist(), w.pricing.values[:, 0].tolist(), strict=True)
    )
    v0 = np.array([pv[int(u)] for u in book_uids], dtype=np.float64)
    binding = FX.TaylorBaseBinding(
        as_of_ord32=w.as_of + as_of_shift,
        base_levels_digest=FX.base_levels_digest(
            factor_ids=w.su.required_factor_ids, base_levels=w.su.base_levels
        ),
        discounting_mode=FX.DiscountingMode.NONE.value,
        static_config_digest=FX.static_config_content_digest(w.rsc),
        valuation_contract_fingerprint=FX.valuation_contract_fingerprint(),
        base_settlement_values_digest=FX.base_settlement_values_digest(
            position_uids=book_uids, values=v0
        ),
    )
    # F-1 (D082): the adapter builds the book on the declared axis. `axis="run"`
    # is the production path; `axis="own"` declares the book's own referenced
    # factors explicitly (used only to prove the run refuses such a book).
    axis_ids = {
        "run": w.su.required_factor_ids,
        "run_sorted": np.sort(w.su.required_factor_ids),
        "own": sub.active_factor_rf_ids,
    }[axis]
    spec = FX.build_sensitivity_book_spec(
        sensitivities=sub, required_factor_ids=axis_ids
    )
    declared = (
        FX.TaylorOrder.SECOND_ORDER_FULL
        if order == "full"
        else FX.TaylorOrder.SECOND_ORDER_TRUNCATED
    )
    declaration = FX.derive_taylor_declaration(
        sensitivities=sub, order=declared, base_binding=binding
    )
    if tamper:  # certify at a deliberately different base
        binding = dataclasses.replace(binding, **dict(tamper))
        declaration = dataclasses.replace(declaration, base_binding=binding)
    spec = dataclasses.replace(spec, taylor_declaration=declaration)
    return TaylorBlock(calculator=FX.SensitivityPnlCalculator.compile(spec))


def context(*, uids: tuple[int, ...] = BASE, rf=None) -> ValuationContext:
    """The run-owned base valuation context (P1-B, N1) of a world."""
    w = world(uids, rf)
    return ValuationContext(
        static_config=w.rsc,
        ingest_batch=w.batch,
        uix_mapper=w.mapper,
        discounting_mode=FX.DiscountingMode.NONE,
    )


def reporting(universe, *, uids: tuple[int, ...] = BASE, rf=None) -> ReportingSpec:
    """USD reporting; settlement currencies and FX rows are run-derived (D085)."""
    return ReportingSpec(reporting_ccy="USD")


def inputs(
    *,
    uids=BASE,
    rf=None,
    full=None,
    derived=None,
    universe=None,
    batch_size=4,
    data=None,
    order="truncated",
):
    """Run inputs for a partition of the six positions (U defaults to F ∪ D)."""
    w = world(uids, rf)
    full = slots(uids, FULL_SLOTS) if full is None else full
    derived = slots(uids, DERIVED_SLOTS) if derived is None else derived
    universe = tuple(set(full) | set(derived)) if universe is None else universe
    return dict(
        scenario_config=HISTORICAL,
        data=w.historical if data is None else data,
        required_factor_ids=w.su.required_factor_ids,
        required_shift_type_id=w.su.required_shift_type_id,
        proxy_terms=w.su.proxy_terms,
        base_levels=w.su.base_levels,
        methodology=MethodologyDeclaration(
            production_position_uids=list(universe),
            full_reval_uids=list(full),
            taylor_supplied_uids=[],
            taylor_derived_uids=list(derived),
        ),
        reporting=reporting(universe, uids=uids, rf=rf),
        confidence_level=2.0 / 3.0,
        batch_size=batch_size,
        valuation_context=context(uids=uids, rf=rf),
        taylor_derived=derived_block(
            tuple(sorted(derived)), uids=uids, rf=rf, order=order
        )
        if derived
        else None,
    )


def history(*, move: set[str]) -> RiskFactorMatrix:
    """Historical data where only the named source factors move (others flat)."""
    w = world()
    m = w.historical
    name_of = {v: k for k, v in w.rf.items()}
    values = np.array(m.values, copy=True)
    for j, rid in enumerate(m.factor_ids.tolist()):
        if name_of[int(rid)] not in move:
            values[:, j] = values[-1, j]
    return RiskFactorMatrix(
        dates=m.dates, factor_ids=m.factor_ids, values=np.asfortranarray(values)
    )


def shocked_levels(data=None, *, uids=BASE, rf=None) -> np.ndarray:
    """
    Canonical (K, S) scenario levels, rebuilt from the D088 scenario seam in one
    batch: the fixture's authored terms compiled over the run axis, lowered,
    bound to the world's (d, tau, h) columns, streamed, then X = sum beta Y and
    T^-1_kappa. The tests compare the run's one shocked-level buffer with this
    rebuild bitwise (plumbing, not an independent numerical oracle).
    """
    w = world(uids, rf)
    m = w.historical if data is None else data
    operator = lower_axis_operator(
        model=compile_proxy_model(
            authored_terms=w.su.proxy_terms,
            static_config=w.rsc,
            as_of_date=ord_to_date(int(w.as_of)),
            run_axis_rf_ids=w.su.required_factor_ids,
        ),
        axis_rf_ids=w.su.required_factor_ids,
        axis_shift_type_ids=w.su.required_shift_type_id,
    )
    binding = bind_operator_to_world(operator, HISTORICAL, m)
    gen = build_scenario_generator(HISTORICAL, data=m)
    k, s = w.su.required_factor_ids.shape[0], gen.total_scenarios
    out = np.empty((k, s), dtype=np.float64)
    for b in gen.iter_batches(driver_rf_ids=binding.request_driver_rf_ids,
        chart_ids=binding.request_chart_ids,
        batch_size=s,
    ):
        chunk = np.empty((k, b.shocks.shape[0]), dtype=np.float64)
        apply_linear_terms_to_base_levels(
            base_levels=w.su.base_levels,
            row_term_offsets=operator.row_term_offsets,
            term_column_uix=binding.term_column_uix,
            term_loadings=operator.term_loadings,
            target_shift_types=operator.target_shift_type_ids,
            shocks=b.shocks,
            out=chunk,
        )
        out[:, b.scenario_offset : b.scenario_offset + chunk.shape[1]] = chunk
    return out


def _terms(uid: int, levels: np.ndarray, uids=BASE, rf=None):
    w = world(uids, rf)
    r = w.frame.loc[uid]
    price = levels[_row(w, r["underlying"])]
    fx = (
        levels[_row(w, "USDEUR")]
        if r["settlement_ccy"] == "EUR"
        else np.ones_like(price)
    )
    return float(r["quantity"]), price, fx, r["settlement_ccy"]


def l1_value(uid: int, levels: np.ndarray, uids=BASE, rf=None) -> np.ndarray:
    qty, price, fx, _ = _terms(uid, levels, uids, rf)
    return qty * price * fx


def l4_taylor(
    uid: int, base: np.ndarray, shocked: np.ndarray, *, cross: bool, uids=BASE, rf=None
) -> np.ndarray:
    """First order, zero diagonals, and the P-X cross term only if declared."""
    qty, p0, x0, _ = _terms(uid, base[:, None], uids, rf)
    _, ps, xs, _ = _terms(uid, shocked, uids, rf)
    dp, dx = ps - p0, xs - x0
    return qty * (x0 * dp + p0 * dx) + (qty * dp * dx if cross else 0.0)


def reporting_rate(uid: int, levels: np.ndarray, uids=BASE, rf=None) -> np.ndarray:
    _, _, _, ccy = _terms(uid, levels, uids, rf)
    usdeur = levels[_row(world(uids, rf), "USDEUR")]
    return np.ones_like(usdeur) if ccy == "USD" else 1.0 / usdeur


def l0_reporting(v0: float, dv: np.ndarray, q0: float, qs: np.ndarray) -> np.ndarray:
    return (v0 + dv) * qs - v0 * q0
