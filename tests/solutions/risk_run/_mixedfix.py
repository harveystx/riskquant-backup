# tests/solutions/risk_run/_mixedfix.py
"""
Mixed-methodology fixture on the REAL production path.

Six SPOT positions are ingested, compiled and valued through the same public
functions the D019 demo uses (static config, proxy topology, snapshot,
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

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.risk_run.schemas import (
    FullRevalBlock,
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
)
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.sensitivity_pnl.api import FactorAxis

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
def _shared() -> SimpleNamespace:
    rsc = FX._run_risk_static_config(verbose=False)
    return SimpleNamespace(
        rsc=rsc,
        topo=FX._run_proxy_topology(rsc, verbose=False),
        snapshot=FX._run_risk_factor_snapshot(rsc, verbose=False),
        historical=FX._build_historical_matrix(rsc),
        as_of=int(FX.date_to_ord32(FX._SNAPSHOT_AS_OF_DATE)),
    )


@functools.cache
def world(uids: tuple[int, ...] = BASE) -> SimpleNamespace:
    sh = _shared()
    batch = FX.compile_raw_df_into_ingest_batch(
        raw_df=positions(uids), static_config=sh.rsc
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
        proxy_topology=sh.topo,
        uix_mapper=mapper,
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
        frame=positions(uids).set_index("uid"),
    )


def _row(w, name: str) -> int:
    return int(np.flatnonzero(w.su.required_factor_ids == w.rf[name])[0])


@functools.cache
def derived_block(
    book: tuple[int, ...],
    *,
    uids: tuple[int, ...] = BASE,
    order: str = "truncated",
    as_of_shift: int = 0,
    on_run_axis: bool = False,
) -> TaylorBlock:
    """Certify the production sensitivities of `book` as TAYLOR/DERIVED.

    `on_run_axis=True` recompiles the same certified spec on the full run
    axis (inactive factors included): the documented F-1 construction path.
    """
    w = world(uids)
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
    spec = FX.build_sensitivity_book_spec(sensitivities=sub)
    declared = (
        FX.TaylorOrder.SECOND_ORDER_FULL
        if order == "full"
        else FX.TaylorOrder.SECOND_ORDER_TRUNCATED
    )
    declaration = FX.derive_taylor_declaration(
        sensitivities=sub, order=declared, base_binding=binding
    )
    spec = dataclasses.replace(spec, taylor_declaration=declaration)
    if on_run_axis:
        ids = w.su.required_factor_ids.tolist()
        spec = dataclasses.replace(
            spec,
            factor_axis=FactorAxis.from_keys(
                pl.Series("factor_id", ids, dtype=pl.Int32)
            ),
            factor_meta=pl.DataFrame({
                "factor_id": pl.Series(ids, dtype=pl.Int32),
                "displacement_convention_id": pl.Series([1] * len(ids), dtype=pl.Int8),
                "quote_base": pl.Series([1.0] * len(ids), dtype=pl.Float64),
            }),
        )
    calc = FX.SensitivityPnlCalculator.compile(spec)
    return TaylorBlock(
        calculator=calc, base_settlement_values=v0, expected_binding=binding
    )


def full_block(portfolio=None, *, uids: tuple[int, ...] = BASE) -> FullRevalBlock:
    w = world(uids)
    return FullRevalBlock(
        portfolio=w.portfolio if portfolio is None else portfolio,
        uix_mapper=w.mapper,
        universe_map=w.universe_map,
    )


def reporting(universe, *, uids: tuple[int, ...] = BASE) -> ReportingSpec:
    w = world(uids)
    ordered = np.array(sorted(universe), dtype=np.int64)
    ccy = {int(u): w.frame.loc[u, "settlement_ccy"] for u in w.uids}
    return ReportingSpec(
        reporting_ccy="USD",
        position_uids=ordered,
        settlement_ccys=tuple(ccy.get(int(u), "USD") for u in ordered),
        fx_factor_names=("USDEUR",),
        fx_factor_rows=np.array([_row(w, "USDEUR")], dtype=np.int32),
    )


def inputs(
    *,
    uids=BASE,
    full=None,
    derived=None,
    universe=None,
    batch_size=4,
    data=None,
    order="truncated",
):
    """Run inputs for a partition of the six positions (U defaults to F ∪ D)."""
    w = world(uids)
    full = slots(uids, FULL_SLOTS) if full is None else full
    derived = slots(uids, DERIVED_SLOTS) if derived is None else derived
    universe = tuple(set(full) | set(derived)) if universe is None else universe
    return dict(
        scenario_config=HISTORICAL,
        data=w.historical if data is None else data,
        required_factor_ids=w.su.required_factor_ids,
        required_shift_type_id=w.su.required_shift_type_id,
        source_factor_ids=w.su.source_factor_ids,
        shock_scale=w.su.shock_scale,
        base_levels=w.su.base_levels,
        methodology=MethodologyDeclaration(
            production_position_uids=list(universe),
            full_reval_uids=list(full),
            taylor_supplied_uids=[],
            taylor_derived_uids=list(derived),
        ),
        reporting=reporting(universe, uids=uids),
        confidence_level=2.0 / 3.0,
        batch_size=batch_size,
        full_reval=full_block(uids=uids) if full else None,
        taylor_derived=derived_block(tuple(sorted(derived)), uids=uids, order=order)
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


def shocked_levels(data=None) -> np.ndarray:
    """Canonical (K, S) scenario levels, rebuilt from the scenario seam in one batch."""
    w = world()
    gen = build_scenario_generator(
        HISTORICAL, data=w.historical if data is None else data
    )
    k, s = w.su.required_factor_ids.shape[0], gen.total_scenarios
    out = np.empty((k, s), dtype=np.float64)
    for b in gen.iter_batches(
        required_shift_type_id=w.su.required_shift_type_id,
        source_factor_ids=w.su.source_factor_ids,
        shock_scale=w.su.shock_scale,
        batch_size=s,
    ):
        chunk = np.empty((k, b.shocks.shape[0]), dtype=np.float64)
        apply_shocks_to_base_levels(base_levels=w.su.base_levels, batch=b, out=chunk)
        out[:, b.scenario_offset : b.scenario_offset + chunk.shape[1]] = chunk
    return out


def _terms(uid: int, levels: np.ndarray, uids=BASE):
    w = world(uids)
    r = w.frame.loc[uid]
    price = levels[_row(w, r["underlying"])]
    fx = (
        levels[_row(w, "USDEUR")]
        if r["settlement_ccy"] == "EUR"
        else np.ones_like(price)
    )
    return float(r["quantity"]), price, fx, r["settlement_ccy"]


def l1_value(uid: int, levels: np.ndarray, uids=BASE) -> np.ndarray:
    qty, price, fx, _ = _terms(uid, levels, uids)
    return qty * price * fx


def l4_taylor(
    uid: int, base: np.ndarray, shocked: np.ndarray, *, cross: bool, uids=BASE
) -> np.ndarray:
    """First order, zero diagonals, and the P-X cross term only if declared."""
    qty, p0, x0, _ = _terms(uid, base[:, None], uids)
    _, ps, xs, _ = _terms(uid, shocked, uids)
    dp, dx = ps - p0, xs - x0
    return qty * (x0 * dp + p0 * dx) + (qty * dp * dx if cross else 0.0)


def reporting_rate(uid: int, levels: np.ndarray, uids=BASE) -> np.ndarray:
    _, _, _, ccy = _terms(uid, levels, uids)
    usdeur = levels[_row(world(uids), "USDEUR")]
    return np.ones_like(usdeur) if ccy == "USD" else 1.0 / usdeur


def l0_reporting(v0: float, dv: np.ndarray, q0: float, qs: np.ndarray) -> np.ndarray:
    return (v0 + dv) * qs - v0 * q0
