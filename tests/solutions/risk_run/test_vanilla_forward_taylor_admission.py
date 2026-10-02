# tests/solutions/risk_run/test_vanilla_forward_taylor_admission.py
"""
D102 (WS-A): the source-specific admission boundary for VANILLA_FORWARD_NODE.

    D102.3  every FORWARD-family ingest row of a UID that owns a
            `vanilla_forward` row is VANILLA_FORWARD_NODE, else refused
            (every methodology partition; checked first)
    D102.4  such a UID is admitted through FULL_REVAL only (until D103)

Both refusals happen in `_require_forward_capability`, directly after the D084
SPOT check and BEFORE `_row_aggregation`. Spy set S records
`build_scenario_generator`, `allocate_pnl_grid`,
`build_market_data_matrix_from_flat_levels` and `_row_aggregation`; "no
downstream work" means zero calls in S.

Book (hostile static configuration of the market-data remediation suite; UIDs
sparse and not ascending; the forward rows are not adjacent; rf ids sparse):

    9007  SPOT    EQ_USD           q=+100  settle EUR  direct FX (USDEUR)
    4002  FORWARD BRENT_FWD[M+0]   q=+100  settle USD  K=80   rf 2001
    12    SPOT    EQ_USD           q=-40   settle USD  no FX
    555   FORWARD BRENT_FWD[M+1]   q=-40   settle EUR  K=81   rf 2002

The curve-point rf ids 2001 and 2002 are in `GlobalIdNamespace.NODES`, the same
namespace as scalar node ids, so the namespace cannot identify the source. P1
changes only `IngestBatch.instrument_ids` and P2 appends one CURVE row; both keep
the compiled `VanillaForwardBlock`, so the block type cannot identify the source.
Multi-row UIDs (D055) are built by concatenating production ingest batches:
`IngestBatch` permits repeated UIDs; only raw-frame ingest rejects them.

A1, A2 and P1-D use a caller-built TAYLOR/DERIVED book for UID 4002 with a
deliberately wrong first-order coefficient (2*Q; the exact NONE value is Q).
P1-A certifies base, order and completeness, not Greek values, so the book is
certified. Without the D102.4 guard it is priced (A2, D059).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

_REMEDIATION = Path(__file__).resolve().parents[2] / "market_data" / "remediation"
sys.path.insert(0, str(_REMEDIATION))
import hostile_fixture as hf

import riskquant.solutions.risk_run.api as api
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.namespace import GlobalIdNamespace
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
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.portfolio.instruments import (
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
)
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
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
from riskquant.valuation.blocks.block_type import BlockType
from riskquant.valuation.engine import run_valuation

SPOT_EUR, FWD_USD, SPOT_USD, FWD_EUR = 9_007, 4_002, 12, 555
FWD_RF = {FWD_USD: 2001, FWD_EUR: 2002}
_NONE = DiscountingMode.NONE
_HISTORY_DAYS = 80
#: Base levels by NAME; the forward levels extend the hostile fixture's market.
_BASE_LEVELS = {**hf._BASE, "BRENT_FWD[M+0]": 81.37, "BRENT_FWD[M+1]": 80.64}


def _spot(uid: int, und: str, settle: str, qty: float) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "SPOT",
        "quantity": qty,
        "underlying": und,
        "settlement_ccy": settle,
    }


def _fwd(
    uid: int, und: str, settle: str, qty: float, strike: float, delivery: dt.date
) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "VANILLA_FORWARD_NODE",
        "quantity": qty,
        "underlying": und,
        "settlement_ccy": settle,
        "strike_price": strike,
        "strike_ccy": "USD",
        "delivery_date": delivery,
        "settlement_date": delivery + dt.timedelta(days=2),
        "under_discount_curve": "USD_OIS",
        "strike_discount_curve": "USD_OIS",
    }


#: Row order is the ingest order: UIDs not ascending, forward rows not adjacent.
ROWS = [
    _spot(SPOT_EUR, "EQ_USD", "EUR", 100.0),
    _fwd(FWD_USD, "BRENT_FWD[M+0]", "USD", 100.0, 80.0, dt.date(2027, 2, 15)),
    _spot(SPOT_USD, "EQ_USD", "USD", -40.0),
    _fwd(FWD_EUR, "BRENT_FWD[M+1]", "EUR", -40.0, 81.0, dt.date(2027, 3, 15)),
]


def _ingest(rsc: Any, rows: list[dict[str, Any]]) -> IngestBatch:
    """Production raw-frame ingest (dispatcher, identity gate, unique UIDs)."""
    return compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(pd.DataFrame(rows)),
        static_config=rsc,
        as_of_date=hf.AS_OF,
    )


def _concat(a: IngestBatch, b: IngestBatch) -> IngestBatch:
    """Append the rows of `b` to `a`: the run boundary accepts repeated UIDs."""
    assert a.as_of_ord32 == b.as_of_ord32
    arrays = {
        f.name: np.ascontiguousarray(
            np.concatenate((getattr(a, f.name), getattr(b, f.name)))
        )
        for f in dataclasses.fields(IngestBatch)
        if isinstance(getattr(a, f.name), np.ndarray)
    }
    return dataclasses.replace(a, size=a.size + b.size, **arrays)


def _relabel(batch: IngestBatch, uid: int, instrument_id: int) -> IngestBatch:
    """Set the instrument id of every row of `uid`; nothing else changes."""
    ids = np.array(batch.instrument_ids, copy=True)
    ids[np.asarray(batch.uids) == uid] = instrument_id
    return dataclasses.replace(batch, instrument_ids=np.ascontiguousarray(ids))


def _context(w: SimpleNamespace, batch: IngestBatch) -> ValuationContext:
    """A fresh context compiled by the canonical compiler from `batch`."""
    mapper = build_uix_mapper([batch], as_of_ord32=w.as_of)
    return ValuationContext(
        static_config=w.rsc,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=_NONE,
    )


def _with_batch(w: SimpleNamespace, batch: IngestBatch) -> ValuationContext:
    """A valid NODE context whose ingest metadata is replaced AFTER compilation."""
    context = _context(w, w.batch)
    object.__setattr__(context, "ingest_batch", batch)
    return context


@pytest.fixture(scope="module")
def world() -> SimpleNamespace:
    rsc = compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())
    as_of = int(date_to_ord32(hf.AS_OF))
    batch = _ingest(rsc, ROWS)
    w = SimpleNamespace(rsc=rsc, as_of=as_of, batch=batch)
    context = _context(w, batch)
    layout = build_universe_mapping(
        static_config=rsc, uix_mapper=context.uix_mapper, discounting_mode=_NONE
    )
    name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    fids = np.asarray(layout.required_factor_ids, dtype=np.int32)
    base = np.array([_BASE_LEVELS[name[int(f)]] for f in fids], dtype=np.float64)
    rng = np.random.default_rng(11)
    values = np.asfortranarray(
        base
        * np.exp(np.cumsum(rng.normal(0.0, 0.01, (_HISTORY_DAYS, fids.size)), axis=0))
    )
    w.context = context
    w.layout = layout
    w.fids = fids
    w.base = base
    w.data = RiskFactorMatrix(
        dates=np.arange(as_of - _HISTORY_DAYS + 1, as_of + 1, dtype=np.int32),
        factor_ids=fids.copy(),
        values=values,
    )
    w.uids = np.array(sorted(r["uid"] for r in ROWS), dtype=np.int64)
    w.reporting = ReportingSpec(reporting_ccy="USD")
    return w


def _sub_portfolio(w: SimpleNamespace, uids: list[int]) -> SimpleNamespace:
    """The production objects of a UID subset, at the run's base levels."""
    batch = _ingest(w.rsc, [r for r in ROWS if r["uid"] in uids])
    mapper = build_uix_mapper([batch], as_of_ord32=w.as_of)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=w.rsc, uix_mapper=mapper, discounting_mode=_NONE
    )
    row = {int(f): i for i, f in enumerate(w.fids)}
    levels = w.base[[row[int(f)] for f in layout.required_factor_ids]]
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=_NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=np.ascontiguousarray(levels[:, None]),
        num_scenarios=1,
    )
    values = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(matrix=matrix, uix_mapper=mapper),
        out=values,
    )
    v0 = dict(zip(portfolio.trade_uids.tolist(), values[:, 0].tolist(), strict=True))
    return SimpleNamespace(portfolio=portfolio, mapper=mapper, matrix=matrix, v0=v0)


def _certified_block(
    w: SimpleNamespace, sens: BlockSensitivities, v0: dict[int, float]
) -> TaylorBlock:
    """Certify `sens` as TAYLOR/DERIVED (SECOND_ORDER_FULL) at the run's base."""
    book = np.unique(np.asarray(sens.position_uids, dtype=np.int64))
    binding = TaylorBaseBinding(
        as_of_ord32=w.as_of,
        base_levels_digest=base_levels_digest(factor_ids=w.fids, base_levels=w.base),
        discounting_mode=_NONE.value,
        static_config_digest=static_config_content_digest(w.rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book, values=np.array([v0[int(u)] for u in book])
        ),
    )
    spec = build_sensitivity_book_spec(sensitivities=sens, required_factor_ids=w.fids)
    spec = dataclasses.replace(
        spec,
        taylor_declaration=derive_taylor_declaration(
            sensitivities=sens,
            order=TaylorOrder.SECOND_ORDER_FULL,
            base_binding=binding,
        ),
    )
    return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


def _caller_built_forward_block(w: SimpleNamespace) -> TaylorBlock:
    """
    Caller-built book for UID 4002 ({F} topology, USD settlement, NONE). The
    coefficient is 2*Q; the exact value is Q*DF_a = Q. P1-A certifies it.
    """
    qty = next(r["quantity"] for r in ROWS if r["uid"] == FWD_USD)
    sens = BlockSensitivities(
        position_uids=np.array([FWD_USD], dtype=np.int64),
        factor_rf_ids=np.array([FWD_RF[FWD_USD]], dtype=np.int32),
        first_order=np.array([2.0 * qty], dtype=np.float64),
        second_order=np.zeros(1, dtype=np.float64),
        cross_position_uids=np.empty(0, dtype=np.int64),
        cross_factor_rf_ids_i=np.empty(0, dtype=np.int32),
        cross_factor_rf_ids_j=np.empty(0, dtype=np.int32),
        cross_second_order=np.empty(0, dtype=np.float64),
    )
    return _certified_block(w, sens, _sub_portfolio(w, [FWD_USD]).v0)


def _spot_block(w: SimpleNamespace, derived: list[int]) -> TaylorBlock:
    """D084.4: production SPOT sensitivities from a SPOT-only sub-portfolio."""
    sub = _sub_portfolio(w, derived)
    assert sub.portfolio.vanilla_forward is None
    sens = run_sensitivities(
        portfolio=sub.portfolio, uix_mapper=sub.mapper, market_data=sub.matrix
    )
    return _certified_block(w, sens, sub.v0)


def _inputs(
    w: SimpleNamespace,
    *,
    context: ValuationContext | None = None,
    derived: tuple[int, ...] = (),
    supplied: tuple[int, ...] = (),
    block: TaylorBlock | None = None,
) -> dict[str, Any]:
    universe = w.uids
    u_d = np.array(sorted(derived), dtype=np.int64)
    u_s = np.array(sorted(supplied), dtype=np.int64)
    return {
        "scenario_config": parse_scenario_model_config({
            "model": "historical",
            "horizon": 1,
        }),
        "data": w.data,
        "required_factor_ids": w.fids,
        "required_shift_type_id": np.asarray(
            w.layout.required_shift_type_ids, dtype=np.int8
        ),
        "proxy_terms": (),
        "base_levels": w.base,
        "methodology": MethodologyDeclaration(
            production_position_uids=universe,
            full_reval_uids=np.setdiff1d(universe, np.union1d(u_d, u_s)),
            taylor_supplied_uids=u_s,
            taylor_derived_uids=u_d,
        ),
        "valuation_context": w.context if context is None else context,
        "reporting": w.reporting,
        "confidence_level": 0.99,
        "batch_size": 16,
        "taylor_derived": block,
    }


@pytest.fixture
def spy(monkeypatch):
    """Spy set S: count downstream stages (calls through to production)."""
    calls = {
        name: 0
        for name in (
            "build_scenario_generator",
            "allocate_pnl_grid",
            "build_market_data_matrix_from_flat_levels",
            "_row_aggregation",
        )
    }

    def _counting(name: str):
        real = getattr(api, name)

        def _wrapped(*args: object, **kwargs: object) -> object:
            calls[name] += 1
            return real(*args, **kwargs)

        return _wrapped

    for name in tuple(calls):
        monkeypatch.setattr(api, name, _counting(name))
    return calls


def _no_downstream_work(calls: dict[str, int]) -> None:
    assert calls == dict.fromkeys(calls, 0), calls


def _position_var(run: Any, uid: int) -> float:
    return float(run.var[int(np.searchsorted(run.position_uids, uid))])


@pytest.fixture(scope="module")
def all_full(world: SimpleNamespace):
    return run_scenario_var_es(**_inputs(world))


# ---- fixture guards ---------------------------------------------------------------


def test_fixture_is_hostile_and_node_sourced(world: SimpleNamespace) -> None:
    batch = world.batch
    forward_rows = np.flatnonzero(batch.instrument_ids == VANILLA_FORWARD_NODE_ID)
    assert batch.uids.tolist() == [SPOT_EUR, FWD_USD, SPOT_USD, FWD_EUR]
    assert forward_rows.tolist() == [1, 3]  # not adjacent
    block = world.context.portfolio.vanilla_forward
    assert sorted(block.trade_uids.tolist()) == sorted(FWD_RF)
    rows = {
        int(u): int(r) for u, r in zip(batch.uids, batch.underlying_ids, strict=True)
    }
    assert {u: rows[u] for u in FWD_RF} == FWD_RF
    assert len(set(np.diff(np.sort(world.fids)).tolist())) > 1  # sparse rf ids


# ---- A-tests ----------------------------------------------------------------------


def test_a1_forward_uid_declared_derived_is_refused_before_downstream_work(
    world: SimpleNamespace, spy
) -> None:
    with pytest.raises(CompositionContractError, match=r"^D102\.4: .*\[4002\]"):
        run_scenario_var_es(
            **_inputs(
                world, derived=(FWD_USD,), block=_caller_built_forward_block(world)
            )
        )
    _no_downstream_work(spy)


def test_a2_without_the_guard_the_caller_built_book_is_priced(
    world: SimpleNamespace, all_full, monkeypatch
) -> None:
    """D059: the D102.4 guard is the only barrier; without it the certified,
    wrong book is priced and moves the forward position's VaR."""
    monkeypatch.setattr(api, "_require_forward_capability", lambda **kw: None)
    run = run_scenario_var_es(
        **_inputs(world, derived=(FWD_USD,), block=_caller_built_forward_block(world))
    )
    assert run.pnl_provenance is not None  # the Taylor block really priced
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()
    assert _position_var(run, FWD_USD) != _position_var(all_full, FWD_USD)


def test_a3_node_full_reval_only_run_is_admitted(world: SimpleNamespace, spy) -> None:
    run = run_scenario_var_es(**_inputs(world))
    assert spy["_row_aggregation"] == 1
    assert run.position_uids.tolist() == world.uids.tolist()
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()


def test_a4_forward_full_reval_beside_spot_derived_is_admitted(
    world: SimpleNamespace, spy
) -> None:
    run = run_scenario_var_es(
        **_inputs(world, derived=(SPOT_EUR,), block=_spot_block(world, [SPOT_EUR]))
    )
    assert spy["_row_aggregation"] == 1
    assert run.pnl_provenance is not None
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()


def _spot_only_world(w: SimpleNamespace) -> SimpleNamespace:
    """The SPOT rows alone, on their own run axis (history rows taken by id)."""
    batch = _ingest(w.rsc, [r for r in ROWS if r["instrument"] == "SPOT"])
    context = _context(w, batch)
    layout = build_universe_mapping(
        static_config=w.rsc, uix_mapper=context.uix_mapper, discounting_mode=_NONE
    )
    fids = np.asarray(layout.required_factor_ids, dtype=np.int32)
    col = [int(np.flatnonzero(w.fids == f)[0]) for f in fids]
    return SimpleNamespace(
        rsc=w.rsc,
        as_of=w.as_of,
        batch=batch,
        context=context,
        layout=layout,
        fids=fids,
        base=w.base[col],
        data=RiskFactorMatrix(
            dates=w.data.dates.copy(),
            factor_ids=fids.copy(),
            values=np.asfortranarray(w.data.values[:, col]),
        ),
        uids=np.array(sorted((SPOT_EUR, SPOT_USD)), dtype=np.int64),
        reporting=w.reporting,
    )


def test_a5_spot_only_run_is_admitted(world: SimpleNamespace, spy) -> None:
    sw = _spot_only_world(world)
    assert sw.context.portfolio.vanilla_forward is None
    run = run_scenario_var_es(
        **_inputs(sw, derived=(SPOT_EUR,), block=_spot_block(sw, [SPOT_EUR]))
    )
    assert spy["_row_aggregation"] == 1
    assert run.pnl_provenance is not None
    assert run.position_uids.tolist() == sw.uids.tolist()


def test_a1s_forward_uid_declared_supplied_keeps_the_p1_refusal(
    world: SimpleNamespace, spy
) -> None:
    """The global P-1 refusal is unchanged and precedes D102."""
    with pytest.raises(CompositionContractError, match=r"^TAYLOR/SUPPLIED is not"):
        run_scenario_var_es(**_inputs(world, supplied=(FWD_USD,)))
    _no_downstream_work(spy)


def test_d102_4_covers_the_supplied_partition_independently_of_p1(
    world: SimpleNamespace,
) -> None:
    methodology = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.setdiff1d(world.uids, [FWD_EUR]),
        taylor_supplied_uids=np.array([FWD_EUR], dtype=np.int64),
        taylor_derived_uids=np.empty(0, dtype=np.int64),
    )
    with pytest.raises(CompositionContractError, match=r"^D102\.4: .*\[555\]"):
        api._require_forward_capability(context=world.context, methodology=methodology)


# ---- P-tests: provenance ------------------------------------------------------------


def test_p1f_relabelled_curve_source_is_refused_in_full_reval(
    world: SimpleNamespace, spy
) -> None:
    context = _with_batch(
        world, _relabel(world.batch, FWD_USD, VANILLA_FORWARD_CURVE_ID)
    )
    block = context.portfolio.vanilla_forward
    assert int(BlockType.VANILLA_FORWARD) == block.BLOCK_TYPE_ID
    assert FWD_USD in block.trade_uids.tolist()
    nodes = GlobalIdNamespace.NODES
    assert nodes.min_id <= FWD_RF[FWD_USD] <= nodes.max_id
    with pytest.raises(CompositionContractError, match=r"^D102\.3: .*\[4002\]"):
        run_scenario_var_es(**_inputs(world, context=context))
    _no_downstream_work(spy)


def test_p1d_provenance_is_checked_before_the_methodology(
    world: SimpleNamespace, spy
) -> None:
    context = _with_batch(
        world, _relabel(world.batch, FWD_USD, VANILLA_FORWARD_CURVE_ID)
    )
    with pytest.raises(CompositionContractError, match=r"^D102\.3: "):
        run_scenario_var_es(
            **_inputs(
                world,
                context=context,
                derived=(FWD_USD,),
                block=_caller_built_forward_block(world),
            )
        )
    _no_downstream_work(spy)


def _mixed_source_batch(w: SimpleNamespace) -> IngestBatch:
    """UID 4002 gains a second forward row whose instrument is CURVE."""
    extra = _ingest(w.rsc, [r for r in ROWS if r["uid"] == FWD_USD])
    return _concat(w.batch, _relabel(extra, FWD_USD, VANILLA_FORWARD_CURVE_ID))


def test_p2_mixed_forward_sources_under_one_uid_are_refused(
    world: SimpleNamespace, spy
) -> None:
    batch = _mixed_source_batch(world)
    rows = np.flatnonzero(batch.uids == FWD_USD)
    assert batch.instrument_ids[rows].tolist() == [
        VANILLA_FORWARD_NODE_ID,
        VANILLA_FORWARD_CURVE_ID,
    ]
    with pytest.raises(CompositionContractError, match=r"^D102\.3: .*\[4002\]"):
        run_scenario_var_es(**_inputs(world, context=_with_batch(world, batch)))
    _no_downstream_work(spy)


def test_p3_homogeneous_multi_row_uid_is_admitted(world: SimpleNamespace, spy) -> None:
    extra = _ingest(
        world.rsc,
        [{**r, "quantity": 25.0} for r in ROWS if r["uid"] == FWD_USD],
    )
    batch = _concat(world.batch, extra)
    context = _context(world, batch)
    block = context.portfolio.vanilla_forward
    assert block.trade_uids.tolist().count(FWD_USD) == 2
    rows = np.flatnonzero(batch.uids == FWD_USD)
    assert batch.instrument_ids[rows].tolist() == [VANILLA_FORWARD_NODE_ID] * 2
    run = run_scenario_var_es(**_inputs(world, context=context))
    assert spy["_row_aggregation"] == 1
    assert run.position_uids.tolist() == world.uids.tolist()
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()


def test_p4_non_forward_row_of_a_forward_uid_is_not_a_source_row(
    world: SimpleNamespace, spy
) -> None:
    """D102.3 inspects the FORWARD-family rows only: a SPOT row under the same
    UID (D055) is not a forward source and does not refuse FULL_REVAL."""
    extra = _ingest(world.rsc, [_spot(FWD_USD, "EQ_USD", "USD", 7.0)])
    context = _context(world, _concat(world.batch, extra))
    assert FWD_USD in context.portfolio.spot.trade_uids.tolist()
    assert FWD_USD in context.portfolio.vanilla_forward.trade_uids.tolist()
    run = run_scenario_var_es(**_inputs(world, context=context))
    assert spy["_row_aggregation"] == 1
    assert run.position_uids.tolist() == world.uids.tolist()


@pytest.mark.parametrize("case", ["p1f", "p2"])
def test_provenance_guard_is_the_barrier_before_row_aggregation(
    world: SimpleNamespace, spy, monkeypatch, case: str
) -> None:
    """D059: without the guard the corrupted context reaches `_row_aggregation`."""
    batch = (
        _relabel(world.batch, FWD_USD, VANILLA_FORWARD_CURVE_ID)
        if case == "p1f"
        else _mixed_source_batch(world)
    )
    monkeypatch.setattr(api, "_require_forward_capability", lambda **kw: None)
    with pytest.raises(CompositionContractError):
        run_scenario_var_es(**_inputs(world, context=_with_batch(world, batch)))
    assert spy["_row_aggregation"] == 1
