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

import copy
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

import riskquant.sensitivities.engine as sensitivity_engine
import riskquant.solutions.risk_run.api as api
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.namespace import GlobalIdNamespace
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import INVALID_UIX
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
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
)
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivity_pnl.compile import DuplicatePolicy
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
    TaylorContractError,
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
    w: SimpleNamespace,
    sens: BlockSensitivities,
    v0: dict[int, float],
    *,
    order: TaylorOrder = TaylorOrder.SECOND_ORDER_FULL,
    policy: DuplicatePolicy = DuplicatePolicy.RAISE,
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
            order=order,
            base_binding=binding,
        ),
        duplicate_policy=policy,
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
            "_value_positions",
            "_certify_taylor_block",
            "convert_scenario_pnl_to_reporting_currency",
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
    # Taylor P&L and sensitivity generation (the run must never reach either
    # before an admission refusal; the run never calls the engine at all).
    calls["run_into_grid"] = 0
    real_grid = SensitivityPnlCalculator.run_into_grid

    def _grid(self: Any, *args: object, **kwargs: object) -> object:
        calls["run_into_grid"] += 1
        return real_grid(self, *args, **kwargs)

    monkeypatch.setattr(SensitivityPnlCalculator, "run_into_grid", _grid)
    calls["resolve_sensitizer"] = 0
    real_resolve = sensitivity_engine.resolve_sensitizer

    def _resolve(*args: object, **kwargs: object) -> object:
        calls["resolve_sensitizer"] += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(sensitivity_engine, "resolve_sensitizer", _resolve)
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


def test_a1_d103_supersedes_d102_4_derived_forward_is_admitted(
    world: SimpleNamespace, spy
) -> None:
    """History: under WS-A (D102.4) this run was refused with "D102.4" before
    `_row_aggregation`. D103.4 (WS-D) admits a TAYLOR/DERIVED forward UID in
    DiscountingMode.NONE with non-inverted FX legs. The caller-built book is
    certified by P1-A (base, order, completeness) and priced."""
    run = run_scenario_var_es(
        **_inputs(world, derived=(FWD_USD,), block=_caller_built_forward_block(world))
    )
    assert spy["_row_aggregation"] == 1 and spy["run_into_grid"] >= 1
    assert run.pnl_provenance is not None
    assert spy["resolve_sensitizer"] == 0  # the run never generates sensitivities


def test_a2_d103_supersedes_the_d102_4_guard_p1a_does_not_check_greek_values(
    world: SimpleNamespace, all_full
) -> None:
    """History: under WS-A this was the D059 proof that only the D102.4 guard
    stopped a certified, wrong (2*Q) book. After D103.4 the run admits it: P1-A
    certifies base, order and completeness, not Greek values, so the wrong book
    is priced and moves the VaR. Correct forward numbers require the production
    sensitizer (D103.1); `test_vanilla_forward_node_production.py` proves them."""
    run = run_scenario_var_es(
        **_inputs(world, derived=(FWD_USD,), block=_caller_built_forward_block(world))
    )
    assert run.pnl_provenance is not None
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
    with pytest.raises(
        CompositionContractError, match=r"^D102\.4: .*\[555\]\)\. .*\(D103\.5\)"
    ) as refusal:
        api._require_forward_capability(context=world.context, methodology=methodology)
    assert "until D103" not in str(refusal.value)  # no future SUPPLIED admission


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


# ==== WS-D: D103.3 and D103.4 at the run boundary =======================================
#
# Positive cases run the real `run_scenario_var_es` with a book generated by the
# production engine from the U_D sub-portfolio (D103.1). Refusals assert the
# message prefix and zero calls in the widened spy set S (row aggregation, base
# valuation, Taylor certification, scenario generation, grid, Taylor P&L,
# conversion, sensitivity dispatch).

_FWD_ROW = {r["uid"]: r for r in ROWS if r["instrument"] == "VANILLA_FORWARD_NODE"}


def _eur_strike(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "strike_ccy": "EUR", "strike_discount_curve": "EUR_OIS"}


def _batch(w: SimpleNamespace, rows: list[dict[str, Any]]) -> IngestBatch:
    """Production ingest per row, concatenated: repeated UIDs are allowed (D055)."""
    out = _ingest(w.rsc, [rows[0]])
    for row in rows[1:]:
        out = _concat(out, _ingest(w.rsc, [row]))
    return out


def _book_from_batch(
    w: SimpleNamespace,
    batch: IngestBatch,
    *,
    zero_pairs: tuple[tuple[int, int, int], ...] = (),
    order: TaylorOrder = TaylorOrder.SECOND_ORDER_FULL,
    policy: DuplicatePolicy = DuplicatePolicy.RAISE,
) -> TaylorBlock:
    """U_D book generated by the production engine from `batch` at the run base;
    explicit zero cross pairs complete F(p) across blocks (D086, Q3)."""
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
    uids = np.asarray(portfolio.trade_uids, dtype=np.int64)
    v0 = {int(u): float(values[uids == u, 0].sum()) for u in np.unique(uids)}
    sens = run_sensitivities(portfolio=portfolio, uix_mapper=mapper, market_data=matrix)
    if zero_pairs:
        p, i, j = (np.array(c) for c in zip(*zero_pairs, strict=True))
        sens = dataclasses.replace(
            sens,
            cross_position_uids=np.r_[sens.cross_position_uids, p].astype(np.int64),
            cross_factor_rf_ids_i=np.r_[sens.cross_factor_rf_ids_i, i].astype(np.int32),
            cross_factor_rf_ids_j=np.r_[sens.cross_factor_rf_ids_j, j].astype(np.int32),
            cross_second_order=np.r_[sens.cross_second_order, np.zeros(p.size)],
        )
    return _certified_block(w, sens, v0, order=order, policy=policy)


def _derived_run(w, rows, derived, *, zero_pairs=(), policy=DuplicatePolicy.RAISE):
    batch = _batch(w, rows)
    book_rows = [r for r in rows if r["uid"] in derived]
    block = _book_from_batch(
        w, _batch(w, book_rows), zero_pairs=zero_pairs, policy=policy
    )
    return run_scenario_var_es(
        **_inputs(w, context=_context(w, batch), derived=derived, block=block)
    )


def _all_rows() -> list[dict[str, Any]]:
    return [dict(r) for r in ROWS]


# ---- positive: newly admitted TAYLOR/DERIVED forward UIDs ---------------------------


def test_d103_relative_node_derived_reaches_the_taylor_path(world, spy) -> None:
    run = _derived_run(world, _all_rows(), (FWD_USD, FWD_EUR))
    assert spy["_row_aggregation"] == 1 and spy["_certify_taylor_block"] == 1
    assert spy["run_into_grid"] >= 1 and run.pnl_provenance is not None
    assert np.isfinite(run.var).all() and np.isfinite(run.es).all()


def test_d103_repeated_forward_uid_derived_is_admitted(world, spy) -> None:
    rows = [*_all_rows(), {**_FWD_ROW[FWD_USD], "quantity": 25.0}]
    run = _derived_run(world, rows, (FWD_USD,))
    assert spy["_row_aggregation"] == 1 and run.pnl_provenance is not None


def test_d103_4a_forward_and_spot_rows_under_one_uid_are_admitted(world, spy) -> None:
    """D103.4(a): a SPOT row (direct FX, D084-admissible) under the forward UID.
    F(p) spans two blocks; the explicit zero (EQ_USD, BRENT) pair completes it."""
    rows = [*_all_rows(), _spot(FWD_USD, "EQ_USD", "USD", 7.0)]
    pair = (FWD_USD, 7, FWD_RF[FWD_USD])
    run = _derived_run(world, rows, (FWD_USD,), zero_pairs=(pair,))
    assert spy["_row_aggregation"] == 1 and run.pnl_provenance is not None


def test_d103_4a_shared_rf_mixed_uid_is_admitted_with_explicit_sum(world, spy) -> None:
    """Forward (USDEUR legs) and SPOT (USDEUR leg) under one UID: the caller
    opts into DuplicatePolicy.SUM (Q2); the run admits and prices the book."""
    rows = [*_all_rows(), _spot(FWD_EUR, "EQ_USD", "EUR", 5.0)]
    pair = (FWD_EUR, 7, FWD_RF[FWD_EUR])
    run = _derived_run(
        world, rows, (FWD_EUR,), zero_pairs=(pair,), policy=DuplicatePolicy.SUM
    )
    assert spy["_row_aggregation"] == 1 and run.pnl_provenance is not None


@pytest.mark.parametrize("kind", ["inverted_strike", "option_row"])
def test_d103_does_not_touch_full_reval(world, spy, kind) -> None:
    """FULL_REVAL: an inverted forward leg and a non-SPOT extra row stay
    admitted (D102.3 only)."""
    if kind == "inverted_strike":
        rows = [_eur_strike(r) if r["uid"] == FWD_USD else r for r in _all_rows()]
        context = _context(world, _batch(world, rows))
    else:
        rows = [*_all_rows(), _spot(FWD_USD, "EQ_USD", "USD", 3.0)]
        batch = _batch(world, rows)
        context = _with_batch(world, batch)
        object.__setattr__(context, "portfolio", _context(world, batch).portfolio)
        object.__setattr__(
            context,
            "ingest_batch",
            _relabel_row(batch, len(rows) - 1, VANILLA_EUROPEAN_OPTION_NODE_ID),
        )
    run_scenario_var_es(**_inputs(world, context=context))
    assert spy["_row_aggregation"] == 1


# ---- negative: refusals before any downstream work ----------------------------------


def _relabel_row(batch: IngestBatch, row: int, instrument_id: int) -> IngestBatch:
    ids = np.array(batch.instrument_ids, copy=True)
    ids[row] = instrument_id
    return dataclasses.replace(batch, instrument_ids=np.ascontiguousarray(ids))


def _refused(world, spy, context, derived, match) -> None:
    with pytest.raises(CompositionContractError, match=match):
        run_scenario_var_es(
            **_inputs(
                world,
                context=context,
                derived=derived,
                block=_caller_built_forward_block(world),
            )
        )
    _no_downstream_work(spy)


def _inverted_strike_context(world, mode=_NONE) -> ValuationContext:
    rows = [_eur_strike(r) if r["uid"] == FWD_USD else r for r in _all_rows()]
    batch = _batch(world, rows)
    mapper = build_uix_mapper([batch], as_of_ord32=world.as_of)
    return ValuationContext(
        static_config=world.rsc,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=mode,
    )


def test_d103_3_inverted_strike_leg_is_refused(world, spy) -> None:
    context = _inverted_strike_context(world)
    block = context.portfolio.vanilla_forward
    r = int(np.flatnonzero(block.trade_uids == FWD_USD)[0])
    m = context.uix_mapper
    assert m.efp_effective_quads[block.effective_forward_price_uix[r], 3] == INVALID_UIX
    leg = m.edf_effective_triples[block.effective_discount_factor_uix[r], 2]
    assert int(m.fx_effective_pairs[leg, 1]) == 1  # only the strike leg is inverted
    _refused(world, spy, context, (FWD_USD,), r"^D103\.3: .*\[4002\]")


def test_d103_3_inverted_asset_leg_is_refused(world, spy) -> None:
    """DEFENSIVE coverage of the compiled identity graph: the invert flag is set
    by the test on the asset leg's FX row (the strike leg has no FX). The
    real-ingest proof of the same state is
    `test_g1_real_ingest_asset_leg_inversion_is_refused_in_derived`."""
    rows = [_eur_strike(r) if r["uid"] == FWD_EUR else r for r in _all_rows()]
    context = _context(world, _batch(world, rows))
    block = context.portfolio.vanilla_forward
    r = int(np.flatnonzero(block.trade_uids == FWD_EUR)[0])
    m = copy.copy(context.uix_mapper)
    leg = int(m.efp_effective_quads[block.effective_forward_price_uix[r], 3])
    assert (
        m.edf_effective_triples[block.effective_discount_factor_uix[r], 2]
        == INVALID_UIX
    )
    pairs = np.array(m.fx_effective_pairs, copy=True)
    assert pairs[leg, 1] == 0
    pairs[leg, 1] = 1
    object.__setattr__(m, "fx_effective_pairs", np.ascontiguousarray(pairs))
    object.__setattr__(context, "uix_mapper", m)
    _refused(world, spy, context, (FWD_EUR,), r"^D103\.3: .*\[555\]")


def test_d103_3_every_row_of_a_repeated_uid_is_checked(world, spy) -> None:
    rows = [*_all_rows(), dict(_FWD_ROW[FWD_USD]), _eur_strike(_FWD_ROW[FWD_USD])]
    context = _context(world, _batch(world, rows))
    assert context.portfolio.vanilla_forward.trade_uids.tolist().count(FWD_USD) == 3
    _refused(world, spy, context, (FWD_USD,), r"^D103\.3: .*\[4002\]")


def test_d103_3_precedes_d103_4(world, spy) -> None:
    _refused(
        world,
        spy,
        _inverted_strike_context(world, DiscountingMode.CURVE),
        (FWD_USD,),
        r"^D103\.3: ",
    )


def test_d103_4b_curve_mode_is_refused(world, spy) -> None:
    context = ValuationContext(
        static_config=world.rsc,
        ingest_batch=world.batch,
        uix_mapper=world.context.uix_mapper,
        discounting_mode=DiscountingMode.CURVE,
    )
    _refused(world, spy, context, (FWD_USD,), r"^D103\.4: .*DiscountingMode\.NONE only")


def _option_context(world, mode=_NONE) -> ValuationContext:
    """Forward UID 4002 plus a second row whose ingest instrument is an option."""
    rows = [*_all_rows(), _spot(FWD_USD, "EQ_USD", "USD", 3.0)]
    batch = _batch(world, rows)
    context = ValuationContext(
        static_config=world.rsc,
        ingest_batch=batch,
        uix_mapper=build_uix_mapper([batch], as_of_ord32=world.as_of),
        discounting_mode=mode,
    )
    object.__setattr__(
        context,
        "ingest_batch",
        _relabel_row(batch, len(rows) - 1, VANILLA_EUROPEAN_OPTION_NODE_ID),
    )
    return context


def test_d103_4a_non_spot_row_under_a_derived_forward_uid_is_refused(
    world, spy
) -> None:
    """PROVENANCE (adversarial): the ingest row is an option while the compiled
    portfolio holds no option block for it (the row is relabelled after
    compilation). Source provenance is the ingest row, so D103.4(a) must refuse
    whatever the compiled blocks say. The functional real-ingest case is
    `test_g3_real_option_row_under_a_derived_forward_uid_is_refused`."""
    _refused(
        world,
        spy,
        _option_context(world),
        (FWD_USD,),
        r"^D103\.4: .*not SPOT .*\[4002\]",
    )


def test_d103_4_b_is_applied_before_a(world, spy) -> None:
    """Implementation precedence (not a contract ordering): mode before rows."""
    _refused(
        world,
        spy,
        _option_context(world, DiscountingMode.CURVE),
        (FWD_USD,),
        r"^D103\.4: .*DiscountingMode\.NONE only",
    )


def test_d084_still_refuses_an_inverted_spot_row_of_a_derived_forward_uid(
    world, spy
) -> None:
    rows = [*_all_rows(), _spot(FWD_USD, "EQ_GBP", "EUR", 4.0)]
    _refused(world, spy, _context(world, _batch(world, rows)), (FWD_USD,), r"D084")


def test_d102_3_precedes_d103_3(world, spy) -> None:
    rows = [*_all_rows(), _eur_strike(_FWD_ROW[FWD_USD])]
    batch = _batch(world, rows)
    context = _context(world, batch)
    object.__setattr__(
        context,
        "ingest_batch",
        _relabel_row(batch, len(rows) - 1, VANILLA_FORWARD_CURVE_ID),
    )
    _refused(world, spy, context, (FWD_USD,), r"^D102\.3: ")


@pytest.mark.parametrize("case", ["d103_3", "d103_4b", "d103_4a"])
def test_d103_guard_is_the_barrier_before_row_aggregation(
    world, spy, monkeypatch, case
):
    """D059: with the helper disabled, each refused composition passes the
    boundary and reaches `_row_aggregation`."""
    context = {
        "d103_3": lambda: _inverted_strike_context(world),
        "d103_4b": lambda: _inverted_strike_context(world, DiscountingMode.CURVE),
        "d103_4a": lambda: _option_context(world),
    }[case]()
    monkeypatch.setattr(api, "_require_forward_capability", lambda **kw: None)
    with pytest.raises((CompositionContractError, ValueError, NotImplementedError)):
        run_scenario_var_es(
            **_inputs(
                world,
                context=context,
                derived=(FWD_USD,),
                block=_caller_built_forward_block(world),
            )
        )
    assert spy["_row_aggregation"] == 1


# ---- D086 completeness across blocks (Q3 evidence; production unchanged) -------------


def _mixed_sens(world, rows):
    batch = _batch(world, rows)
    mapper = build_uix_mapper([batch], as_of_ord32=world.as_of)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=world.rsc, uix_mapper=mapper, discounting_mode=_NONE
    )
    row = {int(f): i for i, f in enumerate(world.fids)}
    flat = world.base[[row[int(f)] for f in layout.required_factor_ids]]
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=_NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=np.ascontiguousarray(flat[:, None]),
        num_scenarios=1,
    )
    return run_sensitivities(portfolio=portfolio, uix_mapper=mapper, market_data=matrix)


def _plus_pair(sens, uid, i, j, value):
    lo, hi = min(i, j), max(i, j)
    return dataclasses.replace(
        sens,
        cross_position_uids=np.r_[sens.cross_position_uids, uid].astype(np.int64),
        cross_factor_rf_ids_i=np.r_[sens.cross_factor_rf_ids_i, lo].astype(np.int32),
        cross_factor_rf_ids_j=np.r_[sens.cross_factor_rf_ids_j, hi].astype(np.int32),
        cross_second_order=np.r_[sens.cross_second_order, value],
    )


def _certify(
    world, sens, order=TaylorOrder.SECOND_ORDER_FULL, policy=DuplicatePolicy.RAISE
):
    v0 = dict.fromkeys(np.unique(sens.position_uids).tolist(), 0.0)
    return _certified_block(world, sens, v0, order=order, policy=policy)


def test_d086_full_completeness_spans_blocks(world) -> None:
    """F(p) is global to the position: block A gives the forward factor, block B
    the SPOT factor; FULL needs the cross-block pair, an explicit zero satisfies
    it, and P1-A does not check its value."""
    sens = _mixed_sens(world, [_FWD_ROW[FWD_USD], _spot(FWD_USD, "EQ_USD", "USD", 7.0)])
    assert sens.cross_position_uids.size == 0
    with pytest.raises(TaylorContractError, match="MISSING"):
        _certify(world, sens)
    _certify(world, _plus_pair(sens, FWD_USD, 7, FWD_RF[FWD_USD], 0.0))
    _certify(world, _plus_pair(sens, FWD_USD, 7, FWD_RF[FWD_USD], 1.0))


def test_d086_shared_rf_raise_and_explicit_sum(world) -> None:
    sens = _mixed_sens(world, [_FWD_ROW[FWD_EUR], _spot(FWD_EUR, "EQ_USD", "EUR", 5.0)])
    with pytest.raises(TaylorContractError, match="MISSING"):  # completeness first
        _certify(world, sens)
    complete = _plus_pair(sens, FWD_EUR, 7, FWD_RF[FWD_EUR], 0.0)
    with pytest.raises(ValueError, match="duplicate"):  # then the RAISE guard
        _certify(world, complete)
    with pytest.raises(TaylorContractError, match="MISSING"):  # SUM cannot add a pair
        _certify(world, sens, policy=DuplicatePolicy.SUM)
    _certify(world, complete, policy=DuplicatePolicy.SUM)


@pytest.mark.parametrize(
    "order,match",
    [
        (TaylorOrder.FIRST_ORDER, "cross terms"),
        (TaylorOrder.SECOND_ORDER_TRUNCATED, "omits cross"),
    ],
)
def test_d086_lower_orders_refuse_forward_cross_terms(world, order, match) -> None:
    sens = _mixed_sens(world, [_FWD_ROW[FWD_EUR]])
    assert sens.cross_position_uids.size == 1
    with pytest.raises(TaylorContractError, match=match):
        _certify(world, sens, order=order)
    _certify(world, sens)  # the complete FULL book is admitted


def test_d103_4a_applies_to_derived_uids_only(world, spy) -> None:
    """A FULL_REVAL forward UID with a non-SPOT row beside an admitted
    TAYLOR/DERIVED forward UID: D103.4(a) inspects the derived UID only."""
    rows = [*_all_rows(), _spot(FWD_USD, "EQ_USD", "USD", 3.0)]
    batch = _batch(world, rows)
    context = _context(world, batch)
    object.__setattr__(
        context,
        "ingest_batch",
        _relabel_row(batch, len(rows) - 1, VANILLA_EUROPEAN_OPTION_NODE_ID),
    )
    block = _book_from_batch(world, _batch(world, [_FWD_ROW[FWD_EUR]]))
    run_scenario_var_es(
        **_inputs(world, context=context, derived=(FWD_EUR,), block=block)
    )
    assert spy["_row_aggregation"] == 1 and spy["run_into_grid"] >= 1


# ==== WS-D evidence hardening: real ingest in the canonical world ========================
#
# Canonical static configuration plus the WS-C test-owned exact-NODE curves
# (including the EUR LABEL curve TEST_EXACT_EUR), compiled by production ingest.
# These states are reachable from valid source data: nothing is set on the
# compiled identity graph.

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sensitivities"))
import test_vanilla_forward_sensitizer as tv  # canonical + test-owned curves


def _canonical_context(rows: list[dict[str, Any]]) -> ValuationContext:
    """Production ingest per row (repeated UIDs allowed, D055), concatenated."""
    batches = [
        compile_raw_df_into_ingest_batch(
            raw_df=pl.from_pandas(pd.DataFrame([row])),
            static_config=tv.RSC,
            as_of_date=tv.fx.AS_OF,
        )
        for row in rows
    ]
    batch = batches[0]
    for other in batches[1:]:
        batch = _concat(batch, other)
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(tv.fx.AS_OF)))
    return ValuationContext(
        static_config=tv.RSC,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=_NONE,
    )


def _canonical_run(context: ValuationContext, *, derived: tuple[int, ...] = ()):
    """`run_scenario_var_es` on a flat 2-scenario history at the canonical base.
    A TAYLOR/DERIVED run gets an uncertified one-row book: every case below is
    refused before `_require_taylor_block` / certification is reached."""
    axis = api.derive_run_factor_axis(valuation_context=context, reporting_ccy="USD")
    name = {v: k for k, v in tv.RF.items()}
    base = np.array([tv.LEVELS.get(name[int(f)], 1.0) for f in axis.factor_ids])
    ids = np.sort(axis.factor_ids).astype(np.int32)
    as_of = int(date_to_ord32(tv.fx.AS_OF))
    data = RiskFactorMatrix(
        dates=np.arange(as_of - 2, as_of + 1, dtype=np.int32),
        factor_ids=ids,
        values=np.asfortranarray(
            np.column_stack([np.full(3, tv.LEVELS.get(name[int(f)], 1.0)) for f in ids])
        ),
    )
    uids = np.unique(np.asarray(context.ingest_batch.uids, dtype=np.int64))
    u_d = np.array(sorted(derived), dtype=np.int64)
    block = None
    if u_d.size:
        sens = BlockSensitivities(
            position_uids=u_d,
            factor_rf_ids=np.full(u_d.size, int(axis.factor_ids[0]), dtype=np.int32),
            first_order=np.ones(u_d.size),
            second_order=np.zeros(u_d.size),
            cross_position_uids=np.empty(0, dtype=np.int64),
            cross_factor_rf_ids_i=np.empty(0, dtype=np.int32),
            cross_factor_rf_ids_j=np.empty(0, dtype=np.int32),
            cross_second_order=np.empty(0),
        )
        prefix = np.asarray(
            axis.factor_ids[: axis.num_valuation_factors], dtype=np.int32
        )
        spec = build_sensitivity_book_spec(
            sensitivities=sens, required_factor_ids=prefix
        )
        block = TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))
    return run_scenario_var_es(
        scenario_config=parse_scenario_model_config({
            "model": "historical",
            "horizon": 1,
        }),
        data=data,
        required_factor_ids=axis.factor_ids.copy(),
        required_shift_type_id=axis.shift_type_ids.copy(),
        proxy_terms=(),
        base_levels=base,
        methodology=MethodologyDeclaration(
            production_position_uids=uids,
            full_reval_uids=np.setdiff1d(uids, u_d),
            taylor_supplied_uids=np.empty(0, dtype=np.int64),
            taylor_derived_uids=u_d,
        ),
        valuation_context=context,
        reporting=ReportingSpec(reporting_ccy="USD"),
        confidence_level=0.5,
        batch_size=2,
        taylor_derived=block,
    )


#: EUR exact-NODE forward settled in USD with a USD strike: the ASSET leg is
#: EUR -> USD (USDEUR quotes EUR per USD, so 1 / USDEUR: invert = 1); the
#: strike leg has no FX.
_ASSET_INVERTED = tv._fwd(41, "TEST_EXACT_EUR[1M]", "EUR", 10.0, 50.0, "USD", "USD")


def _asset_inverted_context() -> ValuationContext:
    context = _canonical_context([_ASSET_INVERTED])
    block, m = context.portfolio.vanilla_forward, context.uix_mapper
    asset = int(m.efp_effective_quads[block.effective_forward_price_uix[0], 3])
    strike = int(m.edf_effective_triples[block.effective_discount_factor_uix[0], 2])
    assert asset != INVALID_UIX and int(m.fx_effective_pairs[asset, 1]) == 1
    assert strike == INVALID_UIX  # only the asset leg carries FX, and it is inverted
    return context


def test_g1_real_ingest_asset_leg_inversion_is_refused_in_derived(spy) -> None:
    """G-1, case A: D103.3 on a REAL-INGEST inverted asset leg, before any
    downstream work."""
    with pytest.raises(CompositionContractError, match=r"^D103\.3: .*\[41\]"):
        _canonical_run(_asset_inverted_context(), derived=(41,))
    _no_downstream_work(spy)


def test_g1_real_ingest_asset_leg_inversion_is_admitted_in_full_reval(spy) -> None:
    """G-1, case B: the same row in FULL_REVAL is admitted and valued."""
    run = _canonical_run(_asset_inverted_context())
    assert run.position_uids.tolist() == [41]
    assert spy["_row_aggregation"] == 1 and spy["_value_positions"] >= 1
    assert spy["build_scenario_generator"] == 1 and spy["allocate_pnl_grid"] == 1
    assert spy["convert_scenario_pnl_to_reporting_currency"] == 1


def test_g2_full_reval_inverted_forward_is_admitted_beside_a_derived_forward(
    world: SimpleNamespace, spy
) -> None:
    """G-2: D103.3 is TAYLOR/DERIVED-only. UID 4002 (FULL_REVAL) has an
    inverted strike leg (real ingest); UID 555 is an admitted TAYLOR/DERIVED
    forward, so the D103.3 check executes for this run. 4002 must stay admitted
    and be valued by full revaluation."""
    rows = [_eur_strike(r) if r["uid"] == FWD_USD else r for r in _all_rows()]
    context = _context(world, _batch(world, rows))
    block, m = context.portfolio.vanilla_forward, context.uix_mapper
    r = int(np.flatnonzero(block.trade_uids == FWD_USD)[0])
    leg = int(m.edf_effective_triples[block.effective_discount_factor_uix[r], 2])
    assert int(m.fx_effective_pairs[leg, 1]) == 1  # 4002 is really inverted
    book = _book_from_batch(world, _batch(world, [_FWD_ROW[FWD_EUR]]))
    run = run_scenario_var_es(
        **_inputs(world, context=context, derived=(FWD_EUR,), block=book)
    )
    assert spy["_row_aggregation"] == 1 and spy["run_into_grid"] >= 1
    assert FWD_USD in run.position_uids.tolist()
    assert np.isfinite(_position_var(run, FWD_USD))


def test_g3_real_option_row_under_a_derived_forward_uid_is_refused(spy) -> None:
    """G-3 functional: a REAL-INGEST, really compiled NODE option row under the
    forward UID (compiled into the option block) fails D103.4(a)."""
    forward = tv._fwd(
        777,
        "BRENT_CRUDE_FWD[M+0]",
        "USD",
        60.0,
        80.0,
        "USD",
        "USD",
        dt.date(2026, 2, 16),
    )
    option = {
        "uid": 777,
        "instrument": "VANILLA_EUROPEAN_OPTION_NODE",
        "quantity": 1_000.0,
        "underlying": "BRENT_CRUDE_FWD[M+0]",
        "settlement_ccy": "USD",
        "strike_price": 70.0,
        "strike_ccy": "USD",
        "is_call": 1.0,
        "expiry_date": dt.date(2026, 2, 18),
        "settlement_date": dt.date(2026, 2, 20),
        "volatility_container_name": "BRENT_CRUDE_ATM_VOL",
        "under_discount_curve": "USD_SOFR_OIS",
        "strike_discount_curve": "USD_SOFR_OIS",
    }
    context = _canonical_context([forward, option])
    assert 777 in context.portfolio.vanilla_european_option.trade_uids.tolist()
    assert 777 in context.portfolio.vanilla_forward.trade_uids.tolist()
    with pytest.raises(
        CompositionContractError, match=r"^D103\.4: .*not SPOT .*\[777\]"
    ):
        _canonical_run(context, derived=(777,))
    _no_downstream_work(spy)


# ==== Evidence closure: scope detectors for D084, D103.4(a) and D103.4(b) ================


def _world_for(w: SimpleNamespace, rows: list[dict[str, Any]]) -> SimpleNamespace:
    """The `world` construction for another row set: same configuration, as-of,
    base levels by name and history rule, so the run axis matches the rows."""
    batch = _batch(w, rows)
    context = _context(w, batch)
    layout = build_universe_mapping(
        static_config=w.rsc, uix_mapper=context.uix_mapper, discounting_mode=_NONE
    )
    name = {v: k for k, v in w.rsc.rel_name_to_rf_id.items()}
    fids = np.asarray(layout.required_factor_ids, dtype=np.int32)
    base = np.array([_BASE_LEVELS[name[int(f)]] for f in fids], dtype=np.float64)
    rng = np.random.default_rng(11)
    values = np.asfortranarray(
        base
        * np.exp(np.cumsum(rng.normal(0.0, 0.01, (_HISTORY_DAYS, fids.size)), axis=0))
    )
    return SimpleNamespace(
        rsc=w.rsc,
        as_of=w.as_of,
        batch=batch,
        context=context,
        layout=layout,
        fids=fids,
        base=base,
        name=name,
        data=RiskFactorMatrix(
            dates=w.data.dates.copy(), factor_ids=fids.copy(), values=values
        ),
        uids=np.unique(np.asarray(batch.uids, dtype=np.int64)),
        reporting=w.reporting,
    )


def _base_values(w: SimpleNamespace) -> dict[int, float]:
    """V_0 per UID on the run's own route: the context's compiled portfolio
    valued at the base levels, rows summed into their parent UID."""
    mapper = w.context.uix_mapper
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=_NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=w.layout,
        flat_levels=np.ascontiguousarray(w.base[:, None]),
        num_scenarios=1,
    )
    portfolio = w.context.portfolio
    values = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(matrix=matrix, uix_mapper=mapper),
        out=values,
    )
    uids = np.asarray(portfolio.trade_uids, dtype=np.int64)
    return {int(u): float(values[uids == u, 0].sum()) for u in np.unique(uids)}


#: UID 4002 owns its valid USD forward row and a SPOT row on EQ_EUR settled in
#: USD (EUR -> USD = 1 / USDEUR: invert = 1). Both rows settle in USD, so no
#: settlement-currency rule applies: D084 is the only contract the TAYLOR/DERIVED
#: composition violates.
_INVERTED_SPOT_ROW = _spot(FWD_USD, "EQ_EUR", "USD", 4.0)


def _d084_world(world: SimpleNamespace) -> SimpleNamespace:
    w = _world_for(world, [*_all_rows(), _INVERTED_SPOT_ROW])
    spot, m = w.context.portfolio.spot, w.context.uix_mapper
    r = int(np.flatnonzero(spot.trade_uids == FWD_USD)[0])
    fx_row = int(m.spot_effective_pairs[spot.effective_spot_price_uix[r], 1])
    assert fx_row != INVALID_UIX and int(m.fx_effective_pairs[fx_row, 1]) == 1
    return w


def _d084_book(w: SimpleNamespace) -> TaylorBlock:
    """Certified caller-built FIRST_ORDER book for UID 4002 (forward + inverted
    SPOT row). P1-A binds base, order and completeness, not Greek values, so
    only D084 stops this book from pricing the inverted leg."""
    rf = w.rsc.rel_name_to_rf_id
    level = {w.name[int(f)]: float(b) for f, b in zip(w.fids, w.base, strict=True)}
    q, s, x = _INVERTED_SPOT_ROW["quantity"], level["EQ_EUR"], level["USDEUR"]
    sens = BlockSensitivities(
        position_uids=np.full(3, FWD_USD, dtype=np.int64),
        factor_rf_ids=np.array(
            [FWD_RF[FWD_USD], rf["EQ_EUR"], rf["USDEUR"]], dtype=np.int32
        ),
        first_order=np.array([100.0, q / x, -q * s / (x * x)], dtype=np.float64),
        second_order=np.zeros(3, dtype=np.float64),
        cross_position_uids=np.empty(0, dtype=np.int64),
        cross_factor_rf_ids_i=np.empty(0, dtype=np.int32),
        cross_factor_rf_ids_j=np.empty(0, dtype=np.int32),
        cross_second_order=np.empty(0, dtype=np.float64),
    )
    return _certified_block(w, sens, _base_values(w), order=TaylorOrder.FIRST_ORDER)


def test_d084_refuses_an_inverted_spot_row_of_a_derived_forward_uid_same_ccy(
    world: SimpleNamespace, spy
) -> None:
    """M10 detector without a settlement-currency confound. Control: the same
    rows in FULL_REVAL are admitted. TAYLOR/DERIVED: D084, before any
    downstream work."""
    w = _d084_world(world)
    full = run_scenario_var_es(**_inputs(w))
    assert spy["_row_aggregation"] == 1
    assert full.position_uids.tolist() == w.uids.tolist()
    for name in spy:
        spy[name] = 0
    block = _d084_book(w)
    with pytest.raises(CompositionContractError, match=r"\[4002\]\)\. .*\(D084\)\.$"):
        run_scenario_var_es(**_inputs(w, derived=(FWD_USD,), block=block))
    _no_downstream_work(spy)


def test_d084_is_the_barrier_for_that_composition(
    world: SimpleNamespace, spy, monkeypatch
) -> None:
    """D059: with the D084 check disabled, the certified book prices the
    inverted SPOT leg of UID 4002 through TAYLOR/DERIVED and the run completes."""
    w = _d084_world(world)
    block = _d084_book(w)
    monkeypatch.setattr(api, "_require_taylor_capable_spot", lambda **kw: None)
    run = run_scenario_var_es(**_inputs(w, derived=(FWD_USD,), block=block))
    assert spy["_certify_taylor_block"] == 1 and spy["run_into_grid"] >= 1
    assert run.pnl_provenance is not None
    assert np.isfinite(_position_var(run, FWD_USD))


def test_d103_4a_does_not_refuse_a_derived_uid_that_owns_no_forward_row() -> None:
    """D103.4 scope: the rule applies to TAYLOR/DERIVED UIDs that own a
    vanilla_forward row; all other contracts apply unchanged (D102 scope).
    UID 888 owns only a real-ingest option row and sits beside the DERIVED
    forward UID 777, so D103.4(a) does not refuse it. Its admission is a
    matter for the global contracts, not for D103."""
    forward = tv._fwd(
        777,
        "BRENT_CRUDE_FWD[M+0]",
        "USD",
        60.0,
        80.0,
        "USD",
        "USD",
        dt.date(2026, 2, 16),
    )
    option = {
        "uid": 888,
        "instrument": "VANILLA_EUROPEAN_OPTION_NODE",
        "quantity": 1_000.0,
        "underlying": "BRENT_CRUDE_FWD[M+0]",
        "settlement_ccy": "USD",
        "strike_price": 70.0,
        "strike_ccy": "USD",
        "is_call": 1.0,
        "expiry_date": dt.date(2026, 2, 18),
        "settlement_date": dt.date(2026, 2, 20),
        "volatility_container_name": "BRENT_CRUDE_ATM_VOL",
        "under_discount_curve": "USD_SOFR_OIS",
        "strike_discount_curve": "USD_SOFR_OIS",
    }
    context = _canonical_context([forward, option])
    assert 888 in context.portfolio.vanilla_european_option.trade_uids.tolist()
    assert 888 not in context.portfolio.vanilla_forward.trade_uids.tolist()
    uids = np.array([777, 888], dtype=np.int64)
    methodology = MethodologyDeclaration(
        production_position_uids=uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=uids,
    )
    api._require_forward_capability(context=context, methodology=methodology)


def test_full_reval_forward_is_admitted_and_valued_in_curve_mode(spy) -> None:
    """D103.4(b) scope: DiscountingMode.CURVE refuses TAYLOR/DERIVED forward
    UIDs only. A FULL_REVAL NODE forward on USD_SOFR_OIS (the constructible
    curve, D080) stays admitted and is valued. Admission evidence only: CURVE
    valuation numbers are not certified here (OD-1)."""
    row = tv._fwd(
        41,
        "BRENT_CRUDE_FWD[M+0]",
        "USD",
        60.0,
        80.0,
        "USD",
        "USD",
        dt.date(2026, 2, 16),
    )
    node = _canonical_context([row])
    context = ValuationContext(
        static_config=node.static_config,
        ingest_batch=node.ingest_batch,
        uix_mapper=node.uix_mapper,
        discounting_mode=DiscountingMode.CURVE,
    )
    curve_axis = api.derive_run_factor_axis(
        valuation_context=context, reporting_ccy="USD"
    )
    node_axis = api.derive_run_factor_axis(valuation_context=node, reporting_ccy="USD")
    assert curve_axis.num_valuation_factors > node_axis.num_valuation_factors
    run = _canonical_run(context)
    assert run.position_uids.tolist() == [41]
    assert run.manifest.pnl.composition.context.discounting_mode == "CURVE"
    assert spy["_row_aggregation"] == 1 and spy["_value_positions"] >= 1
    assert spy["allocate_pnl_grid"] == 1 and spy["run_into_grid"] == 0
