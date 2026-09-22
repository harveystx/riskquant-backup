# tests/solutions/risk_run/test_run_owned_identity.py
"""
P1-B: a RiskRun is tied to the as-of, static configuration, valuation contract
and base state that produced it, and every V_0 is computed by the run.

The run derives each context field from the objects it computes with
(`ValuationContext`), so a mutation of any of them either changes the run
identity or fails closed. V_0 is a derived result: it is computed, checked
against the certified Taylor binding, and never hashed.
Independent V_0 oracle (L1): qty * P[underlying] * (USDEUR if settled in EUR).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

import riskquant.solutions.risk_run.api as api
from riskquant import constants
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError

FX, NONE = mf.FX, mf.FX.DiscountingMode.NONE
ATOL_V0 = 1e-9


def _ctx(w, *, rsc=None, as_of_shift=0):
    mapper = FX.build_uix_mapper([w.batch], as_of_ord32=w.as_of + as_of_shift)
    return ValuationContext(
        static_config=w.rsc if rsc is None else rsc,
        ingest_batch=w.batch,
        uix_mapper=mapper,
        discounting_mode=NONE,
    )


def _mutated_rsc(**shift_by_name):
    sheets = dict(FX._RISK_STATIC_CONFIG_DICT_OF_DFS)
    key = next(k for k, v in sheets.items() if v is FX._NODE_META_DF)
    node = FX._NODE_META_DF
    sheets[key] = node.with_columns(
        pl.Series(
            "shift_type",
            [
                shift_by_name.get(n, s)
                for n, s in zip(node["node_name"], node["shift_type"], strict=True)
            ],
        )
    )
    return FX.compile_risk_static_config_from_dict_of_dfs(sheets)


def _outcome(inp):
    try:
        r = run_scenario_var_es(**inp)
    except (CompositionContractError, TaylorContractError) as exc:
        return ("REJECTED", type(exc).__name__)
    return (r.identity, tuple(v.hex() for v in r.var.tolist()), r)


# ---------------------------------------------------------------------------
# as-of
# ---------------------------------------------------------------------------
def test_as_of_is_run_owned_and_part_of_identity() -> None:
    w = mf.world()
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    inp["valuation_context"] = _ctx(w, as_of_shift=1)
    moved = _outcome(inp)
    assert moved[2].manifest.pnl.composition.context.as_of_ord32 == w.as_of + 1
    assert moved[0] != base[0]  # identity follows the as-of
    assert moved[1] == base[1]  # spot, no discounting: same numbers


def test_market_state_from_another_as_of_fails_closed() -> None:
    w = mf.world()
    other = FX.build_uix_mapper([w.batch], as_of_ord32=w.as_of + 1)
    ctx = _ctx(w, as_of_shift=1)  # N1: compiled by the context itself
    assert (
        ctx.portfolio.uix_fingerprint == ctx.uix_mapper.fingerprint == other.fingerprint
    )
    with pytest.raises(TypeError):  # no caller-built portfolio can enter
        ValuationContext(
            static_config=w.rsc,
            portfolio=w.portfolio,
            uix_mapper=other,
            discounting_mode=NONE,
        )
    inp = mf.inputs()  # DERIVED book certified at the old as-of
    inp["valuation_context"] = _ctx(w, as_of_shift=1)
    with pytest.raises(TaylorContractError, match="as_of_ord32"):
        run_scenario_var_es(**inp)


# ---------------------------------------------------------------------------
# static configuration
# ---------------------------------------------------------------------------
def test_static_configuration_identity_is_content_not_object() -> None:
    w = mf.world()
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    fresh = FX.compile_risk_static_config_from_dict_of_dfs(
        dict(FX._RISK_STATIC_CONFIG_DICT_OF_DFS)
    )
    assert fresh is not w.rsc
    inp["valuation_context"] = _ctx(w, rsc=fresh)
    assert _outcome(inp)[:2] == base[:2]


def test_a_load_bearing_static_configuration_change_moves_identity_or_fails_closed() -> (
    None
):
    w = mf.world()
    rsc = _mutated_rsc(WTI_CRUDE="SIMPLE_RETURN")
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    stale = mf.inputs(full=mf.ALL_UIDS, derived=())
    stale["valuation_context"] = _ctx(w, rsc=rsc)
    assert _outcome(stale) == (
        "REJECTED",
        "CompositionContractError",
    )  # shift types are owned
    fresh = mf.inputs(full=mf.ALL_UIDS, derived=())
    fresh["valuation_context"] = _ctx(w, rsc=rsc)
    fresh["required_shift_type_id"] = np.ascontiguousarray(
        FX.build_universe_mapping(
            static_config=rsc,
            uix_mapper=fresh["valuation_context"].uix_mapper,
            discounting_mode=NONE,
        ).required_shift_type_ids
    )
    moved = _outcome(fresh)
    assert moved[0] != base[0] and moved[1] != base[1]
    mixed = mf.inputs()  # certified under the old configuration
    mixed["valuation_context"] = fresh["valuation_context"]
    mixed["required_shift_type_id"] = fresh["required_shift_type_id"]
    with pytest.raises(TaylorContractError, match="static_config_digest"):
        run_scenario_var_es(**mixed)


# ---------------------------------------------------------------------------
# valuation contract
# ---------------------------------------------------------------------------
def test_a_valuation_contract_change_moves_identity_and_refuses_old_books(
    monkeypatch,
) -> None:
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    mixed = mf.inputs()
    monkeypatch.setattr(constants, "TTE_FLOOR", constants.TTE_FLOOR * 2.0)
    moved = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    assert moved[0] != base[0]
    with pytest.raises(TaylorContractError, match="valuation_contract_fingerprint"):
        run_scenario_var_es(**mixed)


# ---------------------------------------------------------------------------
# Taylor V_0: run-computed, never supplied
# ---------------------------------------------------------------------------
def test_taylor_v0_is_run_computed_and_matches_the_independent_oracle(
    monkeypatch,
) -> None:
    seen = {}
    real = api.convert_scenario_pnl_to_reporting_currency
    monkeypatch.setattr(
        api,
        "convert_scenario_pnl_to_reporting_currency",
        lambda **kw: seen.setdefault("kw", kw) and real(**kw),
    )
    run = run_scenario_var_es(**mf.inputs(full=(), derived=mf.ALL_UIDS))
    base = mf.world().su.base_levels
    for i, uid in enumerate(run.position_uids.tolist()):
        want = float(mf.l1_value(uid, base[:, None])[0])
        assert seen["kw"]["base_settlement_values"][i] == pytest.approx(
            want, abs=ATOL_V0
        )


def test_no_caller_v0_or_binding_can_be_supplied() -> None:
    calc = mf.derived_block(mf.DERIVED_UIDS).calculator
    for extra in ({"base_settlement_values": np.zeros(3)}, {"expected_binding": None}):
        with pytest.raises(TypeError):
            TaylorBlock(calculator=calc, **extra)


def test_a_book_certified_at_another_base_state_fails_closed() -> None:
    import _fixture as fx

    inp = fx.inputs()
    inp["taylor_derived"] = fx.derived(fx.calculator(fx.base_levels() * 1.001))
    with pytest.raises(
        TaylorContractError, match="base_levels_digest.*base_settlement_values_digest"
    ):
        run_scenario_var_es(**inp)


# ---------------------------------------------------------------------------
# identity mutation matrix (one run-defining input at a time)
# ---------------------------------------------------------------------------
def _drop_2006():
    frame = mf.positions(mf.BASE)
    w = mf.build_world(frame[frame["uid"] != 2_006].reset_index(drop=True))
    universe = tuple(u for u in mf.ALL_UIDS if u != 2_006)
    inp = mf.inputs(full=universe, derived=(), universe=universe)
    inp["valuation_context"] = ValuationContext(
        static_config=w.rsc,
        ingest_batch=w.batch,
        uix_mapper=w.mapper,
        discounting_mode=NONE,
    )
    return inp


def _permuted_axis():
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    order = np.r_[1, 0, np.arange(2, inp["required_factor_ids"].shape[0])]
    for k in (
        "required_factor_ids",
        "required_shift_type_id",
        "source_factor_ids",
        "shock_scale",
        "base_levels",
    ):
        inp[k] = np.ascontiguousarray(inp[k][order])
    return inp


def _moved_partition():
    return mf.inputs(full=(2_003, 2_004, 2_005), derived=(2_001, 2_002, 2_006))


def _moved_base():
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    inp["base_levels"] = inp["base_levels"] * 1.001
    return inp


@pytest.mark.parametrize(
    "name,build,expect",
    [
        (
            "identical_objects",
            lambda: mf.inputs(full=mf.ALL_UIDS, derived=()),
            ("same", "same"),
        ),
        (
            "as_of",
            lambda: {
                **mf.inputs(full=mf.ALL_UIDS, derived=()),
                "valuation_context": _ctx(mf.world(), as_of_shift=1),
            },
            ("moved", "same"),
        ),
        ("economic_universe", _drop_2006, ("moved", "moved")),
        ("methodology_partition", _moved_partition, ("moved", None)),
        ("factor_axis_order", _permuted_axis, ("REJECTED", None)),
        ("base_state", _moved_base, ("moved", "moved")),
    ],
)
def test_identity_mutation_matrix(name, build, expect) -> None:
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    got = _outcome(build())
    if expect[0] == "REJECTED":
        assert got[0] == "REJECTED", name
        return
    assert (got[0] == base[0]) == (expect[0] == "same"), name
    if expect[1] is not None:
        assert (got[1] == base[1]) == (expect[1] == "same"), name


# ---------------------------------------------------------------------------
# discounting mode (valuation audit): never silently shares an identity
# ---------------------------------------------------------------------------
def test_discounting_mode_is_bound_to_identity_or_refused() -> None:
    """A CURVE context over the NONE-mode world either fails closed or yields a
    run whose identity differs; it can never reproduce the NONE identity."""
    w = mf.world()
    base = _outcome(mf.inputs(full=mf.ALL_UIDS, derived=()))
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    inp["valuation_context"] = ValuationContext(
        static_config=w.rsc,
        ingest_batch=w.batch,
        uix_mapper=w.mapper,
        discounting_mode=FX.DiscountingMode.CURVE,
    )
    try:
        got = _outcome(inp)
    except Exception as exc:  # any refusal is acceptable; silence is not
        got = ("REJECTED", type(exc).__name__)
    assert got[0] == "REJECTED" or got[0] != base[0], got[:2]
