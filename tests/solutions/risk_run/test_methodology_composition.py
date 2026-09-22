# tests/solutions/risk_run/test_methodology_composition.py
"""
D081 §11 production composition at `run_scenario_var_es`.

FULL_REVAL, TAYLOR/DERIVED and the refused TAYLOR/SUPPLIED over ONE explicit
production universe U, ONE canonical scenario stream, ONE PnlGrid, ONE D081
conversion. Row-level numbers are compared with independent references (L1
valuation, L4 Taylor, L0 reporting) at the exact matrices the production path
hands to reporting and to VaR/ES. Fail-closed contracts are proven to fail
BEFORE any numerics (no scenario generated, no grid allocated). The fixture is
identifiable: the DERIVED book is SECOND_ORDER_TRUNCATED, so FULL and Taylor
values differ on quanto rows, and HOSTILE UIDs make every index map non-trivial.

Tolerance. The settlement grid is float32 by contract; |dV| < 2^11 here, so one
rounding errs by <= 2^-13 ~ 1.2e-4. ATOL = 1e-3 bounds that rounding of dV plus
the float64 reporting arithmetic; V0 is float64 end to end (ATOL_V0 = 1e-9).
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

import riskquant.solutions.risk_run.api as api
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
    ValuationContext,
)
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError
from riskquant.valuation.blocks.spot_block import SpotBlock

ATOL = 1e-3
ATOL_V0 = 1e-9
WORLDS = [
    pytest.param(mf.BASE, id="base_uids"),
    pytest.param(mf.HOSTILE, id="hostile_uids"),
]


@pytest.fixture
def spy(monkeypatch):
    """Record what the production path hands to each stage (calls through)."""
    seen: dict = {"generators": 0, "shock_batches": 0, "grids": 0, "matrices": []}
    real = {
        n: getattr(api, n)
        for n in (
            "convert_scenario_pnl_to_reporting_currency",
            "compute_var_es",
            "allocate_pnl_grid",
            "build_scenario_generator",
            "apply_shocks_to_base_levels",
            "build_market_data_matrix_from_flat_levels",
            "_write_full_reval_block",
        )
    }
    real_into = SensitivityPnlCalculator.run_into_grid

    def _convert(**kw):
        seen["convert"] = kw
        return real["convert_scenario_pnl_to_reporting_currency"](**kw)

    def _var_es(groups, confidence):
        seen["groups"] = np.array(groups, copy=True)
        return real["compute_var_es"](groups, confidence)

    def _alloc(**kw):
        seen["grids"] += 1
        seen["grid"] = real["allocate_pnl_grid"](**kw)
        return seen["grid"]

    def _gen(*a, **kw):
        seen["generators"] += 1
        return real["build_scenario_generator"](*a, **kw)

    def _shock(**kw):
        seen["shock_batches"] += 1
        return real["apply_shocks_to_base_levels"](**kw)

    def _matrix(**kw):
        seen["matrices"].append({
            k: (np.array(v, copy=True) if k == "flat_levels" else v)
            for k, v in kw.items()
        })
        return real["build_market_data_matrix_from_flat_levels"](**kw)

    def _full(**kw):
        seen["full_shocked"] = kw["shocked_levels"]
        return real["_write_full_reval_block"](**kw)

    def _into(self, market, **kw):
        seen["taylor_market"] = market
        return real_into(self, market, **kw)

    for name, fn in (
        ("convert_scenario_pnl_to_reporting_currency", _convert),
        ("compute_var_es", _var_es),
        ("allocate_pnl_grid", _alloc),
        ("build_scenario_generator", _gen),
        ("apply_shocks_to_base_levels", _shock),
        ("build_market_data_matrix_from_flat_levels", _matrix),
        ("_write_full_reval_block", _full),
    ):
        monkeypatch.setattr(api, name, fn)
    monkeypatch.setattr(SensitivityPnlCalculator, "run_into_grid", _into)
    return seen


def _no_numerics(seen) -> None:
    assert seen["generators"] == 0 and seen["grids"] == 0 and not seen["matrices"]


def _no_scenario_numerics(seen) -> None:
    """Only the base state (S = 1) was valued: no scenario, no grid (P1-B gate)."""
    assert seen["generators"] == 0 and seen["grids"] == 0
    assert [m["num_scenarios"] for m in seen["matrices"]] == [1]


def _rows(run, seen):
    """uid -> (reporting P&L row, settlement dV row, V0) at the production handoff."""
    conv = seen["convert"]
    return {
        int(u): (
            seen["groups"][i],
            conv["settlement_pnl"][i],
            float(conv["base_settlement_values"][i]),
        )
        for i, u in enumerate(run.position_uids)
    }


def _oracle(uid, *, taylor, uids=mf.BASE, data=None):
    base, shocked = mf.world(uids).su.base_levels, mf.shocked_levels(data)
    v0 = float(mf.l1_value(uid, base[:, None], uids)[0])
    dv = (
        mf.l4_taylor(uid, base, shocked, cross=False, uids=uids)
        if taylor
        else mf.l1_value(uid, shocked, uids) - v0
    )
    q0 = float(mf.reporting_rate(uid, base[:, None], uids)[0])
    return v0, dv, mf.l0_reporting(v0, dv, q0, mf.reporting_rate(uid, shocked, uids))


# ---------------------------------------------------------------------------
# A/B. Methodology declaration: explicit U, exact partition, owned arrays
# ---------------------------------------------------------------------------
def _decl(**over):
    kw = dict(
        production_position_uids=[9007, 12, 4002],
        full_reval_uids=[4002],
        taylor_supplied_uids=[],
        taylor_derived_uids=[9007, 12],
    )
    kw.update(over)
    return MethodologyDeclaration(**kw)


def test_declaration_is_canonical_and_order_free() -> None:
    a = _decl()
    b = _decl(production_position_uids=[4002, 9007, 12], taylor_derived_uids=[12, 9007])
    assert a.production_position_uids.tolist() == [12, 4002, 9007]
    assert [x.tolist() for x in a.blocks()] == [[4002], [], [12, 9007]]
    assert a.digest() == b.digest()
    assert (
        _decl(full_reval_uids=[12], taylor_derived_uids=[9007, 4002]).digest()
        != a.digest()
    )


@pytest.mark.parametrize(
    "over,match",
    [
        ({"production_position_uids": [9007, 12, 4002, 73]}, r"no methodology.*\[73\]"),
        ({"production_position_uids": [9007, 12]}, r"not in U.*\[4002\]"),
        ({"production_position_uids": [9007, 12, 4002, 12]}, r"more than once.*\[12\]"),
        ({"taylor_derived_uids": [9007, 12, 9007]}, r"more than once.*\[9007\]"),
        ({"full_reval_uids": [4002, 12]}, r"declared both.*\[12\]"),
        (
            {
                "production_position_uids": [],
                "full_reval_uids": [],
                "taylor_derived_uids": [],
            },
            "empty",
        ),
        ({"full_reval_uids": [4002.0]}, "integer"),
        ({"full_reval_uids": np.array([True])}, "integer"),
        ({"full_reval_uids": [[4002]]}, "1-D"),
    ],
)
def test_declaration_fails_closed(over, match) -> None:
    with pytest.raises(CompositionContractError, match=match):
        _decl(**over)


def test_declaration_owns_its_arrays() -> None:
    caller = np.array([9007, 12, 4002], dtype=np.int64)
    d = _decl(production_position_uids=caller)
    digest = d.digest()
    caller[:] = [7, 8, 9]
    assert (
        d.production_position_uids.tolist() == [12, 4002, 9007] and d.digest() == digest
    )
    for arr in (d.production_position_uids, *d.blocks()):
        with pytest.raises(ValueError):
            arr[...] = 0


# ---------------------------------------------------------------------------
# Wiring, SUPPLIED (P-1), alignment, base coherence: all before numerics
# ---------------------------------------------------------------------------
def test_supplied_is_refused_under_open_policy(spy) -> None:
    inp = mf.inputs()
    inp["methodology"] = MethodologyDeclaration(
        production_position_uids=list(mf.ALL_UIDS),
        full_reval_uids=list(mf.FULL_UIDS),
        taylor_supplied_uids=[2_001],
        taylor_derived_uids=[2_005, 2_006],
    )
    with pytest.raises(CompositionContractError, match="P-1"):
        run_scenario_var_es(**inp)
    inp = mf.inputs()
    inp["taylor_supplied"] = inp["taylor_derived"]
    with pytest.raises(CompositionContractError, match="P-1"):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


@pytest.mark.parametrize("drop,match", [("taylor_derived", "no TAYLOR/DERIVED block")])
def test_declared_partition_without_block_fails(spy, drop, match) -> None:
    inp = mf.inputs()
    inp[drop] = None
    with pytest.raises(CompositionContractError, match=match):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_unused_block_fails(spy) -> None:
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    inp["taylor_derived"] = mf.derived_block(mf.DERIVED_UIDS)
    with pytest.raises(CompositionContractError, match="unused"):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_derived_book_must_price_exactly_u_d(spy) -> None:
    inp = mf.inputs(full=(2_001, 2_002, 2_003, 2_004), derived=(2_005, 2_006))
    inp["taylor_derived"] = mf.derived_block(mf.DERIVED_UIDS)  # book prices 2001 too
    with pytest.raises(
        CompositionContractError, match=r"does not price exactly.*\[2001\]"
    ):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_reporting_metadata_must_be_keyed_to_u(spy) -> None:
    inp = mf.inputs()
    rep = inp["reporting"]
    inp["reporting"] = ReportingSpec(
        reporting_ccy=rep.reporting_ccy,
        position_uids=rep.position_uids[::-1].copy(),
        settlement_ccys=rep.settlement_ccys[::-1],
        fx_factor_names=rep.fx_factor_names,
        fx_factor_rows=rep.fx_factor_rows,
    )
    with pytest.raises(CompositionContractError, match="ascending"):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_mixed_run_refuses_two_bases(spy) -> None:
    inp = mf.inputs()
    inp["taylor_derived"] = mf.derived_block(mf.DERIVED_UIDS, as_of_shift=1)
    with pytest.raises(TaylorContractError, match="as_of_ord32"):
        run_scenario_var_es(**inp)
    _no_scenario_numerics(spy)


def test_uncertified_derived_is_refused_before_any_grid_write(spy) -> None:
    inp = mf.inputs()
    block = inp["taylor_derived"]
    inp["taylor_derived"] = dataclasses.replace(
        block, calculator=dataclasses.replace(block.calculator, taylor_declaration=None)
    )
    with pytest.raises(TaylorContractError, match="UNCERTIFIED"):
        run_scenario_var_es(**inp)
    _no_scenario_numerics(spy)


def test_book_certified_at_another_v0_is_refused_before_any_grid_write(spy) -> None:
    """P1-B: V_0 is run-computed; a book bound to any other V_0 fails closed."""
    inp = mf.inputs()
    inp["taylor_derived"] = mf.derived_block(
        mf.DERIVED_UIDS, tamper=(("base_settlement_values_digest", "0" * 64),)
    )
    with pytest.raises(TaylorContractError, match="base_settlement_values_digest"):
        run_scenario_var_es(**inp)
    _no_scenario_numerics(spy)


# ---------------------------------------------------------------------------
# D. Factor axis: ordered identity with the market-data layout (F-1 included)
# ---------------------------------------------------------------------------
def _axis_case(case, inp):
    ids, types = inp["required_factor_ids"], inp["required_shift_type_id"]
    order = np.arange(ids.shape[0])
    if case == "permuted":
        order = np.r_[order[1], order[0], order[2:]]
    elif case == "missing":
        order = order[:-1]
    for key in (
        "required_factor_ids",
        "required_shift_type_id",
        "source_factor_ids",
        "shock_scale",
        "base_levels",
    ):
        inp[key] = np.ascontiguousarray(inp[key][order])
    if case == "extra":
        inp["required_factor_ids"] = np.append(ids, np.int32(999_999)).astype(np.int32)
    if case == "shift_type":
        wrong = np.array(types, copy=True)
        wrong[0] = np.int8(wrong[0] + 1)
        inp["required_shift_type_id"] = wrong
    return inp


@pytest.mark.parametrize("case", ["permuted", "missing", "extra", "shift_type"])
def test_factor_axis_mismatch_fails_before_numerics(spy, case) -> None:
    inp = _axis_case(case, mf.inputs(full=mf.ALL_UIDS, derived=()))
    with pytest.raises(CompositionContractError, match="universe map"):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_derived_book_on_a_factor_subset_is_refused_before_numerics(spy) -> None:
    """F-1: a book DECLARED on its own 2 referenced factors (not the 4-factor
    run axis) is refused before numerics; the composition never re-maps it."""
    inp = mf.inputs(full=(2_002, 2_003, 2_004, 2_005), derived=(2_001, 2_006))
    inp["taylor_derived"] = mf.derived_block((2_001, 2_006), axis="own")
    with pytest.raises(CompositionContractError, match="factor axis"):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_f1_adapter_builds_the_subset_book_on_the_run_axis(spy) -> None:
    """F-1 resolved at the adapter (D082): a book whose positions depend on 2 of
    the 4 run-axis factors is built by `build_sensitivity_book_spec(
    required_factor_ids=<run axis>)` and prices correctly in a mixed run."""
    inp = mf.inputs(full=(2_002, 2_003, 2_004, 2_005), derived=(2_001, 2_006))
    calc = inp["taylor_derived"].calculator
    assert calc.axis.keys.to_list() == inp["required_factor_ids"].tolist()
    assert calc.book.num_active_factors == 2
    run = run_scenario_var_es(**inp)
    rows = _rows(run, spy)
    for uid in mf.ALL_UIDS:
        _, ref_dv, ref_dw = _oracle(uid, taylor=uid in (2_001, 2_006))
        np.testing.assert_allclose(rows[uid][1], ref_dv, rtol=0, atol=ATOL)
        np.testing.assert_allclose(rows[uid][0], ref_dw, rtol=0, atol=ATOL)


# ---------------------------------------------------------------------------
# C. FULL_REVAL: independent route, whole-portfolio valuation, U_F selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("uids", WORLDS)
def test_all_full_rows_match_l1_and_l0(spy, uids) -> None:
    run = run_scenario_var_es(**mf.inputs(uids=uids, full=uids, derived=()))
    assert run.pnl_provenance is None
    assert run.position_uids.tolist() == sorted(uids)
    assert spy["groups"].shape[0] == len(uids) + 1
    translation = 0.0
    for uid, (dw, dv, v0) in _rows(run, spy).items():
        ref_v0, ref_dv, ref_dw = _oracle(uid, taylor=False, uids=uids)
        assert v0 == pytest.approx(ref_v0, abs=ATOL_V0)
        np.testing.assert_allclose(dv, ref_dv, rtol=0, atol=ATOL)
        np.testing.assert_allclose(dw, ref_dw, rtol=0, atol=ATOL)
        translation = max(
            translation,
            float(
                np.abs(
                    dw - dv * mf.reporting_rate(uid, mf.shocked_levels(), uids)
                ).max()
            ),
        )
    assert min(r[2] for r in _rows(run, spy).values()) < 0.0  # a short: sign exercised
    assert translation > 100 * ATOL  # V0(qs-q0) is material


def test_full_route_never_touches_taylor(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("FULL_REVAL entered a Taylor path")

    for owner, name in (
        (api.FactorAxis, "from_keys"),
        (api, "hash_compiled_book"),
        (api, "_certify_taylor_block"),
        (SensitivityPnlCalculator, "run"),
        (SensitivityPnlCalculator, "run_into_grid"),
    ):
        monkeypatch.setattr(owner, name, boom)
    run = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=()))
    assert np.isfinite(run.var).all()


def test_v0_and_scenarios_share_one_valuation_route(spy) -> None:
    """Every market state is built by the same call with the same mapper,
    universe map, as-of and mode; only the levels differ. First the base at
    S = 1, then the canonical shocked columns in order."""
    inp = mf.inputs(full=mf.ALL_UIDS, derived=(), batch_size=4)
    run_scenario_var_es(**inp)
    calls, shocked = spy["matrices"], mf.shocked_levels()
    assert len(calls) == 1 + -(-shocked.shape[1] // 4)
    first = calls[0]
    for c in calls:
        for key in ("uix_mapper", "universe_map"):
            assert c[key] is first[key]
        assert (
            c["as_of_date"] == first["as_of_date"]
            and c["discounting_mode"] is first["discounting_mode"]
        )
    assert first["num_scenarios"] == 1
    assert np.array_equal(first["flat_levels"][:, 0], inp["base_levels"])
    assert np.array_equal(np.hstack([c["flat_levels"] for c in calls[1:]]), shocked)


def test_undeclared_valuation_uid_fails(spy) -> None:
    universe = tuple(
        u for u in mf.ALL_UIDS if u != 2_006
    )  # portfolio still values 2006
    inp = mf.inputs(full=universe, derived=(), universe=universe)
    with pytest.raises(
        CompositionContractError, match=r"undeclared \(first 10: \[2006\]\)"
    ):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


def test_declared_full_uid_without_valuation_rows_fails(spy) -> None:
    universe = mf.ALL_UIDS + (9_999,)
    inp = mf.inputs(full=universe, derived=(), universe=universe)
    with pytest.raises(
        CompositionContractError, match=r"no valuation row \(first 10: \[9999\]\)"
    ):
        run_scenario_var_es(**inp)
    _no_numerics(spy)


@pytest.mark.parametrize("uids", WORLDS)
def test_whole_portfolio_is_valued_but_only_u_f_enters_block_0(spy, uids) -> None:
    """Methodology ownership comes from the declaration, not from which engine
    happened to compute a value (the FULL pass values U_D positions too)."""
    assert np.array_equal(
        mf.world(uids).su.required_factor_ids, mf.world().su.required_factor_ids
    )
    full, derived = mf.slots(uids, mf.FULL_SLOTS), mf.slots(uids, mf.DERIVED_SLOTS)
    run = run_scenario_var_es(**mf.inputs(uids=uids))
    grid = spy["grid"]
    assert grid.block_uids(0).tolist() == sorted(full)
    assert grid.block_uids(1).tolist() == []
    assert grid.block_uids(2).tolist() == sorted(derived)
    assert run.position_uids.tolist() == sorted(uids)
    assert grid.trade_uids.tolist() != sorted(uids)  # the scatter is non-trivial
    rows = _rows(run, spy)
    leak = 0.0
    for uid in uids:  # row-level mixed oracle
        _, ref_dv, ref_dw = _oracle(uid, taylor=uid in derived, uids=uids)
        np.testing.assert_allclose(rows[uid][1], ref_dv, rtol=0, atol=ATOL)
        np.testing.assert_allclose(rows[uid][0], ref_dw, rtol=0, atol=ATOL)
        if uid in derived:
            leak = max(
                leak,
                float(np.abs(_oracle(uid, taylor=False, uids=uids)[1] - ref_dv).max()),
            )
    assert (
        leak > 100 * ATOL
    )  # identifiable: a FULL value in a DERIVED row would fail above
    np.testing.assert_allclose(
        spy["groups"][-1], spy["groups"][:-1].sum(axis=0), rtol=0, atol=1e-9
    )
    for uid in derived:  # run-computed V0 = L1
        assert rows[uid][2] == pytest.approx(
            _oracle(uid, taylor=True, uids=uids)[0], abs=ATOL_V0
        )
    assert run.pnl_provenance is not None


def test_blocks_are_isolated(spy) -> None:
    """Each block's rows are bitwise the rows its methodology produces alone."""
    mixed = run_scenario_var_es(**mf.inputs())
    m_rows = _rows(mixed, spy)
    alone_full = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=()))
    f_rows = _rows(alone_full, spy)
    sub = mf.build_world(
        mf.positions(mf.BASE).iloc[list(mf.DERIVED_SLOTS)].reset_index(drop=True)
    )
    inp = mf.inputs(full=(), derived=mf.DERIVED_UIDS)  # P1-B: the context IS U
    inp["valuation_context"] = ValuationContext(
        static_config=sub.rsc,
        portfolio=sub.portfolio,
        uix_mapper=sub.mapper,
        discounting_mode=mf.FX.DiscountingMode.NONE,
    )
    alone_derived = run_scenario_var_es(**inp)
    d_rows = _rows(alone_derived, spy)
    for uid in mf.FULL_UIDS:
        assert np.array_equal(m_rows[uid][1], f_rows[uid][1])
    for uid in mf.DERIVED_UIDS:
        assert np.array_equal(m_rows[uid][1], d_rows[uid][1])


def _decomposed(portfolio, uid, splits, permutation=None):
    blk = portfolio.blocks[0]
    u, q, x = [], [], []
    for r in range(blk.size):
        parts = splits if int(blk.trade_uids[r]) == uid else [float(blk.qty[r])]
        for part in parts:
            u.append(int(blk.trade_uids[r]))
            q.append(part)
            x.append(int(blk.effective_spot_price_uix[r]))
    order = np.arange(len(u)) if permutation is None else np.asarray(permutation)
    new = SpotBlock(
        trade_uids=np.asarray(u, np.int64)[order],
        qty=np.asarray(q, np.float64)[order],
        effective_spot_price_uix=np.asarray(x, np.int32)[order],
    )
    return CompiledPortfolio(
        unique_dates_ord32=portfolio.unique_dates_ord32,
        spot=new,
        uix_fingerprint=portfolio.uix_fingerprint,
    )


@pytest.mark.parametrize("permutation", [None, [6, 0, 3, 1, 7, 2, 5, 4]])
def test_decomposition_changes_rows_not_economic_identity(spy, permutation) -> None:
    """Aggregation-bridge test on the D055 row model: UID 2003 (75 WTI) valued
    as three rows 25 + 30 + 20, contiguous or interleaved with other UIDs.
    Still ONE economic row whose V0 is the L1 sum of its rows. (Not a proof
    of any future cashflow product; those stay D054-blocked.)"""
    w = mf.world()
    split = _decomposed(w.portfolio, 2_003, [25.0, 30.0, 20.0], permutation)
    assert split.total_trades == 8 and (split.trade_uids == 2_003).sum() == 3
    whole = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=()))
    base_rows = _rows(whole, spy)
    inp = mf.inputs(full=mf.ALL_UIDS, derived=())
    inp["valuation_context"] = mf.context(split)
    parts = run_scenario_var_es(**inp)
    part_rows = _rows(parts, spy)
    assert parts.position_uids.tolist() == whole.position_uids.tolist()
    assert spy["groups"].shape[0] == len(mf.ALL_UIDS) + 1
    wti0 = float(w.su.base_levels[mf._row(w, "WTI_CRUDE")])
    assert part_rows[2_003][2] == pytest.approx(75.0 * wti0, abs=ATOL_V0)
    for uid in mf.ALL_UIDS:
        np.testing.assert_allclose(
            part_rows[uid][0], base_rows[uid][0], rtol=0, atol=ATOL
        )
    assert parts.identity != whole.identity  # representation is identity


# ---------------------------------------------------------------------------
# E. Scenario identity, batching, ordering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 2, 4, 64])
def test_one_canonical_scenario_stream(spy, batch_size) -> None:
    inp = mf.inputs(batch_size=batch_size)
    run_scenario_var_es(**inp)
    shocked = mf.shocked_levels()
    assert spy["generators"] == 1
    assert spy["shock_batches"] == -(
        -shocked.shape[1] // batch_size
    )  # each column shocked once
    assert np.array_equal(spy["full_shocked"], shocked)  # FULL: canonical levels
    market = spy["taylor_market"]
    assert np.array_equal(market.shocked_levels, shocked)  # DERIVED: the same levels
    assert np.array_equal(market.base_levels, inp["base_levels"])


@pytest.mark.parametrize(
    "partition", [(mf.ALL_UIDS, ()), (mf.FULL_UIDS, mf.DERIVED_UIDS)]
)
def test_batch_size_is_operational(partition) -> None:
    runs = [
        run_scenario_var_es(
            **mf.inputs(full=partition[0], derived=partition[1], batch_size=b)
        )
        for b in (1, 2, 4, 9, 64)
    ]
    for r in runs[1:]:
        assert r.identity == runs[0].identity
        assert [v.hex() for v in r.var.tolist()] == [
            v.hex() for v in runs[0].var.tolist()
        ]
        assert [v.hex() for v in r.es.tolist()] == [
            v.hex() for v in runs[0].es.tolist()
        ]


def test_scenario_order_permutes_full_columns_only() -> None:
    w = mf.world()
    universe = np.array(mf.ALL_UIDS, dtype=np.int64)
    agg = api._row_aggregation(portfolio=w.portfolio, universe=universe)
    route = api._ValuationRoute(
        portfolio=w.portfolio, uix_mapper=w.mapper, universe_map=w.universe_map
    )
    pos = api._positions(agg.economic_uids, universe)
    v0 = api._value_positions(
        route=route,
        aggregation=agg,
        flat_levels=np.ascontiguousarray(w.su.base_levels[:, None]),
    )[pos, 0]
    levels = mf.shocked_levels()
    perm = np.random.default_rng(3).permutation(levels.shape[1])
    outs = []
    for lv in (levels, np.ascontiguousarray(levels[:, perm])):
        buf = np.empty((universe.shape[0], lv.shape[1]), dtype=np.float32)
        api._write_full_reval_block(
            route=route,
            aggregation=agg,
            full_positions=pos,
            v0_full=v0,
            shocked_levels=lv,
            batch_size=3,
            out=buf,
        )
        outs.append(buf)
    assert np.array_equal(outs[0][:, perm], outs[1])


@pytest.mark.parametrize("uids", WORLDS)
def test_declaration_order_does_not_move_rows(spy, uids) -> None:
    full, derived = mf.slots(uids, mf.FULL_SLOTS), mf.slots(uids, mf.DERIVED_SLOTS)
    inp = mf.inputs(uids=uids)
    inp["methodology"] = MethodologyDeclaration(
        production_position_uids=list(reversed(uids)),
        full_reval_uids=list(reversed(full)),
        taylor_supplied_uids=[],
        taylor_derived_uids=[derived[2], derived[0], derived[1]],
    )
    a = run_scenario_var_es(**inp)
    rows_a = _rows(a, spy)
    b = run_scenario_var_es(**mf.inputs(uids=uids))
    rows_b = _rows(b, spy)
    assert a.identity == b.identity and a.position_uids.tolist() == sorted(uids)
    for uid in uids:
        assert np.array_equal(rows_a[uid][0], rows_b[uid][0])


def test_run_does_not_mutate_caller_arrays() -> None:
    inp = mf.inputs()
    keys = (
        "required_factor_ids",
        "required_shift_type_id",
        "source_factor_ids",
        "shock_scale",
        "base_levels",
    )
    before = {k: np.array(inp[k], copy=True) for k in keys}
    run = run_scenario_var_es(**inp)
    for k in keys:
        assert np.array_equal(inp[k], before[k]), k
    with pytest.raises(ValueError):
        run.position_uids[0] = 0


# ---------------------------------------------------------------------------
# G. D081 through the FULL route: one equation, FX once, right direction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "move,expect",
    [
        ({"USDEUR"}, "fx_only"),
        ({"AAPL_EQ", "BRENT_CRUDE"}, "valuation_only"),
        ({"USDEUR", "AAPL_EQ", "BRENT_CRUDE"}, "combined"),
    ],
)
def test_full_reporting_equation_isolated_moves(spy, move, expect) -> None:
    data = mf.history(move=move)
    run = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=(), data=data))
    rows = _rows(run, spy)
    shocked, base = mf.shocked_levels(data), mf.world().su.base_levels
    usdeur = shocked[mf._row(mf.world(), "USDEUR")]
    assert (np.ptp(usdeur) > 0) == (
        expect != "valuation_only"
    )  # the move is what it says
    for uid in mf.ALL_UIDS:
        _, _, ref = _oracle(uid, taylor=False, data=data)
        np.testing.assert_allclose(rows[uid][0], ref, rtol=0, atol=ATOL)
    if expect == "fx_only":  # EUR-settled, USD-quoted: FX cancels in USD reporting
        for uid in (2_002, 2_004):
            np.testing.assert_allclose(rows[uid][0], 0.0, atol=ATOL)
            assert np.abs(rows[uid][1]).max() > 1.0  # while settlement dV moved
    if expect == "valuation_only":  # q_s = q_0: dW = q * dV
        for uid in (2_002, 2_004):
            q = mf.reporting_rate(uid, base[:, None])[0]
            np.testing.assert_allclose(
                rows[uid][0], q * rows[uid][1], rtol=0, atol=ATOL
            )


def test_full_vs_derived_reporting_reconciliation_identity(spy) -> None:
    """dW_F - dW_T = q_s (dV_F - dV_T) per position and scenario: the reporting
    reconciliation IDENTITY of the shared D081 step. Non-vacuous here (the
    truncated book omits the quanto cross term). It is NOT a proof of Taylor
    accuracy; that is the L4 comparison above."""
    full = run_scenario_var_es(**mf.inputs(full=mf.ALL_UIDS, derived=()))
    rf = _rows(full, spy)
    taylor = run_scenario_var_es(**mf.inputs(full=(), derived=mf.ALL_UIDS))
    rt = _rows(taylor, spy)
    shocked, gap = mf.shocked_levels(), 0.0
    for uid in mf.ALL_UIDS:
        qs = mf.reporting_rate(uid, shocked)
        np.testing.assert_allclose(
            rf[uid][0] - rt[uid][0], qs * (rf[uid][1] - rt[uid][1]), rtol=0, atol=ATOL
        )
        assert rf[uid][2] == pytest.approx(
            rt[uid][2], abs=ATOL_V0
        )  # one V0 per economic UID
        gap = max(gap, float(np.abs(rf[uid][1] - rt[uid][1]).max()))
    assert gap > 100 * ATOL
    assert full.identity != taylor.identity  # partition is identity


# ---------------------------------------------------------------------------
# Identity: result-defining composition inputs, not derived results
# ---------------------------------------------------------------------------
def test_composition_is_in_the_identity_and_reproducible() -> None:
    base = run_scenario_var_es(**mf.inputs())
    comp = base.manifest.pnl.composition
    assert set(comp.model_dump()) == {"methodology", "context"}
    assert set(comp.context.model_dump()) == {
        "as_of_ord32",
        "static_config_digest",
        "valuation_contract_fingerprint",
        "discounting_mode",
        "portfolio_digest",
        "uix_mapper_fingerprint",
        "universe_map_digest",
    }  # no V0: a result
    again = run_scenario_var_es(**mf.inputs(), reproduce=base.manifest)
    assert again.identity == base.identity
    moved = mf.inputs(full=(2_003, 2_004, 2_005), derived=(2_001, 2_002, 2_006))
    other = run_scenario_var_es(**moved)
    assert other.identity != base.identity
    from riskquant.solutions.run_identity.api import InputMismatchError

    with pytest.raises(InputMismatchError):
        run_scenario_var_es(**moved, reproduce=base.manifest)


def test_composition_performs_no_factor_realignment(spy, monkeypatch) -> None:
    """Alignment happens at the adapter/compile boundary, never in the run: during
    a mixed run no factor key is relabelled, and the DERIVED calculator receives
    a market on the run axis exactly."""
    inp = mf.inputs(full=(2_002, 2_003, 2_004, 2_005), derived=(2_001, 2_006))

    def boom(*a, **k):
        raise AssertionError("the production run re-keyed factor data")

    monkeypatch.setattr(api.FactorAxis, "relabel", boom)
    run_scenario_var_es(**inp)
    market = spy["taylor_market"]
    assert market.axis.keys.to_list() == inp["required_factor_ids"].tolist()
    assert inp["taylor_derived"].calculator.axis.matches(market.axis)


def test_f1_unsorted_authoritative_run_axis_end_to_end(spy, monkeypatch) -> None:
    """The run axis comes from the REAL `build_universe_mapping` on a static
    config with hostile node ids: [9007, 12, 101, 4002], unsorted and
    non-contiguous. Nothing sorts it. The adapter receives it, the certified
    DERIVED book carries it (SLOT encoding), the run consumes it without any
    re-keying, every mixed row reconciles with the independent oracle on the
    run's canonical levels, and the DERIVED rows equal the SAME certified book
    compiled on the sorted axis and priced on the SAME canonical levels
    (reordered by factor identity in this test)."""
    uids, rf = mf.HOSTILE, mf.HOSTILE_RF
    w = mf.world(uids, rf)
    axis = w.su.required_factor_ids.tolist()
    assert axis == [9007, 12, 101, 4002] and axis != sorted(axis)
    assert w.universe_map.required_factor_ids.tolist() == axis
    inp = mf.inputs(uids=uids, rf=rf)
    assert inp["required_factor_ids"].tolist() == axis  # run keeps the order
    calc = inp["taylor_derived"].calculator
    assert calc.axis.keys.to_list() == axis and calc.axis.slot_map is not None

    def boom(*a, **k):
        raise AssertionError("the production run re-keyed factor data")

    with monkeypatch.context() as trap:  # undone on exit; spy stays
        trap.setattr(api.FactorAxis, "relabel", boom)
        run = run_scenario_var_es(**inp)
    rows = _rows(run, spy)
    market = spy["taylor_market"]
    assert market.axis.keys.to_list() == axis and calc.axis.matches(market.axis)
    shocked, base = mf.shocked_levels(uids=uids, rf=rf), w.su.base_levels
    derived = mf.slots(uids, mf.DERIVED_SLOTS)
    for uid in uids:  # independent oracle, by name
        v0 = float(mf.l1_value(uid, base[:, None], uids, rf)[0])
        dv = (
            mf.l4_taylor(uid, base, shocked, cross=False, uids=uids, rf=rf)
            if uid in derived
            else mf.l1_value(uid, shocked, uids, rf) - v0
        )
        q0 = float(mf.reporting_rate(uid, base[:, None], uids, rf)[0])
        ref = mf.l0_reporting(v0, dv, q0, mf.reporting_rate(uid, shocked, uids, rf))
        assert rows[uid][2] == pytest.approx(v0, abs=ATOL_V0)
        np.testing.assert_allclose(rows[uid][1], dv, rtol=0, atol=ATOL)
        np.testing.assert_allclose(rows[uid][0], ref, rtol=0, atol=ATOL)
    canonical = spy["full_shocked"]  # the run's one buffer
    order = [axis.index(f) for f in sorted(axis)]  # identity reorder, test-side
    twin = mf.derived_block(
        tuple(sorted(derived)), uids=uids, rf=rf, axis="run_sorted"
    ).calculator
    assert twin.axis.keys.to_list() == sorted(axis) and twin.axis.slot_map is None
    res = twin.run(
        twin.axis.levels(
            base_levels=np.ascontiguousarray(base[order]),
            shocked_levels=np.ascontiguousarray(canonical[order]),
        )
    )
    for i, uid in enumerate(res.position_uids.tolist()):  # representation-invariant
        np.testing.assert_allclose(
            rows[uid][1], np.asarray(res.pnl[i], np.float64), rtol=0, atol=ATOL
        )
    assert run.position_uids.tolist() == sorted(uids)


def test_p0_unsorted_run_axis_scenarios_follow_factor_identity(spy) -> None:
    """SimLab P0 reconciliation, on the hostile world that exposed it.

    (a) The run's canonical shocked levels equal an independent log-return
        oracle BY FACTOR NAME in the hostile-id world and in the sorted-id world
        (absolute 1e-5: float32 shocks with |shock| < 0.05 err by <= 4e-9, times
        levels <= 250, so the level error is <= 1e-6).
    (b) Hence the same economics give the same per-position reporting P&L in
        both worlds (float32 ATOL)."""
    uids = mf.HOSTILE
    src_of = {
        "USDEUR": ("USDEUR", 1.0),
        "AAPL_EQ": ("AAPL_EQ", 1.0),
        "BRENT_CRUDE": ("BRENT_CRUDE", 1.0),
        "WTI_CRUDE": ("BRENT_CRUDE", 0.95),
    }
    rows = {}
    for tag, rf in (("hostile", mf.HOSTILE_RF), ("sorted", None)):
        w = mf.world(uids, rf)
        vals, ids = np.asarray(w.historical.values), w.historical.factor_ids.tolist()
        run = run_scenario_var_es(**mf.inputs(uids=uids, rf=rf))
        canonical = spy["full_shocked"]
        for name, (src, scale) in src_of.items():
            col = vals[:, ids.index(w.rf[src])]  # by identity
            want = w.su.base_levels[mf._row(w, name)] * np.exp(
                scale * np.log(col[1:] / col[:-1])
            )
            np.testing.assert_allclose(
                canonical[mf._row(w, name)],
                want,
                rtol=0,
                atol=1e-5,
                err_msg=f"{tag}:{name}",
            )
        rows[tag] = _rows(run, spy)
    for uid in uids:
        np.testing.assert_allclose(
            rows["hostile"][uid][0], rows["sorted"][uid][0], rtol=0, atol=ATOL
        )
