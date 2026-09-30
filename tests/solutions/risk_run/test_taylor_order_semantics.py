# tests/solutions/risk_run/test_taylor_order_semantics.py
"""
PR-05: Taylor order semantics, from declaration to reproduction (D081 §5, D086).

Contract under test:

    declared order  --(compile: validate_taylor_completeness on the spec)-->
    certified pair (declaration, compiled book), coherent wherever the
    calculator is built (verify_declared_order_against_book)
        --> the kernel evaluates exactly the stored terms
        --> the identity names the stored terms (hash_compiled_book)
        --> persistence / reproduce= through that digest

The order is a certified property of the representation, not a separate
execution switch: two orders that store different terms get different
identities; two orders coherent with the SAME stored terms denote the same
function and share one identity (for the admitted SPOT sensitizer,
FIRST_ORDER and SECOND_ORDER_TRUNCATED always coincide: V_ii = 0).

Classification: EXACT (identity, order, bitwise), APPROXIMATE (oracle, bound
derived from the float32 displacement and grid contracts).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reporting_axis import (
    AS_OF,
    NAME,
    NONE,
    RSC,
    World,
    capture,  # noqa: F401  (fixture)
    hf,
    spot,
)

import riskquant.solutions.risk_run.api as api
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import RiskRun, TaylorBlock
from riskquant.solutions.run_identity.api import InputMismatchError
from riskquant.solutions.run_identity.hashing import hash_compiled_book
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
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
from riskquant.valuation.engine import run_valuation

FO, SOT, SOF = (
    TaylorOrder.FIRST_ORDER,
    TaylorOrder.SECOND_ORDER_TRUNCATED,
    TaylorOrder.SECOND_ORDER_FULL,
)

# Uid 2 is a quanto (EQ_USD settled EUR, direct USDEUR leg): F(p) = {S, X},
# one cross pair. Uid 1 has F(p) = {S}. USD reporting: no suffix.
Q_ROWS = [spot(1, "EQ_USD", "USD", 3.0), spot(2, "EQ_USD", "EUR", 2.0)]
Q_BASE = {"EQ_USD": 212.37, "USDEUR": 0.9259}
_R = np.random.default_rng(5)
Q_HIST = {
    n: v * np.exp(np.cumsum(np.r_[0.0, _R.normal(0.0, 0.02, 40)]))
    for n, v in Q_BASE.items()
}


def _quanto_world() -> World:
    return World(Q_ROWS, "USD", Q_BASE, Q_HIST)


def _sensitivities(w: World, uids: list[int]):
    """Production sensitizer output for `uids` at the world's base, and V_0."""
    sub = w.book[w.book.uid.isin(uids)].reset_index(drop=True)
    b = compile_raw_df_into_ingest_batch(raw_df=pl.from_pandas(sub), static_config=RSC)
    m = build_uix_mapper([b], as_of_ord32=AS_OF)
    p = compile_portfolio_from_ingest_batch(ingest_batch=b, uix_mapper=m)
    layout = build_universe_mapping(
        static_config=RSC, uix_mapper=m, discounting_mode=NONE
    )
    levels = np.array([w.base[w.row(NAME[int(f)])] for f in layout.required_factor_ids])
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
    return run_sensitivities(portfolio=p, uix_mapper=m, market_data=mx), out[:, 0]


def _replace_greeks(sens, *, d2=None, cross=None):
    """Same F(p) and rows, different values (a certified but fabricated book;
    P1-A binds base, order and completeness, not Greek values -- D084)."""
    fields: dict[str, Any] = {}
    if d2 is not None:
        fields["second_order"] = np.asarray(d2, dtype=sens.second_order.dtype)
    if cross is not None:
        fields["cross_second_order"] = np.asarray(
            cross, dtype=sens.cross_second_order.dtype
        )
    return dataclasses.replace(sens, **fields)


def _drop_cross(sens):
    fields = {
        f.name: getattr(sens, f.name)[:0]
        for f in dataclasses.fields(sens)
        if f.name.startswith("cross_") and isinstance(getattr(sens, f.name), np.ndarray)
    }
    return dataclasses.replace(sens, **fields)


def certified(w: World, uids: list[int], order: TaylorOrder, sens=None) -> TaylorBlock:
    """A certified TAYLOR/DERIVED block on the valuation prefix, at the run base."""
    base_sens, v0 = _sensitivities(w, uids)
    s = base_sens if sens is None else sens
    if order is not SOF and s.cross_second_order.size:
        s = _drop_cross(s)
    k = w.axis.num_valuation_factors
    book_uids = np.unique(np.asarray(s.position_uids, dtype=np.int64))
    binding = TaylorBaseBinding(
        as_of_ord32=AS_OF,
        base_levels_digest=base_levels_digest(
            factor_ids=w.axis.factor_ids[:k], base_levels=w.base[:k]
        ),
        discounting_mode=NONE.value,
        static_config_digest=static_config_content_digest(RSC),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book_uids, values=v0
        ),
    )
    spec = build_sensitivity_book_spec(
        sensitivities=s, required_factor_ids=np.array(w.axis.factor_ids[:k])
    )
    spec = dataclasses.replace(
        spec,
        taylor_declaration=derive_taylor_declaration(
            sensitivities=s, order=order, base_binding=binding
        ),
    )
    return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


def _run(w: World, block: TaylorBlock, uids=(1, 2), **over: Any) -> RiskRun:
    return run_scenario_var_es(**w.inputs(derived=uids, block=block, **over))


# ---- 1. order domain and declaration/book coherence (EXACT, fail fast) ---------------


@pytest.mark.parametrize("bad", ["FIRST_ORDER", 1, 2, True, None, 1.0])
def test_undeclared_order_is_refused_at_compile_and_at_construction(bad) -> None:
    """The domain is exactly the three TaylorOrder members: a value that merely
    equals one ("FIRST_ORDER" == FIRST_ORDER, True == 1) is not a declaration."""
    w = _quanto_world()
    good = certified(w, [1, 2], FO).calculator
    sens, _ = _sensitivities(w, [1, 2])
    spec = build_sensitivity_book_spec(
        sensitivities=_drop_cross(sens), required_factor_ids=good.axis.keys.to_numpy()
    )
    decl = dataclasses.replace(good.taylor_declaration, order=bad)
    with pytest.raises(TaylorContractError, match="undeclared Taylor order"):
        SensitivityPnlCalculator.compile(
            dataclasses.replace(spec, taylor_declaration=decl)
        )
    with pytest.raises(TaylorContractError, match="undeclared Taylor order"):
        dataclasses.replace(good, taylor_declaration=decl)


@pytest.mark.parametrize(
    "book_order, claimed, match",
    [
        (SOF, FO, "FIRST_ORDER declaration .* cross term"),
        (SOF, SOT, "TRUNCATED declaration omits cross terms"),
        (FO, SOF, "not every unordered pair exactly once"),
        (SOT, SOF, "not every unordered pair exactly once"),
    ],
)
def test_a_declaration_cannot_certify_a_book_whose_terms_it_does_not_describe(
    book_order, claimed, match
) -> None:
    """Mutation E. Before PR-05 each of these re-labelled calculators was
    accepted by the run and priced the book's terms under a false order."""
    w = _quanto_world()
    calc = certified(w, [1, 2], book_order).calculator
    lie = dataclasses.replace(calc.taylor_declaration, order=claimed)
    with pytest.raises(TaylorContractError, match=match):
        dataclasses.replace(calc, taylor_declaration=lie)
    with pytest.raises(TaylorContractError, match=match):
        SensitivityPnlCalculator(book=calc.book, axis=calc.axis, taylor_declaration=lie)


def test_first_order_cannot_certify_non_zero_diagonal_curvature() -> None:
    w = _quanto_world()
    sens, _ = _sensitivities(w, [1, 2])
    calc = certified(
        w, [1, 2], SOT, sens=_replace_greeks(sens, d2=[0.37, -0.21, 0.13])
    ).calculator
    lie = dataclasses.replace(calc.taylor_declaration, order=FO)
    with pytest.raises(TaylorContractError, match="non-zero second-order"):
        dataclasses.replace(calc, taylor_declaration=lie)


def test_coherent_relabels_are_admitted_because_they_denote_the_same_function() -> None:
    """A book storing zero diagonal curvature and no cross term is at once a
    FIRST_ORDER and a SECOND_ORDER_TRUNCATED representation (both true)."""
    w = _quanto_world()
    calc = certified(w, [1, 2], FO).calculator
    as_sot = dataclasses.replace(
        calc, taylor_declaration=dataclasses.replace(calc.taylor_declaration, order=SOT)
    )
    assert hash_compiled_book(as_sot.book) == hash_compiled_book(calc.book)


def test_uncertified_calculator_is_still_constructible_and_refused_by_the_run() -> None:
    """The coherence check applies to certified pairs only; the run's P1-A gate
    still refuses an uncertified book (unchanged)."""
    w = _quanto_world()
    calc = dataclasses.replace(
        certified(w, [1, 2], FO).calculator, taylor_declaration=None
    )
    with pytest.raises(TaylorContractError, match="UNCERTIFIED"):
        _run(w, TaylorBlock(calculator=calc))


# ---- 2. identity semantics (EXACT) ----------------------------------------------------


def test_identity_follows_the_stored_terms() -> None:
    """Mutation A: FIRST vs FULL stores a different term set for the quanto ->
    different book digest, identity and result. FIRST vs TRUNCATED stores the
    same terms for SPOT (V_ii = 0) -> one digest, identity, result_digest."""
    w = _quanto_world()
    runs = {o: _run(w, certified(w, [1, 2], o)) for o in TaylorOrder}
    assert runs[FO].identity != runs[SOF].identity
    assert (
        runs[FO].manifest.pnl.sensitivities_digest
        != runs[SOF].manifest.pnl.sensitivities_digest
    )
    assert runs[FO].var.tobytes() != runs[SOF].var.tobytes()
    assert runs[FO].identity == runs[SOT].identity
    assert runs[FO].result_digest == runs[SOT].result_digest
    assert runs[FO].var.tobytes() == runs[SOT].var.tobytes()


def test_changed_derivative_data_under_the_same_order_changes_identity() -> None:
    """Mutation B: same declared order (TRUNCATED), different diagonal curvature."""
    w = _quanto_world()
    sens, _ = _sensitivities(w, [1, 2])
    a = _run(w, certified(w, [1, 2], SOT))
    b = _run(
        w, certified(w, [1, 2], SOT, sens=_replace_greeks(sens, d2=[0.37, -0.21, 0.13]))
    )
    assert a.identity != b.identity and a.var.tobytes() != b.var.tobytes()


def test_methodology_change_with_numerically_equal_results_changes_identity() -> None:
    """Mutation C (D081 §7, methodology partition): uid 1 is linear, so its
    FULL_REVAL and SECOND_ORDER_FULL rows coincide; the identity still differs."""
    w = _quanto_world()
    derived = _run(w, certified(w, [1], SOF), uids=(1,))
    full = run_scenario_var_es(**w.inputs())
    assert derived.identity != full.identity


def test_representation_detail_that_is_not_economic_keeps_identity() -> None:
    """Mutation D: an independently recompiled, content-identical book."""
    w = _quanto_world()
    assert (
        _run(w, certified(w, [1, 2], SOF)).identity
        == _run(w, certified(w, [1, 2], SOF)).identity
    )


# ---- 3. independent numerical oracle per order (APPROXIMATE) --------------------------


def _oracle(sens, levels, base, row_of, order):
    """sum d1 dx + 1/2 sum d2 dx^2 [+ sum_{i<j} c dx_i dx_j], ABSOLUTE dx rounded
    to float32 once (displacement contract), float64 accumulation. Written from
    D081 §5, not from the kernel."""
    dx = {
        rf: (levels[i] - base[i]).astype(np.float32).astype(np.float64)
        for rf, i in row_of.items()
    }
    out: dict[int, np.ndarray] = {}
    for p, f, d1, d2 in zip(
        sens.position_uids.tolist(),
        sens.factor_rf_ids.tolist(),
        sens.first_order.tolist(),
        sens.second_order.tolist(),
        strict=True,
    ):
        term = d1 * dx[f] + (0.5 * d2 * dx[f] ** 2 if order is not FO else 0.0)
        out[p] = out.get(p, 0.0) + term
    if order is SOF:
        for p, fi, fj, c in zip(
            sens.cross_position_uids.tolist(),
            sens.cross_factor_rf_ids_i.tolist(),
            sens.cross_factor_rf_ids_j.tolist(),
            sens.cross_second_order.tolist(),
            strict=True,
        ):
            out[p] = out[p] + c * dx[fi] * dx[fj]
    return out


@pytest.mark.parametrize("order", list(TaylorOrder))
def test_each_order_executes_exactly_its_terms(capture, order) -> None:
    """Non-zero diagonal curvature and cross term (certified, fabricated values)
    so every term of every order is live. Bound: one float32 ulp of the grid
    value (float64 accumulation order is the only difference)."""
    w = _quanto_world()
    sens, _ = _sensitivities(w, [1, 2])
    fab = _replace_greeks(sens, d2=[0.37, -0.21, 0.13], cross=[0.0071])
    if order is FO:
        fab = _replace_greeks(fab, d2=[0.0, 0.0, 0.0])
    _run(w, certified(w, [1, 2], order, sens=fab))
    levels = np.hstack(capture["shocked"])
    row_of = {int(f): i for i, f in enumerate(w.axis.factor_ids)}
    oracle = _oracle(fab, levels, w.base, row_of, order)
    got = capture["convert_in"]["settlement_pnl"]
    for i, uid in enumerate(w.uids.tolist()):
        exp32 = np.asarray(oracle[uid], dtype=np.float32)
        bound = np.spacing(np.abs(exp32)).astype(np.float64)
        assert (np.abs(got[i] - exp32.astype(np.float64)) <= bound).all(), (order, uid)
    if order is not FO:  # the curvature is live, not dropped
        assert not np.allclose(
            got[0], _oracle(fab, levels, w.base, row_of, FO)[1], atol=0
        )


def test_full_revaluation_rows_do_not_depend_on_the_taylor_order(capture) -> None:
    """FULL_REVAL isolation: uid 1 FULL, uid 2 DERIVED under each order."""
    w = _quanto_world()
    rows = {}
    for order in TaylorOrder:
        capture["shocked"].clear()
        run_scenario_var_es(**w.inputs(derived=(2,), block=certified(w, [2], order)))
        rows[order] = capture["convert_in"]["settlement_pnl"][0].tobytes()
    assert rows[FO] == rows[SOT] == rows[SOF]


# ---- 4. valuation-prefix isolation per order (EXACT) ----------------------------------


@pytest.mark.parametrize("order", list(TaylorOrder))
def test_reporting_suffix_poison_never_reaches_taylor(monkeypatch, order) -> None:
    rows = [spot(1, "EQ_USD", "USD", 3.0), spot(2, "EQ_USD", "USD", -1.0)]
    base = {"EQ_USD": 100.0, "USDEUR": 0.90}
    hist = {
        n: v * np.exp(np.cumsum(np.r_[0.0, _R.normal(0.0, 0.02, 11)]))
        for n, v in base.items()
    }
    w = World(rows, "EUR", base, hist)
    k = w.axis.num_valuation_factors
    assert w.axis.reporting_factor_ids.size == 1
    block = certified(w, [1, 2], order)
    assert block.calculator.axis.keys.to_list() == w.axis.factor_ids[:k].tolist()

    def settled(poison):
        seen: dict[str, Any] = {}
        real_apply, real_conv = (
            api.apply_linear_terms_to_base_levels,
            api.convert_scenario_pnl_to_reporting_currency,
        )

        def apply(**kw: Any) -> None:
            real_apply(**kw)
            if poison is not None:
                kw["out"][k:, :] = poison

        def conv(**kw: Any):
            seen["dv"] = np.array(kw["settlement_pnl"])
            return real_conv(**kw)

        monkeypatch.setattr(api, "apply_linear_terms_to_base_levels", apply)
        monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", conv)
        run_scenario_var_es(**w.inputs(derived=(1, 2), block=block))
        monkeypatch.undo()
        return seen["dv"]

    assert settled(None).tobytes() == settled(0.5).tobytes()


# ---- 5. scenario worlds and batching (EXACT) ------------------------------------------

_WORLDS = {
    "historical": ({"model": "historical", "horizon": 1}, "matrix"),
    "sbb": (
        {
            "model": "sbb",
            "horizon": 1,
            "total_scenarios": 200,
            "expected_block_size": 4.0,
            "seed": 3,
        },
        "matrix",
    ),
    "filtered_bootstrap": (
        {
            "model": "filtered_bootstrap",
            "horizon": 1,
            "total_scenarios": 200,
            "seed": 3,
        },
        "matrix",
    ),
    "parametric": (
        {
            "model": "parametric",
            "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
            "total_scenarios": 200,
            "distribution": {"kind": "standard_normal"},
            "seed": 3,
        },
        "parametric",
    ),
}


def _data(w: World, kind: str):
    if kind == "matrix":
        return w.data()
    ids = np.sort(w.axis.factor_ids).astype(np.int32)
    return ParametricRiskData(
        factor_ids=ids,
        cholesky_matrix=np.ascontiguousarray(
            np.linalg.cholesky(np.array([[1.0, 0.4], [0.4, 1.0]]))
        ),
        volatilities=np.array([0.015, 0.006]),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(ids.shape[0], SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


@pytest.mark.parametrize("world", list(_WORLDS))
def test_order_semantics_hold_in_every_scenario_world(world) -> None:
    w = _quanto_world()
    cfg, kind = _WORLDS[world]
    over = {"scenario_config": parse_scenario_model_config(cfg), "data": _data(w, kind)}
    r = {o: _run(w, certified(w, [1, 2], o), **over) for o in TaylorOrder}
    assert (
        r[FO].identity != r[SOF].identity
        and r[FO].var.tobytes() != r[SOF].var.tobytes()
    )
    assert (
        r[FO].identity == r[SOT].identity
        and r[FO].var.tobytes() == r[SOT].var.tobytes()
    )
    for order in TaylorOrder:  # batching is operational under every order (D041)
        b = _run(w, certified(w, [1, 2], order), **over, batch_size=3)
        assert (
            b.identity == r[order].identity
            and b.var.tobytes() == r[order].var.tobytes()
        )
        assert b.es.tobytes() == r[order].es.tobytes()


# ---- 6. persistence and reproduction (EXACT) ------------------------------------------


@pytest.mark.parametrize("order", list(TaylorOrder))
def test_persisted_run_round_trips_and_reproduces_under_its_order(order) -> None:
    w = _quanto_world()
    run = _run(w, certified(w, [1, 2], order))
    back = RiskRun.from_json(run.to_json())
    assert back.identity == run.identity and back.result_digest == run.result_digest
    again = _run(w, certified(w, [1, 2], order), reproduce=back.manifest)
    assert (
        again.var.tobytes() == run.var.tobytes()
        and again.es.tobytes() == run.es.tobytes()
    )


def test_reproducing_with_a_different_term_set_is_refused_before_numerics() -> None:
    w = _quanto_world()
    full = _run(w, certified(w, [1, 2], SOF))
    with pytest.raises(InputMismatchError):
        _run(w, certified(w, [1, 2], FO), reproduce=full.manifest)


def test_the_artifact_carries_no_separate_order_claim() -> None:
    """No field can claim an order other than the one the digested book
    realises; an equivalent certified book reproduces bitwise."""
    w = _quanto_world()
    first = _run(w, certified(w, [1, 2], FO))
    import json

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    assert not [
        k for k in keys(json.loads(first.to_json())) if k in ("order", "taylor_order")
    ]
    again = _run(w, certified(w, [1, 2], SOT), reproduce=first.manifest)
    assert (
        again.identity == first.identity and again.var.tobytes() == first.var.tobytes()
    )
