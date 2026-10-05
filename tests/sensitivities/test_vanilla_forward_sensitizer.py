# tests/sensitivities/test_vanilla_forward_sensitizer.py
"""
WS-C (D103): the VANILLA_FORWARD sensitizer on the production compile path.

Every book is compiled by production ingest, `build_uix_mapper` and the
canonical portfolio compiler; market grids come from
`build_market_data_matrix_from_flat_levels`. The oracle (`_oracle_*` below) is
test-owned: it reads levels BY NAME and evaluates the section 4 contract with
scalars and dictionaries. It does not read the effective-factor quads, the
folds, the kernel or the sensitizer.

    PV = Q * F * DF_a * X_a - Q * K * DF_k * X_k
    dPV/dF = Q DF_a X_a ; dPV/dX_a = Q DF_a F ; dPV/dX_k = -Q K DF_k ; d2PV/dF dX_a = Q DF_a

Tolerance (revision 8, §5.12): coefficients within 2^-49 * A_bar, with A_bar the
sum of the absolute contributions of the coefficient.

Topology book (UIDs sparse, not ascending; long and short of each topology):

    {F}             TEST_EXACT_SIMPLE[1M]  USD strike, USD settle
    {F, X_a}        TEST_EXACT_SIMPLE[3M]  EUR strike, EUR settle    X_a = USDEUR
    {F, X_k}        TEST_EXACT_EUR[1M]     USD strike, EUR settle    X_k = USDEUR
    {F, X_a, X_k}   TEST_EXACT_DIFF[3M]    EUR strike, GBP settle    USDGBP, EURGBP
    {F, X}          BRENT/WTI_CRUDE_FWD[M+0] USD strike, EUR settle  X_a = X_k = USDEUR

The CURVE check (`test_curve_matrix_discount_factors_attach_to_their_legs`) is a
FORMULA CHECK ONLY: it makes DF_a and DF_k observable. It is not CURVE
admission; D103.4b is unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solutions" / "risk_run"))
import _fwdfix as fx

import riskquant.sensitivities.sensitizers.vanilla_forward as sensitizer_module
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.constants import INVALID_UIX
from riskquant.exceptions import PrettyKeyError
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizers.vanilla_forward import (
    sensitize_vanilla_forward_block,
)
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.book_adapter import (
    build_sensitivity_book_spec,
)
from riskquant.utils.dates import date_to_ord32

NONE, CURVE = DiscountingMode.NONE, DiscountingMode.CURVE
TOL = 2.0**-49
D1 = dt.date(2026, 3, 16)
_EUR_CURVE = (250_015_000, "TEST_EXACT_EUR", ((16_021, "[1M]"),))

LEVELS: dict[str, float] = {
    **fx.BASE_LEVELS,
    "TEST_EXACT_SIMPLE[3M]": 148.0,
    "TEST_EXACT_EUR[1M]": 57.3,
    "USDGBP": 0.7891,
    "EURGBP": 0.8637,
}


def _static_config_dfs() -> dict[str, pl.DataFrame]:
    dfs = fx.static_config_dfs()
    cm, cp = dfs["curve_meta"], dfs["curve_point_meta"]
    curve_id, name, pts = _EUR_CURVE
    row: dict[str, object] = dict(
        cm.filter(pl.col("curve_name") == "TEST_EXACT_SIMPLE").row(0, named=True)
    )
    row.update(
        curve_id=curve_id, curve_name=name, quote_ccy="EUR", shift_type="LOG_RETURN"
    )
    dfs["curve_meta"] = pl.concat([cm, pl.DataFrame([row], schema=cm.schema)])
    points = [{"rf_id": rf, "curve_id": curve_id, "tenor_token": t} for rf, t in pts]
    dfs["curve_point_meta"] = pl.concat([cp, pl.DataFrame(points, schema=cp.schema)])
    return dfs


RSC = compile_risk_static_config_from_dict_of_dfs(_static_config_dfs())
RF = dict(RSC.rel_name_to_rf_id)
NAME = {v: k for k, v in RF.items()}
CURVE_ID = dict(
    zip(
        _static_config_dfs()["curve_meta"]["curve_name"],
        _static_config_dfs()["curve_meta"]["curve_id"],
        strict=True,
    )
)
_CURVE_OF_CCY = {"USD": "USD_SOFR_OIS", "EUR": "EUR_ESTR_OIS", "GBP": "GBP_SONIA_OIS"}


def _fwd(
    uid,
    und,
    und_ccy,
    qty,
    strike,
    k_ccy,
    settle,
    delivery=D1,
    settlement=None,
    *,
    under_curve=None,
    strike_curve=None,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "instrument": "VANILLA_FORWARD_NODE",
        "quantity": qty,
        "underlying": und,
        "settlement_ccy": settle,
        "strike_price": strike,
        "strike_ccy": k_ccy,
        "delivery_date": delivery,
        "settlement_date": settlement or (delivery or D1) + dt.timedelta(days=2),
        "under_discount_curve": under_curve or _CURVE_OF_CCY[und_ccy],
        "strike_discount_curve": strike_curve or _CURVE_OF_CCY[k_ccy],
    }


#: uid -> (row, oracle view (fname, xa_name, xk_name)).
TOPOLOGY = {
    9_101: (
        _fwd(9_101, "TEST_EXACT_SIMPLE[1M]", "USD", 120.0, 97.0, "USD", "USD"),
        ("TEST_EXACT_SIMPLE[1M]", None, None),
    ),
    31: (
        _fwd(31, "TEST_EXACT_SIMPLE[1M]", "USD", -45.0, 99.0, "USD", "USD"),
        ("TEST_EXACT_SIMPLE[1M]", None, None),
    ),
    70_002: (
        _fwd(70_002, "TEST_EXACT_SIMPLE[3M]", "USD", 80.0, 150.0, "EUR", "EUR"),
        ("TEST_EXACT_SIMPLE[3M]", "USDEUR", None),
    ),
    5: (
        _fwd(5, "TEST_EXACT_SIMPLE[3M]", "USD", -33.0, 140.0, "EUR", "EUR"),
        ("TEST_EXACT_SIMPLE[3M]", "USDEUR", None),
    ),
    818: (
        _fwd(818, "TEST_EXACT_EUR[1M]", "EUR", 64.0, 55.0, "USD", "EUR"),
        ("TEST_EXACT_EUR[1M]", None, "USDEUR"),
    ),
    4_040: (
        _fwd(4_040, "TEST_EXACT_EUR[1M]", "EUR", -21.0, 59.0, "USD", "EUR"),
        ("TEST_EXACT_EUR[1M]", None, "USDEUR"),
    ),
    66: (
        _fwd(66, "TEST_EXACT_DIFF[3M]", "USD", 90.0, 40.0, "EUR", "GBP"),
        ("TEST_EXACT_DIFF[3M]", "USDGBP", "EURGBP"),
    ),
    12_345: (
        _fwd(12_345, "TEST_EXACT_DIFF[3M]", "USD", -17.0, 41.0, "EUR", "GBP"),
        ("TEST_EXACT_DIFF[3M]", "USDGBP", "EURGBP"),
    ),
    777: (
        _fwd(
            777,
            "BRENT_CRUDE_FWD[M+0]",
            "USD",
            60.0,
            80.0,
            "USD",
            "EUR",
            dt.date(2026, 2, 16),
        ),
        ("BRENT_CRUDE_FWD[M+0]", "USDEUR", "USDEUR"),
    ),
    2: (
        _fwd(
            2,
            "WTI_CRUDE_FWD[M+0]",
            "USD",
            -90.0,
            76.0,
            "USD",
            "EUR",
            dt.date(2026, 2, 16),
        ),
        ("WTI_CRUDE_FWD[M+0]", "USDEUR", "USDEUR"),
    ),
}


# ---- production compile path ------------------------------------------------------------


def _build(rows, *, mode=NONE, as_of=fx.AS_OF, columns=None) -> SimpleNamespace:
    """Production ingest -> mapper -> portfolio -> layout -> market grids.
    `columns`: list of {name: level} overrides, one per scenario column."""
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(pd.DataFrame(list(rows))),
        static_config=RSC,
        as_of_date=as_of,
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(as_of)))
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=RSC, uix_mapper=mapper, discounting_mode=mode
    )
    columns = columns or [{}]

    def level(rf: int, over: dict[str, float]) -> float:
        name = NAME[int(rf)]
        if name in over:
            return over[name]
        return LEVELS[name]

    flat = np.array(
        [[level(rf, over) for over in columns] for rf in layout.required_factor_ids],
        dtype=np.float64,
    )
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=mode,
        as_of_date=as_of,
        uix_mapper=mapper,
        universe_map=layout,
        flat_levels=np.ascontiguousarray(flat),
        num_scenarios=len(columns),
    )
    return SimpleNamespace(
        batch=batch,
        mapper=mapper,
        portfolio=portfolio,
        layout=layout,
        matrix=matrix,
        block=portfolio.vanilla_forward,
        as_of=as_of,
    )


def _sens(w: SimpleNamespace, scenario: int = 0) -> BlockSensitivities:
    return sensitize_vanilla_forward_block(
        block=w.block, uix_mapper=w.mapper, market_data=w.matrix, scenario=scenario
    )


@pytest.fixture(scope="module")
def topo() -> SimpleNamespace:
    w = _build([row for row, _ in TOPOLOGY.values()])
    w.sens = _sens(w)
    return w


# ---- test-owned oracle -----------------------------------------------------------------


def _oracle(
    q: float, k: float, view, lv: dict[str, float], df_a: float = 1.0, df_k: float = 1.0
) -> tuple[dict, dict]:
    """diag {rf: [d1, d2, A_bar]} and cross {(lo, hi): [v, A_bar]} of one row,
    over the COMPLETE factor set, from levels by name."""
    fname, xa_name, xk_name = view
    f = lv[fname]
    x_a = 1.0 if xa_name is None else lv[xa_name]
    terms = [(RF[fname], q * df_a * x_a)]
    if xa_name is not None:
        terms.append((RF[xa_name], q * df_a * f))
    if xk_name is not None:
        terms.append((RF[xk_name], -q * k * df_k))
    diag: dict[int, list[float]] = {}
    for rf, v in terms:
        d = diag.setdefault(rf, [0.0, 0.0, 0.0])
        d[0] += v
        d[2] += abs(v)
    cross: dict[tuple[int, int], list[float]] = {
        pair: [0.0, 0.0] for pair in itertools.combinations(sorted(diag), 2)
    }
    if xa_name is not None:
        w = q * df_a
        a, b = RF[fname], RF[xa_name]
        if a == b:
            diag[a][1] += 2.0 * w
            diag[a][2] += abs(2.0 * w)
        else:
            cross[(min(a, b), max(a, b))][0] += w
            cross[(min(a, b), max(a, b))][1] += abs(w)
    return diag, cross


def _emitted(s: BlockSensitivities, uid: int) -> tuple[dict, dict]:
    m = s.position_uids == uid
    diag = {
        int(r): (float(d1), float(d2))
        for r, d1, d2 in zip(
            s.factor_rf_ids[m], s.first_order[m], s.second_order[m], strict=True
        )
    }
    c = s.cross_position_uids == uid
    cross = {
        (int(i), int(j)): float(v)
        for i, j, v in zip(
            s.cross_factor_rf_ids_i[c],
            s.cross_factor_rf_ids_j[c],
            s.cross_second_order[c],
            strict=True,
        )
    }
    return diag, cross


def _assert_matches(s: BlockSensitivities, uid: int, oracle: tuple[dict, dict]) -> None:
    diag, cross = _emitted(s, uid)
    o_diag, o_cross = oracle
    assert set(diag) == set(o_diag) and set(cross) == set(o_cross), uid
    for rf, (d1, d2, a_bar) in o_diag.items():
        assert abs(diag[rf][0] - d1) <= TOL * a_bar, (uid, rf)
        assert abs(diag[rf][1] - d2) <= TOL * a_bar, (uid, rf)
    for pair, (v, a_bar) in o_cross.items():
        assert abs(cross[pair] - v) <= TOL * a_bar, (uid, pair)


def _assert_p_n(s: BlockSensitivities, w: SimpleNamespace) -> None:
    """P-n on the compiled book: n_m = d_m + c_m <= 32."""
    spec = build_sensitivity_book_spec(
        sensitivities=s,
        required_factor_ids=np.asarray(w.layout.required_factor_ids, dtype=np.int32),
    )
    book = SensitivityPnlCalculator.compile(spec).book
    n_m = np.diff(book.position_offsets) + np.diff(book.cross_offsets)
    assert n_m.size and int(n_m.max()) <= 32


# ---- C1 to C5, C9, C13, C16 -------------------------------------------------------------


@pytest.mark.parametrize("uid", sorted(TOPOLOGY))
def test_c1_c5_coefficients_equal_the_contract(topo, uid: int) -> None:
    row, view = TOPOLOGY[uid]
    _assert_matches(
        topo.sens, uid, _oracle(row["quantity"], row["strike_price"], view, LEVELS)
    )


def test_c1_c5_quantity_sign_reaches_every_first_order_term(topo) -> None:
    """Long and short of each topology: every first-order term has the sign of
    its own oracle term (levels, DF and FX are positive; K enters negatively)."""
    for uid, (row, view) in TOPOLOGY.items():
        diag, _ = _emitted(topo.sens, uid)
        o_diag, _ = _oracle(row["quantity"], row["strike_price"], view, LEVELS)
        for rf, (d1, _, _) in o_diag.items():
            assert np.sign(diag[rf][0]) == np.sign(d1) != 0.0, (uid, rf)
    signs = {np.sign(TOPOLOGY[u][0]["quantity"]) for u in TOPOLOGY}
    assert signs == {-1.0, 1.0}


def test_c9_keys_are_global_rf_ids_never_uixs(topo) -> None:
    expected = {RF[n] for _, view in TOPOLOGY.values() for n in view if n is not None}
    keys = set(topo.sens.factor_rf_ids.tolist())
    assert keys == expected
    uixs = {int(np.searchsorted(topo.mapper.market_id_universe, rf)) for rf in keys}
    assert keys != uixs  # a uix-keyed emission would not equal the rf set


def test_c13_shared_fx_leg_merges_into_one_row(topo) -> None:
    for uid in (777, 2):
        diag, cross = _emitted(topo.sens, uid)
        assert sorted(diag) == sorted({RF[TOPOLOGY[uid][1][0]], RF["USDEUR"]})
        assert all(d2 == 0.0 for _, d2 in diag.values())
        assert len(cross) == 1 and all(i != j for i, j in cross)


def test_c16_emitted_keys_are_unique_and_canonical(topo) -> None:
    s = topo.sens
    diag = list(zip(s.position_uids.tolist(), s.factor_rf_ids.tolist(), strict=True))
    cross = list(
        zip(
            s.cross_position_uids.tolist(),
            s.cross_factor_rf_ids_i.tolist(),
            s.cross_factor_rf_ids_j.tolist(),
            strict=True,
        )
    )
    assert len(diag) == len(set(diag)) and len(cross) == len(set(cross))
    assert all(i < j for _, i, j in cross)


def test_p_n_holds_on_the_compiled_topology_book(topo) -> None:
    _assert_p_n(topo.sens, topo)


# ---- C6, C15, C14 ------------------------------------------------------------------------


def test_c6_zero_quantity_keeps_the_full_topology(topo) -> None:
    zero = dataclasses.replace(topo.block, qty=np.zeros_like(topo.block.qty))
    s = sensitize_vanilla_forward_block(
        block=zero, uix_mapper=topo.mapper, market_data=topo.matrix
    )
    assert np.array_equal(s.position_uids, topo.sens.position_uids)
    assert np.array_equal(s.factor_rf_ids, topo.sens.factor_rf_ids)
    assert np.array_equal(s.cross_factor_rf_ids_i, topo.sens.cross_factor_rf_ids_i)
    assert not np.any(s.first_order) and not np.any(s.second_order)
    assert not np.any(s.cross_second_order)
    _assert_p_n(s, topo)


def test_c15_empty_block_gives_empty_valid_sensitivities(topo) -> None:
    fields = {
        f.name: getattr(topo.block, f.name)[:0]
        for f in dataclasses.fields(topo.block)
        if isinstance(getattr(topo.block, f.name), np.ndarray)
    }
    empty = dataclasses.replace(topo.block, **fields)
    s = sensitize_vanilla_forward_block(
        block=empty, uix_mapper=topo.mapper, market_data=topo.matrix
    )
    assert s.position_uids.size == 0 and s.cross_position_uids.size == 0


def test_c14_same_rf_product_doubles_the_diagonal(topo) -> None:
    """Hand-built identity: the asset-leg FX pair resolves to F's own rf, so the
    (F, X_a) product is w x^2 and d2PV/dx2 = 2 w."""
    uid = 70_002
    row, (fname, xa_name, _) = TOPOLOGY[uid]
    mapper = copy.copy(topo.mapper)
    universe = np.array(topo.mapper.market_id_universe, copy=True)
    universe[int(np.searchsorted(universe, RF[xa_name]))] = RF[fname]
    object.__setattr__(mapper, "market_id_universe", universe)
    s = sensitize_vanilla_forward_block(
        block=topo.block, uix_mapper=mapper, market_data=topo.matrix
    )
    diag, cross = _emitted(s, uid)
    q = row["quantity"]
    assert list(diag) == [RF[fname]] and cross == {}
    d1, d2 = diag[RF[fname]]
    a_bar = abs(q * LEVELS[xa_name]) + abs(q * LEVELS[fname])
    assert abs(d1 - (q * LEVELS[xa_name] + q * LEVELS[fname])) <= TOL * a_bar
    assert d2 == 2.0 * q


# ---- C7, C8: identity of F --------------------------------------------------------------


def _f_key(w: SimpleNamespace, uid: int) -> int:
    s = _sens(w)
    rows = np.flatnonzero(w.block.trade_uids == uid)
    via_quads = w.mapper.market_id_universe[
        w.mapper.efp_effective_quads[w.block.effective_forward_price_uix[rows], 0]
    ]
    ingest = w.batch.underlying_ids[np.asarray(w.batch.uids) == uid]
    assert len(set(via_quads.tolist()) | set(ingest.tolist())) == 1
    f_rows = s.factor_rf_ids[
        (s.position_uids == uid) & (s.factor_rf_ids == int(ingest[0]))
    ]
    assert f_rows.size == 1
    return int(ingest[0])


def test_c7a_c8a_exact_node_key_and_output_do_not_depend_on_delivery() -> None:
    row = TOPOLOGY[9_101][0]
    variants = [
        row,
        {**row, "delivery_date": None},
        {**row, "delivery_date": dt.date(2026, 5, 15)},
        {**row, "delivery_date": dt.date(2026, 7, 15)},
    ]
    worlds = [_build([v]) for v in variants]
    assert {_f_key(w, 9_101) for w in worlds} == {RF["TEST_EXACT_SIMPLE[1M]"]}
    present, absent = _sens(worlds[0]), _sens(worlds[1])
    for f in dataclasses.fields(BlockSensitivities):
        assert np.array_equal(getattr(present, f.name), getattr(absent, f.name)), f.name


def test_c7b_c8b_relative_node_key_is_the_ingest_resolved_rf(topo) -> None:
    assert _f_key(topo, 777) == RF["BRENT_CRUDE_FWD[M+0]"]


def test_c7c_relative_node_rf_is_fixed_by_name_across_as_of_dates() -> None:
    as_of_2 = dt.date(2026, 2, 17)
    m1 = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=RSC, as_of_date=fx.AS_OF
    )
    m2 = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=RSC, as_of_date=as_of_2
    )
    name = "BRENT_CRUDE_FWD[M+0]"
    assert m1.get_abs_name_for_rel_name(name) != m2.get_abs_name_for_rel_name(name)
    first = _build([TOPOLOGY[777][0]])
    second = _build(
        [{**TOPOLOGY[777][0], "delivery_date": dt.date(2026, 3, 16)}], as_of=as_of_2
    )
    assert _f_key(first, 777) == _f_key(second, 777) == RF[name]
    with pytest.raises(
        PrettyKeyError
    ):  # the first contract's delivery is outside March
        _build([TOPOLOGY[777][0]], as_of=as_of_2)


# ---- C10: D103.3 refusal before any level gathering and before the kernel ----------------


class _NoLevels:
    """Market data that fails on any read: the refusal must precede every read."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"market data read before the D103.3 refusal: {name}")


@pytest.mark.parametrize(
    "row",
    [
        _fwd(
            41, "TEST_EXACT_EUR[1M]", "EUR", 10.0, 50.0, "USD", "USD"
        ),  # asset leg EUR -> USD
        _fwd(
            42, "TEST_EXACT_SIMPLE[1M]", "USD", 10.0, 50.0, "EUR", "USD"
        ),  # strike leg EUR -> USD
    ],
    ids=["asset_leg", "strike_leg"],
)
def test_c10_inverted_fx_is_refused_before_levels_and_kernel(row, monkeypatch) -> None:
    w = _build([row])
    quads = w.mapper.efp_effective_quads[w.block.effective_forward_price_uix]
    triples = w.mapper.edf_effective_triples[w.block.effective_discount_factor_uix]
    leg = int(quads[0, 3] if row["uid"] == 41 else triples[0, 2])
    assert int(w.mapper.fx_effective_pairs[leg, 1]) == 1  # the intended leg is inverted
    calls = []
    monkeypatch.setattr(
        sensitizer_module,
        "_kernel_sensitize_vanilla_forward",
        lambda *a: calls.append(a),
    )
    with pytest.raises(NotImplementedError, match=r"^D103\.3"):
        sensitize_vanilla_forward_block(
            block=w.block, uix_mapper=w.mapper, market_data=_NoLevels()
        )
    assert calls == []


# ---- C11: index binding and scenario selection ------------------------------------------


def test_c11_kernel_receives_the_production_fold_indices(topo, monkeypatch) -> None:
    seen = []
    real = sensitizer_module._kernel_sensitize_vanilla_forward
    monkeypatch.setattr(
        sensitizer_module,
        "_kernel_sensitize_vanilla_forward",
        lambda *a: (
            seen.append([
                np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a
            ]),
            real(*a),
        ),
    )
    _sens(topo)
    args = seen[0]
    price, curve_a, date_a, fx_a, curve_k, date_k = args[6:12]
    quads = topo.mapper.efp_effective_quads[topo.block.effective_forward_price_uix]
    triples = topo.mapper.edf_effective_triples[
        topo.block.effective_discount_factor_uix
    ]
    assert np.array_equal(np.c_[price, curve_a, date_a, fx_a], quads)
    assert np.array_equal(np.c_[curve_k, date_k], triples[:, :2])
    universe = topo.mapper.market_id_universe
    for r, uid in enumerate(topo.block.trade_uids.tolist()):
        fname, xa_name, _ = TOPOLOGY[uid][1]
        assert int(universe[price[r]]) == RF[fname]
        if xa_name is not None:
            assert (
                int(universe[topo.mapper.fx_effective_pairs[fx_a[r], 0]]) == RF[xa_name]
            )
    m = topo.matrix
    efp = assemble_valuation_state(
        matrix=m, uix_mapper=topo.mapper
    ).effective_forward_prices
    has = fx_a != INVALID_UIX
    x = np.where(has, m.fx_spots_effective[np.where(has, fx_a, 0), 0], 1.0)
    rebuilt = m.forward_nodes[price, 0] * m.discount_factor_grid[curve_a, date_a, 0] * x
    ref = efp[topo.block.effective_forward_price_uix, 0]
    assert np.all(np.abs(rebuilt - ref) <= 2.0**-50 * np.abs(ref))


def test_c11_scenario_column_is_respected() -> None:
    rows = [TOPOLOGY[u][0] for u in (66, 777)]
    bumped = {n: 1.07 * v for n, v in LEVELS.items()}
    w = _build(rows, columns=[{}, bumped])
    for scenario, lv in ((0, LEVELS), (1, bumped)):
        s = _sens(w, scenario)
        for uid in (66, 777):
            row, view = TOPOLOGY[uid]
            _assert_matches(
                s, uid, _oracle(row["quantity"], row["strike_price"], view, lv)
            )


# ---- C18: Proxy V1 Run A / Run B contexts ------------------------------------------------


def test_c18_sensitizer_has_no_beta_and_uses_the_target_level() -> None:
    """(a) the Run A and Run B contexts give bitwise equal output; (b) the F4
    X-row uses x_A,0, not x_D,0; (c) F4 rows are keyed by rf_A, never rf_D."""
    rows = tuple(r for r in fx.ROWS if r["uid"] != fx.F5)
    runs = []
    for _ in ("A", "B"):
        w = fx.World(rows=rows)
        layout = build_universe_mapping(
            static_config=w.rsc, uix_mapper=w.mapper, discounting_mode=NONE
        )
        name_of = {v: k for k, v in w.rf.items()}
        flat = np.array([
            [fx.BASE_LEVELS[name_of[int(f)]]] for f in layout.required_factor_ids
        ])
        matrix = build_market_data_matrix_from_flat_levels(
            discounting_mode=NONE,
            as_of_date=fx.AS_OF,
            uix_mapper=w.mapper,
            universe_map=layout,
            flat_levels=np.ascontiguousarray(flat),
            num_scenarios=1,
        )
        runs.append(
            sensitize_vanilla_forward_block(
                block=w.context.portfolio.vanilla_forward,
                uix_mapper=w.mapper,
                market_data=matrix,
            )
        )
    for f in dataclasses.fields(BlockSensitivities):
        a, b = getattr(runs[0], f.name), getattr(runs[1], f.name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f.name
    s = runs[0]
    rf_a, rf_d, rf_x = (
        RF["WTI_CRUDE_FWD[M+0]"],
        RF["BRENT_CRUDE_FWD[M+0]"],
        RF["USDEUR"],
    )
    diag, _ = _emitted(s, fx.F4)
    assert set(diag) == {rf_a, rf_x} and rf_d not in diag
    q, k = -90.0, 76.0
    x_a0, x_d0 = (
        fx.BASE_LEVELS["WTI_CRUDE_FWD[M+0]"],
        fx.BASE_LEVELS["BRENT_CRUDE_FWD[M+0]"],
    )
    a_bar = abs(q * x_a0) + abs(q * k)
    assert abs(diag[rf_x][0] - q * (x_a0 - k)) <= TOL * a_bar
    assert abs(diag[rf_x][0] - q * (x_d0 - k)) > 1e6 * TOL * a_bar


# ---- CURVE: formula check only; not CURVE admission; D103.4b unchanged -------------------


def test_curve_matrix_discount_factors_attach_to_their_legs() -> None:
    """FORMULA CHECK ONLY; NOT CURVE ADMISSION; D103.4b UNCHANGED.

    USD_SOFR_OIS is the only constructible curve (D080.1), so the two legs get
    distinct DFs through their dates: the asset leg at the delivery date D1
    (about two months) and the strike leg at a settlement four years later.
    Each DF is read BY CURVE NAME AND DATE from the CURVE grid."""
    settlement = dt.date(2030, 1, 15)
    rows = [
        _fwd(
            501,
            "TEST_EXACT_SIMPLE[1M]",
            "USD",
            70.0,
            95.0,
            "USD",
            "USD",
            D1,
            settlement,
        ),
        _fwd(
            9,
            "TEST_EXACT_SIMPLE[3M]",
            "USD",
            -40.0,
            151.0,
            "USD",
            "EUR",
            D1,
            settlement,
        ),
    ]
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(pd.DataFrame(rows)),
        static_config=RSC,
        as_of_date=fx.AS_OF,
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(fx.AS_OF)))
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    layout = build_universe_mapping(
        static_config=RSC, uix_mapper=mapper, discounting_mode=CURVE
    )
    levels = {
        NAME[int(rf)]: (
            CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT[NAME[int(rf)]]
            if NAME[int(rf)].startswith("USD_SOFR_OIS[")
            else LEVELS[NAME[int(rf)]]
        )
        for rf in layout.required_factor_ids
    }
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=fx.AS_OF, levels_dict=levels, static_config=RSC
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=CURVE,
        as_of_date=fx.AS_OF,
        uix_mapper=mapper,
        universe_map=layout,
        snapshot=snapshot,
    )
    grid = matrix.discount_factor_grid

    def df(day: dt.date) -> float:
        c = int(
            np.searchsorted(mapper.discount_curve_id_universe, CURVE_ID["USD_SOFR_OIS"])
        )
        d = int(np.searchsorted(mapper.date_ord32_universe, int(date_to_ord32(day))))
        assert mapper.date_ord32_universe[d] == int(date_to_ord32(day))
        return float(grid[c, d, 0])

    df_a, df_k = df(D1), df(settlement)
    assert 0.0 < df_k < df_a < 1.0 and df_a - df_k > 0.05
    s = sensitize_vanilla_forward_block(
        block=portfolio.vanilla_forward, uix_mapper=mapper, market_data=matrix
    )
    views = {
        501: ("TEST_EXACT_SIMPLE[1M]", None, None),
        9: ("TEST_EXACT_SIMPLE[3M]", "USDEUR", "USDEUR"),
    }
    for row in rows:
        _assert_matches(
            s,
            row["uid"],
            _oracle(
                row["quantity"],
                row["strike_price"],
                views[row["uid"]],
                LEVELS,
                df_a,
                df_k,
            ),
        )


def test_curve_identity_of_each_leg_is_preserved() -> None:
    """FORMULA CHECK ONLY; NOT CURVE ADMISSION; D103.4b UNCHANGED.

    Section 4 binds DF_a to the asset-leg fold (curve and date of
    `efp_effective_quads`) and DF_k to the strike-leg fold (curve and date of
    `edf_effective_triples`). The rows below have different asset and strike
    curves (USD_SOFR_OIS and EUR_ESTR_OIS, in both orientations). The test
    replaces the NONE grid (all 1.0) with test-owned values, one distinct value
    per (curve, date) cell, so a swap of the two curve identities, or of the two
    dates, changes the coefficients. Each DF of the oracle is read BY THE ROW'S
    CURVE NAME AND DATE."""
    rows = [TOPOLOGY[u][0] for u in (70_002, 5, 818, 4_040)]
    w = _build(rows)
    grid = np.ascontiguousarray(
        np.array([[0.97, 0.93], [0.89, 0.85]], dtype=np.float64)[:, :, None]
    )
    assert w.matrix.discount_factor_grid.shape == grid.shape
    matrix = copy.copy(w.matrix)
    object.__setattr__(matrix, "discount_factor_grid", grid)

    def df(curve: str, day: dt.date) -> float:
        c = int(np.searchsorted(w.mapper.discount_curve_id_universe, CURVE_ID[curve]))
        d = int(np.searchsorted(w.mapper.date_ord32_universe, int(date_to_ord32(day))))
        assert w.mapper.date_ord32_universe[d] == int(date_to_ord32(day))
        return float(grid[c, d, 0])

    s = sensitize_vanilla_forward_block(
        block=w.block, uix_mapper=w.mapper, market_data=matrix
    )
    for row in rows:
        assert row["under_discount_curve"] != row["strike_discount_curve"]
        df_a = df(row["under_discount_curve"], row["delivery_date"])
        df_k = df(row["strike_discount_curve"], row["settlement_date"])
        _, view = TOPOLOGY[row["uid"]]
        _assert_matches(
            s,
            row["uid"],
            _oracle(row["quantity"], row["strike_price"], view, LEVELS, df_a, df_k),
        )
