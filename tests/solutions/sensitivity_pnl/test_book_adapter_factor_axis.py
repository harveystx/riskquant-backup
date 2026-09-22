# tests/solutions/sensitivity_pnl/test_book_adapter_factor_axis.py
"""
F-1 (D082): the book adapter builds the Taylor book on the DECLARED factor axis.

`build_sensitivity_book_spec(required_factor_ids=...)` must keep the declared
order exactly (never sorted), align every sensitivity to it by FACTOR IDENTITY,
refuse a referenced factor that the axis lacks, refuse a malformed axis, and
create nothing for an axis factor no position references. The calculator keeps
its exact ordered-axis rule. Numbers are checked against an independent oracle:
per-factor Greeks and displacements in plain dicts keyed by factor id, summed
in Python; the market rows are placed by iterating the declared axis list. The
fixture is a positional-order trap: distinctive values per factor, so any
positional reading gives an obviously different P&L.
"""

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivities.result import BlockSensitivities
from riskquant.solutions.run_identity.hashing import hash_compiled_book
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
)
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorContractError,
    TaylorDeclaration,
    TaylorOrder,
)

#: (position, factor) -> (first order, second order): distinctive per factor
GREEKS = {
    (1, 9007): (17.0, 3.0),
    (1, 12): (43.0, 0.0),
    (1, 4002): (91.0, 0.0),
    (2, 9007): (5.0, 0.0),
    (2, 4002): (1000.0, 0.0),
}
#: factor -> displacement per unit scenario scale; 555 is on the axis, unreferenced
DISPLACEMENT = {12: 0.001, 4002: 0.1, 9007: 10.0, 555: 1.0e6}
SCALES = (1.0, -2.0, 0.5)
ATOL = 1e-3  # float32 P&L grid; |P&L| < 2^11, one rounding errs <= 2^-13


def _block(entries):
    pos, fac, d1, d2 = zip(*[(p, f, g[0], g[1]) for (p, f), g in entries])
    return BlockSensitivities(
        position_uids=np.array(pos, np.int64),
        factor_rf_ids=np.array(fac, np.int32),
        first_order=np.array(d1, np.float64),
        second_order=np.array(d2, np.float64),
        cross_position_uids=np.zeros(0, np.int64),
        cross_factor_rf_ids_i=np.zeros(0, np.int32),
        cross_factor_rf_ids_j=np.zeros(0, np.int32),
        cross_second_order=np.zeros(0, np.float64),
    )


def _calc(axis, entries=None):
    blk = _block(list(GREEKS.items()) if entries is None else entries)
    spec = build_sensitivity_book_spec(
        sensitivities=blk, required_factor_ids=np.array(axis, np.int32)
    )
    return spec, SensitivityPnlCalculator.compile(spec)


def _market(calc, displacement=DISPLACEMENT, scales=SCALES):
    rows = [
        [displacement[int(f)] * s for s in scales] for f in calc.axis.keys.to_list()
    ]
    return calc.axis.shocks(np.array(rows, dtype=np.float64))


def _oracle(displacement=DISPLACEMENT, scales=SCALES):
    out = {}
    for p in (1, 2):
        out[p] = [
            sum(
                d1 * displacement[f] * s + 0.5 * d2 * (displacement[f] * s) ** 2
                for (q, f), (d1, d2) in GREEKS.items()
                if q == p
            )
            for s in scales
        ]
    return out


def _pnl(calc, **kw):
    res = calc.run(_market(calc, **kw))
    return {
        int(u): np.asarray(res.pnl[i], np.float64)
        for i, u in enumerate(res.position_uids)
    }


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------
def test_axis_is_the_declared_axis_in_declared_order() -> None:
    spec, calc = _calc([4002, 9007, 12, 555])
    assert spec.factor_axis.keys.to_list() == [4002, 9007, 12, 555]  # never sorted
    assert spec.factor_axis.slot_map is not None  # identity map, not position
    assert sorted(spec.factor_meta["factor_id"].to_list()) == [12, 555, 4002, 9007]
    assert calc.axis.keys.to_list() == [4002, 9007, 12, 555]


@pytest.mark.parametrize(
    "axis",
    [[4002, 9007, 12, 555], [12, 555, 4002, 9007], [555, 12, 9007, 4002]],
    ids=["unsorted", "sorted_direct", "reversed_ish"],
)
@pytest.mark.parametrize(
    "row_order", [None, [4, 2, 0, 3, 1]], ids=["rows_as_given", "rows_permuted"]
)
def test_positional_order_trap_prices_by_factor_identity(axis, row_order) -> None:
    entries = list(GREEKS.items())
    if row_order is not None:
        entries = [entries[i] for i in row_order]
    _, calc = _calc(axis, entries)
    got, want = _pnl(calc), _oracle()
    for p in (1, 2):
        np.testing.assert_allclose(got[p], want[p], rtol=0, atol=ATOL)
    # identifiability: a positional misreading (displacements rotated one row) is far off
    rotated = dict(
        zip(
            DISPLACEMENT,
            list(DISPLACEMENT.values())[1:] + list(DISPLACEMENT.values())[:1],
        )
    )
    assert (
        max(abs(a - b) for p in (1, 2) for a, b in zip(_oracle(rotated)[p], want[p]))
        > 1e3 * ATOL
    )


def test_inactive_axis_factor_gets_no_entry_and_no_value() -> None:
    _, calc = _calc([4002, 9007, 12, 555])
    assert calc.book.num_active_factors == 3  # 555 has no entry
    quiet = {**DISPLACEMENT, 555: 0.0}
    loud, silent = _pnl(calc), _pnl(calc, displacement=quiet)
    for p in (1, 2):
        assert np.array_equal(loud[p], silent[p])  # 555 never read


def test_repeated_build_is_bitwise_identical() -> None:
    _, a = _calc([4002, 9007, 12, 555])
    _, b = _calc([4002, 9007, 12, 555])
    assert hash_compiled_book(a.book) == hash_compiled_book(b.book)
    pa, pb = _pnl(a), _pnl(b)
    assert all(np.array_equal(pa[p], pb[p]) for p in (1, 2))


def test_scenario_order_permutes_columns_only() -> None:
    _, calc = _calc([4002, 9007, 12, 555])
    perm = [2, 0, 1]
    base, moved = _pnl(calc), _pnl(calc, scales=tuple(SCALES[i] for i in perm))
    assert all(np.array_equal(base[p][perm], moved[p]) for p in (1, 2))


# ---------------------------------------------------------------------------
# Negative controls: fail fast, by identity, before any numerics
# ---------------------------------------------------------------------------
def test_referenced_factor_missing_from_the_axis_fails() -> None:
    with pytest.raises(
        ValueError,
        match=r"not on required_factor_ids \(first 10: \[9007\]\).*never dropped",
    ):
        _calc([4002, 12, 555])


def test_duplicate_axis_ids_fail() -> None:
    with pytest.raises(ValueError, match=r"more than once.*\[4002\]"):
        _calc([4002, 9007, 12, 4002])


@pytest.mark.parametrize(
    "axis,error",
    [
        (np.array([4002, 9007, 12], np.int64), TypeError),
        (np.array([[4002, 9007, 12]], np.int32), TypeError),
        ([4002, 9007, 12], TypeError),
        (np.zeros(0, np.int32), ValueError),
    ],
)
def test_malformed_axis_fails(axis, error) -> None:
    with pytest.raises(error):
        build_sensitivity_book_spec(
            sensitivities=_block(list(GREEKS.items())), required_factor_ids=axis
        )


def test_duplicate_sensitivity_rows_fail_before_numerics() -> None:
    entries = list(GREEKS.items()) + [((1, 9007), (1.0, 0.0))]
    with pytest.raises((ValueError, SensitivityPnlInputError)):
        _calc([4002, 9007, 12, 555], entries)


def test_metadata_missing_an_axis_factor_fails_before_numerics() -> None:
    spec, _ = _calc([4002, 9007, 12, 555])
    import dataclasses

    broken = dataclasses.replace(
        spec, factor_meta=spec.factor_meta.filter(pl.col("factor_id") != 12)
    )
    with pytest.raises(SensitivityPnlInputError):
        SensitivityPnlCalculator.compile(broken)


def test_wrong_axis_market_is_refused_by_the_calculator() -> None:
    _, calc = _calc([4002, 9007, 12, 555])
    _, other = _calc([9007, 4002, 12, 555])  # same set, other order
    with pytest.raises(SensitivityPnlInputError, match="different row order"):
        calc.run(_market(other))


def test_declared_dependency_without_derivative_is_never_zero_filled() -> None:
    """555 is on the axis with no entry. If the declaration says position 1
    DEPENDS on 555, the missing derivative fails closed (P1-A); only a factor
    outside every F(p) may stay inactive."""
    spec, _ = _calc([4002, 9007, 12, 555])
    binding = TaylorBaseBinding(
        as_of_ord32=0,
        base_levels_digest="-",
        discounting_mode="NONE",
        static_config_digest="-",
        valuation_contract_fingerprint="-",
        base_settlement_values_digest="-",
    )
    deps = [(p, f) for (p, f) in GREEKS]
    import dataclasses

    def declared(extra):
        rows = deps + extra
        return TaylorDeclaration(
            order=TaylorOrder.SECOND_ORDER_TRUNCATED,
            dependencies=pl.DataFrame({
                "position_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
                "factor_id": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            }),
            zero_exposure_position_uids=(),
            base_binding=binding,
        )

    SensitivityPnlCalculator.compile(
        dataclasses.replace(spec, taylor_declaration=declared([]))
    )
    with pytest.raises(TaylorContractError):
        SensitivityPnlCalculator.compile(
            dataclasses.replace(spec, taylor_declaration=declared([(1, 555)]))
        )


# ---------------------------------------------------------------------------
# F-1 hardening: compile-key semantics, metadata scoping, hostile cross terms
# ---------------------------------------------------------------------------
def test_active_factor_ids_are_compile_keys_and_identity_comes_from_the_axis() -> None:
    """`CompiledSensitivityBook.active_factor_ids` is the CORE compiler's key
    space by design (pipeline.py 'Slot relabelling'; taylor_validity.py): the
    factor ids under DIRECT encoding, axis row positions under SLOT. The
    encoding-independent identity is `axis.keys[book.active_level_uix]`."""
    _, slot = _calc([4002, 9007, 12, 555])
    assert slot.axis.slot_map is not None
    assert slot.book.active_factor_ids.tolist() == [0, 1, 2]  # row positions, not ids
    _, direct = _calc([12, 555, 4002, 9007])
    assert direct.axis.slot_map is None
    assert direct.book.active_factor_ids.tolist() == [12, 4002, 9007]  # ids
    for calc in (slot, direct):
        ids = calc.axis.keys.to_numpy()[calc.book.active_level_uix]
        assert sorted(ids.tolist()) == [12, 4002, 9007]  # identity, both encodings


def _raw_spec(axis, referenced, meta_ids):
    from riskquant.solutions.sensitivity_pnl.api import FactorAxis, SensitivityBookSpec

    greek = {101: 7.0, 205: -3.0, 900: 11.0}
    return SensitivityBookSpec(
        sensitivities=pl.DataFrame({
            "position_id": pl.Series([1] * len(referenced), dtype=pl.Int64),
            "factor_id": pl.Series(referenced, dtype=pl.Int32),
            "first_order": pl.Series([greek[f] for f in referenced], dtype=pl.Float64),
            "second_order": pl.Series([0.0] * len(referenced), dtype=pl.Float64),
        }),
        factor_meta=pl.DataFrame({
            "factor_id": pl.Series(meta_ids, dtype=pl.Int32),
            "displacement_convention_id": pl.Series([1] * len(meta_ids), dtype=pl.Int8),
            "quote_base": pl.Series([1.0] * len(meta_ids), dtype=pl.Float64),
        }),
        factor_axis=FactorAxis.from_keys(pl.Series("factor_id", axis, dtype=pl.Int32)),
        position_uids=[1],
    )


@pytest.mark.parametrize("axis", [[101, 205], [205, 101]], ids=["direct", "slot"])
def test_scope_factor_meta_ignores_only_off_axis_metadata(axis) -> None:
    lean = SensitivityPnlCalculator.compile(_raw_spec(axis, [101, 205], [101, 205]))
    extra = SensitivityPnlCalculator.compile(
        _raw_spec(axis, [101, 205], [101, 205, 900])
    )
    assert hash_compiled_book(lean.book) == hash_compiled_book(
        extra.book
    )  # row 900 has no effect


@pytest.mark.parametrize("axis", [[101, 205], [205, 101]], ids=["direct", "slot"])
def test_a_referenced_off_axis_factor_fails_even_when_its_metadata_exists(axis) -> None:
    """No path scopes away the metadata of a factor the book references and then
    prices: the off-axis reference itself fails, under both encodings."""
    with pytest.raises((ValueError, SensitivityPnlInputError)):
        SensitivityPnlCalculator.compile(
            _raw_spec(axis, [101, 205, 900], [101, 205, 900])
        )


#: (position, factor_i, factor_j) -> full mixed partial (counted once, no 1/2)
CROSS = {(1, 9007, 4002): 7.0, (2, 12, 9007): -250.0}


def _cross_block():
    base = _block(list(GREEKS.items()))
    import dataclasses

    return dataclasses.replace(
        base,
        cross_position_uids=np.array([k[0] for k in CROSS], np.int64),
        cross_factor_rf_ids_i=np.array([k[1] for k in CROSS], np.int32),
        cross_factor_rf_ids_j=np.array([k[2] for k in CROSS], np.int32),
        cross_second_order=np.array(list(CROSS.values()), np.float64),
    )


def _cross_oracle(displacement=DISPLACEMENT):
    out = _oracle(displacement)
    for (p, fi, fj), c in CROSS.items():
        out[p] = [
            v + c * displacement[fi] * s * displacement[fj] * s
            for v, s in zip(out[p], SCALES)
        ]
    return out


@pytest.mark.parametrize(
    "axis",
    [[4002, 9007, 12, 555], [12, 555, 4002, 9007], [9007, 555, 12, 4002]],
    ids=["unsorted", "sorted_direct", "pair_reversed"],
)
def test_cross_terms_are_aligned_by_factor_identity(axis) -> None:
    spec = build_sensitivity_book_spec(
        sensitivities=_cross_block(), required_factor_ids=np.array(axis, np.int32)
    )
    calc = SensitivityPnlCalculator.compile(spec)
    got, want = _pnl(calc), _cross_oracle()
    for p in (1, 2):
        np.testing.assert_allclose(got[p], want[p], rtol=0, atol=ATOL)
    # identifiability: the cross terms move P&L materially, and a positional
    # misreading of either pair would read the 555 row (1e6 per unit scale)
    no_cross = _oracle()
    assert (
        max(abs(a - b) for p in (1, 2) for a, b in zip(want[p], no_cross[p]))
        > 1e3 * ATOL
    )
    misread = {**DISPLACEMENT, 9007: DISPLACEMENT[555]}
    assert (
        max(
            abs(a - b)
            for p in (1, 2)
            for a, b in zip(_cross_oracle(misread)[p], want[p])
        )
        > 1e3 * ATOL
    )
