# tests/sensitivity_pnl/test_golden_master_taylor_validity.py
"""
Two-layer golden master for `riskquant.sensitivity_pnl.validity`.

Layer 1 — structural invariants that must hold for ANY book and ANY market:
          `nonlinearity_fraction` bounded in [0, 1]; `linear + quadratic`
          reproduces the real, independently-audited PnL kernel; a pure-delta
          book has EXACTLY zero quadratic term everywhere; batch size and
          chunk partition are scheduling decisions invisible in the result;
          both streaming engines reject a malformed batch identically, since
          their bounds gate was extracted into one shared module rather than
          written twice.

Layer 2 — a tiny test-owned, hand-computed fixture: three positions (mixed
          delta+gamma+cross, pure gamma, and a declared-empty position),
          two scenarios chosen to exercise both edges of
          `nonlinearity_fraction` — exactly 1 (zero linear term, nonzero
          quadratic) and exactly 0 (both terms exactly zero, the 0/0 case)
          — plus `characteristic_displacement` and `summarize_taylor_validity`
          on the same fixture. Every number below was verified against the
          module before being written down here, not the other way around.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl._batch_contract import check_2d_batch_source
from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_LOG_ID,
)
from riskquant.sensitivity_pnl.engine import (
    SensitivityPnlEngine,
    compute_sensitivity_pnl_from_displacement,
)
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.sensitivity_pnl.validity import (
    TaylorValidityEngine,
    TaylorValidityGrid,
    characteristic_displacement,
    compute_taylor_validity_from_displacement,
    compute_taylor_validity_from_levels,
    summarize_taylor_validity,
)

# =========================================================================
# LAYER 2 FIXTURE — test-owned, frozen, hand-computable
# =========================================================================
#
#   factor  id   role in the fixture
#   10           position 42's delta+gamma factor
#   20           position 42's pure-delta factor; ALSO position 43's pure-gamma factor
#   30           position 42's cross-partner for factor 10 (no diagonal entry of its own)
#
#   position 42 (mixed): d1[10]=100, d2[10]=40 ; d1[20]=-50, d2[20]=0 ; cross(10,30)=5
#   position 43 (pure gamma): d1[20]=0, d2[20]=200
#   position 44: declared, no sensitivities at all
#
#   Scenario A: x10=+0.5   x20=+0.2   x30=-0.3
#   Scenario B: x10=-1.0   x20=+0.0   x30=+0.0
#
#   pos42(A): linear = 100(0.5) + -50(0.2)          = 50 - 10        = 40.0
#             quad   = 0.5(40)(0.5^2) + 0 + 5(0.5)(-0.3) = 5.0 + 0 - 0.75 = 4.25
#             fraction = 4.25 / 44.25 = 0.096045...
#   pos42(B): linear = 100(-1.0) + -50(0)           = -100.0
#             quad   = 0.5(40)(1.0) + 0 + 5(-1.0)(0) = 20.0
#             fraction = 20 / 120 = 0.166666...
#
#   pos43(A): linear = 0 ; quad = 0.5(200)(0.2^2) = 4.0 -> fraction = 1.0 (0 linear, nonzero quad)
#   pos43(B): linear = 0 ; quad = 0.5(200)(0.0^2) = 0.0 -> fraction = 0.0 (the 0/0 case)
#
#   pos44: linear = quad = 0 at both scenarios -> fraction = 0.0 (the 0/0 case)

FIXTURE_FACTOR_IDS = np.array([10, 20, 30], dtype=np.int32)
FIXTURE_UIDS = [42, 43, 44]
FIXTURE_DISPLACEMENT = np.ascontiguousarray(
    np.array([[0.5, -1.0], [0.2, 0.0], [-0.3, 0.0]], dtype=np.float32)
)


def fixture_book():
    sensitivities = pl.DataFrame(
        [
            (42, 10, 100.0, 40.0),
            (42, 20, -50.0, 0.0),
            (43, 20, 0.0, 200.0),
        ],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int32,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross = pl.DataFrame(
        [(42, 10, 30, 5.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.Int32,
            "factor_id_j": pl.Int32,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": FIXTURE_FACTOR_IDS,
            "displacement_convention_id": np.full(
                3, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(3),
        }
    )
    return compile_sensitivity_book(
        sensitivities=sensitivities,
        factor_meta=meta,
        level_universe_factor_ids=FIXTURE_FACTOR_IDS,
        cross_sensitivities=cross,
        extra_position_uids=np.array([44], dtype=np.int64),
    )


def fixture_grid() -> TaylorValidityGrid:
    return compute_taylor_validity_from_displacement(
        book=fixture_book(), displacement=FIXTURE_DISPLACEMENT
    )


# =========================================================================
# LAYER 2 — frozen values
# =========================================================================


def test_frozen_fixture_linear_and_quadratic_terms() -> None:
    grid = fixture_grid()
    assert np.array_equal(grid.position_uids, np.array(FIXTURE_UIDS, dtype=np.int64))
    assert np.allclose(
        grid.linear_term, [[40.0, -100.0], [0.0, 0.0], [0.0, 0.0]], atol=1e-4
    )
    assert np.allclose(
        grid.quadratic_term, [[4.25, 20.0], [4.0, 0.0], [0.0, 0.0]], atol=1e-4
    )


def test_frozen_fixture_nonlinearity_fraction_hits_both_edge_cases() -> None:
    grid = fixture_grid()
    fraction = grid.nonlinearity_fraction()
    assert np.allclose(
        fraction,
        [[4.25 / 44.25, 20.0 / 120.0], [1.0, 0.0], [0.0, 0.0]],
        atol=1e-4,
    )
    # Position 43, scenario A: linear is EXACTLY zero and quadratic is not —
    # the fraction must be EXACTLY 1.0, not merely close to it.
    assert fraction[1, 0] == 1.0
    # Position 44 (declared, no sensitivities) and position 43 at scenario B
    # (zero displacement on its only factor): both terms are EXACTLY zero —
    # the fraction must be EXACTLY 0.0, the 0/0 convention, not NaN.
    assert fraction[1, 1] == 0.0
    assert fraction[2, 0] == 0.0 and fraction[2, 1] == 0.0
    assert bool(np.isfinite(fraction).all()), "0/0 must resolve to 0.0, never NaN"


def test_frozen_fixture_characteristic_displacement() -> None:
    result = characteristic_displacement(fixture_book())
    assert result.position_uids.tolist() == [42, 42, 43]
    assert result.factor_ids.tolist() == [10, 20, 20]
    assert result.active_factor_slot.tolist() == [0, 1, 1]
    # factor 10 on position 42: x* = 2*100/40 = 5.0
    assert result.x_star[0] == pytest.approx(5.0)
    # factor 20 on position 42: second_order == 0 -> no breakdown scale at all
    assert np.isinf(result.x_star[1])
    # factor 20 on position 43: zero delta, nonzero gamma -> x* = 0 exactly,
    # meaning ANY nonzero displacement already has quadratic >= linear.
    assert result.x_star[2] == 0.0


def test_frozen_fixture_summary() -> None:
    summary = summarize_taylor_validity(fixture_grid(), threshold=0.15)
    assert summary.position_uids.tolist() == FIXTURE_UIDS
    assert summary.worst_nonlinearity_fraction[0] == pytest.approx(20.0 / 120.0)
    assert summary.worst_nonlinearity_fraction[1] == 1.0
    assert summary.worst_nonlinearity_fraction[2] == 0.0
    # Position 42's worst cell is scenario B (index 1); position 43's is
    # scenario A (index 0); position 44 has no exposure anywhere, so argmax's
    # first-occurrence tie-break lands on index 0 by numpy convention.
    assert summary.worst_scenario_index.tolist() == [1, 0, 0]
    assert summary.exceedance_fraction.tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert summary.num_positions_exceeding == 2


# =========================================================================
# LAYER 1 — structural invariants
# =========================================================================


def random_book(*, num_positions=200, num_factors=48, entries_per=6, seed=11):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + num_factors, dtype=np.int32)
    pos = np.repeat(np.arange(1, num_positions + 1, dtype=np.int64), entries_per)
    fac = np.concatenate(
        [rng.choice(ids, entries_per, replace=False) for _ in range(num_positions)]
    )
    # Delta an order of magnitude larger than gamma — a realistically "quiet"
    # book, not one engineered to be always-strained like a stress fixture.
    sens = pl.DataFrame(
        {
            "position_id": pos,
            "factor_id": fac,
            "first_order": rng.normal(0.0, 1e4, pos.size),
            "second_order": rng.normal(0.0, 1e2, pos.size),
        }
    )
    pair = np.array([rng.choice(ids, 2, replace=False) for _ in range(num_positions)])
    cross = pl.DataFrame(
        {
            "position_id": np.arange(1, num_positions + 1, dtype=np.int64),
            "factor_id_i": pair[:, 0],
            "factor_id_j": pair[:, 1],
            "cross_second_order": rng.normal(0.0, 5.0, num_positions),
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    book = compile_sensitivity_book(
        sensitivities=sens,
        factor_meta=meta,
        level_universe_factor_ids=ids,
        cross_sensitivities=cross,
    )
    disp = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(num_factors, 811)).astype(np.float32)
    )
    return book, disp


def test_nonlinearity_fraction_is_always_bounded_in_zero_one() -> None:
    _, disp = random_book()
    grid = compute_taylor_validity_from_displacement(
        book=random_book()[0], displacement=disp
    )
    fraction = grid.nonlinearity_fraction()
    assert bool(np.isfinite(fraction).all())
    assert bool((fraction >= 0.0).all())
    assert bool((fraction <= 1.0).all())


def test_pure_delta_book_has_exactly_zero_quadratic_term_everywhere() -> None:
    """A payoff with no gamma and no cross term is exact-linear by construction
    (see the forwards/spots demos): the quadratic term must be EXACTLY zero,
    not merely small, for every scenario."""
    ids = np.arange(10, dtype=np.int32)
    sens = pl.DataFrame(
        [(1, 3, 500.0, 0.0), (1, 7, -250.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int32,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                10, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(10),
        }
    )
    book = compile_sensitivity_book(
        sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
    )
    rng = np.random.default_rng(3)
    disp = np.ascontiguousarray(rng.normal(0.0, 5.0, size=(10, 500)).astype(np.float32))
    grid = compute_taylor_validity_from_displacement(book=book, displacement=disp)
    assert bool((grid.quadratic_term == 0.0).all())
    assert bool((grid.nonlinearity_fraction() == 0.0).all())


def test_linear_plus_quadratic_matches_the_independently_audited_pnl_kernel() -> None:
    """
    Cross-check against `kernel_taylor_pnl` — a battle-tested, separately
    reviewed kernel — rather than only trusting this new kernel's own
    internal consistency. Small tolerance expected: the fused Horner form in
    the PnL kernel and the split accumulation here round intermediate sums
    slightly differently, even though both are algebraically the same sum.
    """
    book, disp = random_book(num_positions=150, entries_per=9)
    reference = compute_sensitivity_pnl_from_displacement(
        book=book, displacement=disp
    ).astype(np.float64)
    grid = compute_taylor_validity_from_displacement(book=book, displacement=disp)
    recovered = grid.total_pnl()
    relative_error = np.abs(reference - recovered) / np.maximum(np.abs(reference), 1.0)
    assert relative_error.max() < 1e-3, relative_error.max()


@pytest.mark.parametrize("batch_size", [1, 7, 250, 811, 4000])
def test_batch_size_invariance(batch_size: int) -> None:
    book, disp = random_book()
    reference = compute_taylor_validity_from_displacement(
        book=book, displacement=disp, batch_size=811
    )
    grid = compute_taylor_validity_from_displacement(
        book=book, displacement=disp, batch_size=batch_size
    )
    assert np.array_equal(grid.linear_term, reference.linear_term)
    assert np.array_equal(grid.quadratic_term, reference.quadratic_term)


def test_levels_path_equals_displacement_path() -> None:
    book = fixture_book()
    base_levels = np.array([1.0, 1.0, 1.0])
    shocked_levels = np.ascontiguousarray(
        base_levels[:, None] + FIXTURE_DISPLACEMENT.astype(np.float64)
    )
    from_levels = compute_taylor_validity_from_levels(
        book=book, base_levels=base_levels, shocked_levels=shocked_levels
    )
    from_displacement = fixture_grid()
    assert np.allclose(
        from_levels.linear_term, from_displacement.linear_term, atol=1e-4
    )
    assert np.allclose(
        from_levels.quadratic_term, from_displacement.quadratic_term, atol=1e-4
    )


@pytest.mark.parametrize("partition", ["single", "per_row", "ragged"])
def test_chunk_partition_is_output_invariant(partition: str) -> None:
    book, disp = random_book(num_positions=60)
    reference = compute_taylor_validity_from_displacement(book=book, displacement=disp)

    m = book.num_positions
    offsets = {
        "single": np.array([0, m], dtype=np.int32),
        "per_row": np.arange(m + 1, dtype=np.int32),
        "ragged": np.array([0, 1, 2, 3, m // 2, m - 1, m], dtype=np.int32),
    }[partition]

    linear_out = np.empty((m, disp.shape[1]), dtype=np.float32)
    quadratic_out = np.empty((m, disp.shape[1]), dtype=np.float32)
    engine = TaylorValidityEngine(
        book=book,
        linear_out=linear_out,
        quadratic_out=quadratic_out,
        batch_size=disp.shape[1],
        chunk_offsets=offsets,
    )
    engine.absorb_displacement(
        displacement=disp, batch_len=disp.shape[1], scenario_offset=0
    )
    assert np.array_equal(linear_out, reference.linear_term)
    assert np.array_equal(quadratic_out, reference.quadratic_term)


# =========================================================================
# LAYER 1 — the shared bounds gate must reject identically on both engines
# =========================================================================


def test_both_engines_reject_the_same_malformed_batch_the_same_way() -> None:
    """
    The regression guard for extracting `_batch_contract` out of
    `SensitivityPnlEngine` rather than duplicating it into `TaylorValidityEngine`:
    if the two ever diverge on this, it will show up here first.
    """
    ids = np.array([1], dtype=np.int32)
    sens = pl.DataFrame(
        [(1, 1, 10.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int32,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.array(
                [int(DISPLACEMENT_LOG_ID)], dtype=np.int8
            ),
            "quote_base": [1.0],
        }
    )
    book = compile_sensitivity_book(
        sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
    )
    base_levels = np.array([2.0])
    poisoned_levels = np.ascontiguousarray(np.array([[-1.0]]))

    pnl_grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=1
    )
    pnl_engine = SensitivityPnlEngine(
        book=book, out=pnl_grid.block_view(0), base_levels=base_levels, batch_size=1
    )
    linear_out = np.empty((1, 1), dtype=np.float32)
    quadratic_out = np.empty((1, 1), dtype=np.float32)
    validity_engine = TaylorValidityEngine(
        book=book,
        linear_out=linear_out,
        quadratic_out=quadratic_out,
        base_levels=base_levels,
        batch_size=1,
    )

    with pytest.raises(
        ValueError, match="strictly positive shocked level"
    ) as pnl_error:
        pnl_engine.absorb_levels(
            shocked_levels=poisoned_levels, batch_len=1, scenario_offset=0
        )
    with pytest.raises(
        ValueError, match="strictly positive shocked level"
    ) as validity_error:
        validity_engine.absorb_levels(
            shocked_levels=poisoned_levels, batch_len=1, scenario_offset=0
        )
    assert str(pnl_error.value) == str(validity_error.value)


def test_check_2d_batch_source_is_the_single_shared_bounds_gate() -> None:
    """Direct exercise of the extracted function, independent of either engine."""
    good = np.zeros((3, 10), dtype=np.float32)
    check_2d_batch_source(
        source=good,
        dtype=np.float32,
        name="x",
        min_source_rows=3,
        batch_len=5,
        batch_size=5,
        source_offset=0,
        scenario_offset=0,
        out_num_scenarios=10,
    )
    with pytest.raises(ValueError, match="must be float32"):
        check_2d_batch_source(
            source=good.astype(np.float64),
            dtype=np.float32,
            name="x",
            min_source_rows=3,
            batch_len=5,
            batch_size=5,
            source_offset=0,
            scenario_offset=0,
            out_num_scenarios=10,
        )
    with pytest.raises(ValueError, match="factor rows"):
        check_2d_batch_source(
            source=good,
            dtype=np.float32,
            name="x",
            min_source_rows=99,
            batch_len=5,
            batch_size=5,
            source_offset=0,
            scenario_offset=0,
            out_num_scenarios=10,
        )


# =========================================================================
# LAYER 1 — construction-time rejection
# =========================================================================


def test_taylor_validity_grid_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="rows"):
        TaylorValidityGrid(
            position_uids=np.array([1, 2], dtype=np.int64),
            linear_term=np.zeros((1, 4), dtype=np.float32),
            quadratic_term=np.zeros((2, 4), dtype=np.float32),
        )


def test_taylor_validity_grid_rejects_non_float32() -> None:
    with pytest.raises(ValueError, match="float32"):
        TaylorValidityGrid(
            position_uids=np.array([1], dtype=np.int64),
            linear_term=np.zeros((1, 4), dtype=np.float64),
            quadratic_term=np.zeros((1, 4), dtype=np.float32),
        )


def test_engine_rejects_mismatched_linear_and_quadratic_out_shapes() -> None:
    book = fixture_book()
    with pytest.raises(ValueError, match="quadratic_out"):
        TaylorValidityEngine(
            book=book,
            linear_out=np.zeros((3, 5), dtype=np.float32),
            quadratic_out=np.zeros((3, 4), dtype=np.float32),
        )


def test_summarize_taylor_validity_rejects_out_of_range_threshold() -> None:
    grid = fixture_grid()
    with pytest.raises(ValueError, match="threshold"):
        summarize_taylor_validity(grid, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        summarize_taylor_validity(grid, threshold=1.5)
