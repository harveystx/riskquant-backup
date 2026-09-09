# tests/sensitivity_pnl/test_golden_master_sensitivity_pnl.py
"""
Two-layer golden master for `riskquant.sensitivity_pnl`.

Layer 1 — structural invariants. Input-agnostic properties that must hold for
          ANY book and ANY scenario set. These survive every change to the
          fixtures and are where the engine's real contracts live.

Layer 2 — a tiny test-OWNED frozen book with hand-computed PnL. Six numbers,
          derivable on paper from the docstrings. Owned here, not sourced from
          a canonical fixture, so it cannot drift when the canonical book does.

`test_thread_determinism` is the guard on the `nb.get_thread_id()` per-thread
accumulator in the Taylor kernel. If that pattern is ever wrong, this is the
only test that catches it. Do not delete it, and do not let it silently pass on
a single-core CI box — it skips loudly.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import (
    DuplicatePolicy,
    compile_sensitivity_book,
)
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_LOG_ID,
    DISPLACEMENT_RELATIVE_ID,
    QUOTE_BASE_NATURAL,
    QUOTE_BASE_PER_VOL_POINT,
)
from riskquant.sensitivity_pnl.engine import (
    SensitivityPnlEngine,
    compute_sensitivity_pnl_from_displacement,
    compute_sensitivity_pnl_from_levels,
    numba_thread_ceiling,
)
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.sensitivity_pnl.scheduling import (
    balanced_chunk_offsets,
    default_num_chunks,
    row_costs,
)

# =========================================================================
# LAYER 2 FIXTURE — test-owned, frozen, hand-computable
# =========================================================================
#
#   factor  convention  base   quote_base
#   10      ABSOLUTE    1.0    1.0
#   20      RELATIVE    4.0    1.0
#   30      LOG         2.0    100.0   (per vol point)
#
#   P1 (uid 1):  d1[10]=2.0  d2[10]=6.0 ;  d1[30]=0.5(quoted)  d2[30]=0.0
#   P2 (uid 2):  cross(10,20) = 3.0
#   P3 (uid 3):  nothing
#
#   Scenario A levels = [1.5, 5.0, 2*exp(0.25)]  ->  x = [ 0.50,  0.25,  0.25]
#   Scenario B levels = [0.5, 2.0, 2*exp(-0.5)]  ->  x = [-0.50, -0.50, -0.50]
#
#   P1(A) = 2(0.50) + 0.5(6)(0.50^2) + 50(0.25)      =  1.00 + 0.75 + 12.50
#   P1(B) = 2(-0.50) + 0.5(6)(-0.50^2) + 50(-0.50)   = -1.00 + 0.75 - 25.00
#   P2(A) = 3(0.50)(0.25)                            =  0.375
#   P2(B) = 3(-0.50)(-0.50)                          =  0.75

FROZEN_LEVEL_IDS = np.array([10, 20, 30], dtype=np.int32)
FROZEN_BASE_LEVELS = np.array([1.0, 4.0, 2.0], dtype=np.float64)

FROZEN_FACTOR_META = pl.DataFrame(
    {
        "factor_id": [10, 20, 30],
        "displacement_convention_id": [
            int(DISPLACEMENT_ABSOLUTE_ID),
            int(DISPLACEMENT_RELATIVE_ID),
            int(DISPLACEMENT_LOG_ID),
        ],
        "quote_base": [
            QUOTE_BASE_NATURAL,
            QUOTE_BASE_NATURAL,
            QUOTE_BASE_PER_VOL_POINT,
        ],
    }
)
FROZEN_SENS = pl.DataFrame(
    {
        "position_id": [1, 1],
        "factor_id": [10, 30],
        "first_order": [2.0, 0.5],
        "second_order": [6.0, 0.0],
    }
)
FROZEN_CROSS = pl.DataFrame(
    {
        "position_id": [2],
        "factor_id_i": [10],
        "factor_id_j": [20],
        "cross_second_order": [3.0],
    }
)
FROZEN_SHOCKED_LEVELS = np.ascontiguousarray(
    np.array(
        [
            [1.5, 0.5],
            [5.0, 2.0],
            [2.0 * math.exp(0.25), 2.0 * math.exp(-0.5)],
        ],
        dtype=np.float64,
    )
)
FROZEN_EXPECTED_PNL = np.array(
    [
        [14.25, -25.25],
        [0.375, 0.75],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)


def frozen_book():
    return compile_sensitivity_book(
        sensitivities=FROZEN_SENS,
        factor_meta=FROZEN_FACTOR_META,
        level_universe_factor_ids=FROZEN_LEVEL_IDS,
        cross_sensitivities=FROZEN_CROSS,
        extra_position_uids=np.array([3], dtype=np.int64),
    )


# =========================================================================
# LAYER 2 — frozen values
# =========================================================================


def test_frozen_book_pnl_matches_hand_computation() -> None:
    pnl = compute_sensitivity_pnl_from_levels(
        book=frozen_book(),
        base_levels=FROZEN_BASE_LEVELS,
        shocked_levels=FROZEN_SHOCKED_LEVELS,
    )
    assert np.allclose(pnl, FROZEN_EXPECTED_PNL, rtol=1e-6, atol=1e-6)


def test_frozen_book_structure() -> None:
    book = frozen_book()
    assert np.array_equal(book.position_uids, np.array([1, 2, 3], dtype=np.int64))
    assert np.array_equal(
        book.active_factor_ids, np.array([10, 20, 30], dtype=np.int32)
    )
    assert np.array_equal(book.active_level_uix, np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(book.position_offsets, np.array([0, 2, 2, 2], dtype=np.int32))
    assert np.array_equal(book.cross_offsets, np.array([0, 0, 1, 1], dtype=np.int32))
    # quote_base folded in: vol-point first order 0.5 -> 50.0 natural.
    assert np.allclose(book.first_order, [2.0, 50.0])
    assert np.allclose(book.second_order, [6.0, 0.0])


# =========================================================================
# LAYER 1 — structural invariants
# =========================================================================


def random_book(*, num_positions=400, num_factors=64, entries_per=6, seed=7):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + num_factors, dtype=np.int32)
    pos = np.repeat(np.arange(1, num_positions + 1, dtype=np.int64), entries_per)
    fac = np.concatenate(
        [rng.choice(ids, entries_per, replace=False) for _ in range(num_positions)]
    )
    sens = pl.DataFrame(
        {
            "position_id": pos,
            "factor_id": fac,
            "first_order": rng.normal(0.0, 100.0, pos.size),
            "second_order": rng.normal(0.0, 50.0, pos.size),
        }
    )
    pair = np.array([rng.choice(ids, 2, replace=False) for _ in range(num_positions)])
    cross = pl.DataFrame(
        {
            "position_id": np.arange(1, num_positions + 1, dtype=np.int64),
            "factor_id_i": pair[:, 0],
            "factor_id_j": pair[:, 1],
            "cross_second_order": rng.normal(0.0, 20.0, num_positions),
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
        rng.normal(0.0, 0.02, size=(num_factors, 1301)).astype(np.float32)
    )
    return book, disp


def test_thread_determinism() -> None:
    """
    Guards the `nb.get_thread_id()` per-thread accumulator. A stale or shared
    scratch row shows up here as another position's PnL and nowhere else.
    """
    max_threads = numba_thread_ceiling()
    if max_threads < 2:
        pytest.skip(
            f"single-thread host (NUMBA_NUM_THREADS={max_threads}); the "
            "per-thread accumulator in kernel_taylor_pnl is UNVERIFIED here"
        )
    book, disp = random_book()
    original = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        serial = compute_sensitivity_pnl_from_displacement(book=book, displacement=disp)
        nb.set_num_threads(max_threads)
        parallel = compute_sensitivity_pnl_from_displacement(
            book=book, displacement=disp
        )
    finally:
        nb.set_num_threads(original)
    assert np.array_equal(serial, parallel)


@pytest.mark.parametrize("batch_size", [1, 7, 250, 1301, 5000])
def test_batch_size_invariance(batch_size: int) -> None:
    book, disp = random_book()
    reference = compute_sensitivity_pnl_from_displacement(
        book=book, displacement=disp, batch_size=1301
    )
    assert np.array_equal(
        compute_sensitivity_pnl_from_displacement(
            book=book, displacement=disp, batch_size=batch_size
        ),
        reference,
    )


def test_zero_displacement_is_exactly_zero() -> None:
    book, disp = random_book()
    pnl = compute_sensitivity_pnl_from_displacement(
        book=book, displacement=np.zeros_like(disp)
    )
    assert np.all(pnl == 0.0)


def test_first_order_homogeneity() -> None:
    """A book with only first-order terms is linear in the displacement."""
    ids = np.array([1, 2], dtype=np.int32)
    sens = pl.DataFrame(
        {
            "position_id": [1, 1],
            "factor_id": [1, 2],
            "first_order": [3.0, -7.0],
            "second_order": [0.0, 0.0],
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": [int(DISPLACEMENT_ABSOLUTE_ID)] * 2,
            "quote_base": [1.0, 1.0],
        }
    )
    book = compile_sensitivity_book(
        sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
    )
    disp = np.ascontiguousarray(np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32))
    base = compute_sensitivity_pnl_from_displacement(book=book, displacement=disp)
    scaled = compute_sensitivity_pnl_from_displacement(
        book=book, displacement=np.ascontiguousarray(disp * np.float32(2.5))
    )
    assert np.allclose(scaled, base * 2.5, rtol=1e-6)


def test_positions_are_independent() -> None:
    """Row m depends on row m's CSR slice alone — no cross-position leakage."""
    book, disp = random_book(num_positions=40)
    full = compute_sensitivity_pnl_from_displacement(book=book, displacement=disp)
    solo_sens = pl.DataFrame(
        {
            "position_id": np.full(
                int(book.position_offsets[1]) - int(book.position_offsets[0]),
                int(book.position_uids[0]),
                dtype=np.int64,
            ),
            "factor_id": book.active_factor_ids[
                book.factor_uix[book.position_offsets[0] : book.position_offsets[1]]
            ],
            "first_order": book.first_order[
                book.position_offsets[0] : book.position_offsets[1]
            ],
            "second_order": book.second_order[
                book.position_offsets[0] : book.position_offsets[1]
            ],
        }
    )
    solo_cross = pl.DataFrame(
        {
            "position_id": np.full(1, int(book.position_uids[0]), dtype=np.int64),
            "factor_id_i": book.active_factor_ids[book.cross_factor_uix_i[:1]],
            "factor_id_j": book.active_factor_ids[book.cross_factor_uix_j[:1]],
            "cross_second_order": book.cross_second_order[:1],
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": book.active_factor_ids,
            "displacement_convention_id": book.displacement_convention_ids,
            "quote_base": np.ones(book.num_active_factors),
        }
    )
    solo = compile_sensitivity_book(
        sensitivities=solo_sens,
        factor_meta=meta,
        level_universe_factor_ids=book.active_factor_ids,
        cross_sensitivities=solo_cross,
    )
    solo_pnl = compute_sensitivity_pnl_from_displacement(
        book=solo, displacement=np.ascontiguousarray(disp[book.active_level_uix])
    )
    assert np.allclose(solo_pnl[0], full[0], rtol=1e-6)


def test_levels_path_equals_displacement_path() -> None:
    """The two input contracts must agree on the same market state."""
    book = frozen_book()
    from_levels = compute_sensitivity_pnl_from_levels(
        book=book,
        base_levels=FROZEN_BASE_LEVELS,
        shocked_levels=FROZEN_SHOCKED_LEVELS,
    )
    displacement = np.ascontiguousarray(
        np.vstack(
            [
                FROZEN_SHOCKED_LEVELS[0] - FROZEN_BASE_LEVELS[0],
                FROZEN_SHOCKED_LEVELS[1] / FROZEN_BASE_LEVELS[1] - 1.0,
                np.log(FROZEN_SHOCKED_LEVELS[2] / FROZEN_BASE_LEVELS[2]),
            ]
        )
    ).astype(np.float32)
    from_displacement = compute_sensitivity_pnl_from_displacement(
        book=book, displacement=displacement
    )
    assert np.allclose(from_levels, from_displacement, rtol=1e-6, atol=1e-6)


def test_grid_views_are_disjoint_and_non_aliasing() -> None:
    grid = allocate_pnl_grid(
        block_position_uids=[
            np.array([10, 11, 12], dtype=np.int64),
            np.array([20, 21], dtype=np.int64),
        ],
        num_scenarios=8,
    )
    grid.block_view(0)[:] = 1.0
    grid.block_view(1)[:] = 2.0
    assert grid.block_view(0).flags.c_contiguous
    assert grid.block_view(1).flags.c_contiguous
    assert np.all(grid.pnl[:3] == 1.0)
    assert np.all(grid.pnl[3:] == 2.0)
    assert grid.to_pnl_matrix().pnl is grid.pnl  # zero-copy handoff


def test_overlapping_blocks_rejected_at_allocation() -> None:
    with pytest.raises(ValueError, match="double-counted"):
        allocate_pnl_grid(
            block_position_uids=[
                np.array([1, 2], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
            ],
            num_scenarios=4,
        )


def test_engine_writes_only_its_own_rows() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(
        block_position_uids=[np.array([900, 901], dtype=np.int64), book.position_uids],
        num_scenarios=2,
    )
    grid.block_view(0)[:] = -1.0
    engine = SensitivityPnlEngine(
        book=book,
        out=grid.block_view(1),
        base_levels=FROZEN_BASE_LEVELS,
        batch_size=2,
    )
    engine.absorb_levels(
        shocked_levels=FROZEN_SHOCKED_LEVELS, batch_len=2, scenario_offset=0
    )
    assert np.all(grid.block_view(0) == -1.0)
    assert np.allclose(grid.block_view(1), FROZEN_EXPECTED_PNL, rtol=1e-6, atol=1e-6)


def test_output_contract() -> None:
    book, disp = random_book(num_positions=16)
    pnl = compute_sensitivity_pnl_from_displacement(book=book, displacement=disp)
    assert pnl.dtype == np.float32
    assert pnl.flags.c_contiguous
    assert pnl.shape == (book.num_positions, disp.shape[1])


# =========================================================================
# LAYER 1 — fail-fast surface
# =========================================================================


def test_factor_outside_level_universe_raises() -> None:
    with pytest.raises(ValueError, match="absent from the universe"):
        compile_sensitivity_book(
            sensitivities=FROZEN_SENS,
            factor_meta=FROZEN_FACTOR_META,
            level_universe_factor_ids=np.array([10, 20], dtype=np.int32),
        )


def test_duplicate_policy_raise_is_default() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        compile_sensitivity_book(
            sensitivities=pl.concat([FROZEN_SENS, FROZEN_SENS.head(1)]),
            factor_meta=FROZEN_FACTOR_META,
            level_universe_factor_ids=FROZEN_LEVEL_IDS,
        )


def test_duplicate_policy_sum_adds_coefficients() -> None:
    book = compile_sensitivity_book(
        sensitivities=pl.concat([FROZEN_SENS, FROZEN_SENS.head(1)]),
        factor_meta=FROZEN_FACTOR_META,
        level_universe_factor_ids=FROZEN_LEVEL_IDS,
        duplicate_policy=DuplicatePolicy.SUM,
    )
    assert book.nnz == 2
    # factor 10 appeared twice: first_order 2.0 + 2.0, second_order 6.0 + 6.0.
    assert np.allclose(book.first_order, [4.0, 50.0])
    assert np.allclose(book.second_order, [12.0, 0.0])


def test_cross_diagonal_pair_rejected() -> None:
    with pytest.raises(ValueError, match="belongs in `second_order`"):
        compile_sensitivity_book(
            sensitivities=FROZEN_SENS,
            factor_meta=FROZEN_FACTOR_META,
            level_universe_factor_ids=FROZEN_LEVEL_IDS,
            cross_sensitivities=FROZEN_CROSS.with_columns(
                pl.col("factor_id_j").alias("factor_id_i")
            ),
        )


def test_non_finite_coefficient_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compile_sensitivity_book(
            sensitivities=FROZEN_SENS.with_columns(
                pl.col("first_order").replace(2.0, float("nan"))
            ),
            factor_meta=FROZEN_FACTOR_META,
            level_universe_factor_ids=FROZEN_LEVEL_IDS,
        )


def test_non_positive_base_level_rejected_for_ratio_convention() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    with pytest.raises(ValueError, match="strictly positive base level"):
        SensitivityPnlEngine(
            book=book,
            out=grid.block_view(0),
            base_levels=np.array([1.0, -4.0, 2.0]),
        )


def test_absorb_levels_without_base_levels_raises() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    engine = SensitivityPnlEngine(book=book, out=grid.block_view(0), batch_size=2)
    with pytest.raises(ValueError, match="requires base_levels"):
        engine.absorb_levels(
            shocked_levels=FROZEN_SHOCKED_LEVELS, batch_len=2, scenario_offset=0
        )


def test_undersized_displacement_rejected() -> None:
    """Cross-only books have nnz == 0 but a populated active-factor axis."""
    ids = np.array([0, 1, 2], dtype=np.int32)
    cross = pl.DataFrame(
        {
            "position_id": [1],
            "factor_id_i": [0],
            "factor_id_j": [2],
            "cross_second_order": [1.0],
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": [int(DISPLACEMENT_ABSOLUTE_ID)] * 3,
            "quote_base": [1.0, 1.0, 1.0],
        }
    )
    book = compile_sensitivity_book(
        sensitivities=pl.DataFrame(
            schema={
                "position_id": pl.Int64,
                "factor_id": pl.Int32,
                "first_order": pl.Float64,
                "second_order": pl.Float64,
            }
        ),
        factor_meta=meta,
        level_universe_factor_ids=ids,
        cross_sensitivities=cross,
    )
    assert book.nnz == 0 and book.num_active_factors == 2
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    engine = SensitivityPnlEngine(book=book, out=grid.block_view(0), batch_size=2)
    with pytest.raises(ValueError, match="factor rows"):
        engine.absorb_displacement(
            displacement=np.ascontiguousarray(np.zeros((1, 2), dtype=np.float32)),
            batch_len=2,
            scenario_offset=0,
        )


def test_non_contiguous_out_rejected() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=8)
    with pytest.raises(ValueError, match="C-contiguous"):
        SensitivityPnlEngine(book=book, out=grid.pnl[:, :4])


# =========================================================================
# LAYER 1 — memory-safety gate (kernels run under boundscheck=False)
# =========================================================================


def _malformed(**overrides):
    fields: dict[str, np.ndarray] = {
        "position_uids": np.array([1, 2], dtype=np.int64),
        "active_factor_ids": np.array([10, 20], dtype=np.int32),
        "active_level_uix": np.array([0, 1], dtype=np.int32),
        "displacement_convention_ids": np.array([1, 1], dtype=np.int8),
        "position_offsets": np.array([0, 0, 0], dtype=np.int32),
        "factor_uix": np.empty(0, dtype=np.int32),
        "first_order": np.empty(0, dtype=np.float64),
        "second_order": np.empty(0, dtype=np.float64),
        "cross_offsets": np.array([0, 1, 1], dtype=np.int32),
        "cross_factor_uix_i": np.array([0], dtype=np.int32),
        "cross_factor_uix_j": np.array([1], dtype=np.int32),
        "cross_second_order": np.array([1.0], dtype=np.float64),
    }
    fields.update(overrides)
    return CompiledSensitivityBook(**fields)


def test_wellformed_control_book_is_accepted() -> None:
    """The malformed-book fixture must be valid before each mutation."""
    assert _malformed().nnz_cross == 1


def test_cross_index_out_of_range_rejected() -> None:
    """Cross indices were unchecked whenever nnz == 0 — a silent OOB read."""
    with pytest.raises(ValueError, match="cross_factor_uix_j out of range"):
        _malformed(cross_factor_uix_j=np.array([9999], dtype=np.int32))


def test_csr_offsets_must_start_at_zero() -> None:
    with pytest.raises(ValueError, match="must start at 0"):
        _malformed(position_offsets=np.array([7, 7, 7], dtype=np.int32))


def test_csr_offsets_must_be_non_decreasing() -> None:
    """[0, 5, 0] terminates at 0 but makes the kernel walk range(0, 5) over NNZ=0."""
    with pytest.raises(ValueError, match="non-decreasing"):
        _malformed(position_offsets=np.array([0, 5, 0], dtype=np.int32))


def test_csr_offsets_must_terminate_at_nnz() -> None:
    with pytest.raises(ValueError, match="expected NNZ"):
        _malformed(cross_offsets=np.array([0, 0, 0], dtype=np.int32))


def test_displacement_dtype_is_enforced() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    engine = SensitivityPnlEngine(book=book, out=grid.block_view(0), batch_size=2)
    with pytest.raises(ValueError, match="must be float32"):
        engine.absorb_displacement(
            # Deliberately float64 — the global f32 fixture sweep must not
            # \"correct\" this one, or the test asserts nothing.
            displacement=np.ascontiguousarray(np.zeros((3, 2), dtype=np.float64)),
            batch_len=2,
            scenario_offset=0,
        )


def test_entry_blocking_boundaries() -> None:
    """
    The Taylor kernel seeds 4-wide, blocks 4-wide, and drains a <4 remainder.
    Every nnz-per-position in 0..9 must therefore be exercised, since the seed
    arm taken and the remainder length both change with it.
    """
    num_factors = 12
    ids = np.arange(num_factors, dtype=np.int32)
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    rng = np.random.default_rng(31)
    for nnz_per in range(10):
        for cross_per in range(3):
            sens = pl.DataFrame(
                {
                    "position_id": np.full(nnz_per, 1, dtype=np.int64),
                    "factor_id": ids[:nnz_per],
                    "first_order": rng.normal(0.0, 10.0, nnz_per),
                    "second_order": rng.normal(0.0, 5.0, nnz_per),
                },
                schema={
                    "position_id": pl.Int64,
                    "factor_id": pl.Int32,
                    "first_order": pl.Float64,
                    "second_order": pl.Float64,
                },
            )
            cross = pl.DataFrame(
                {
                    "position_id": np.full(cross_per, 1, dtype=np.int64),
                    "factor_id_i": ids[:cross_per],
                    "factor_id_j": ids[6 : 6 + cross_per],
                    "cross_second_order": rng.normal(0.0, 3.0, cross_per),
                },
                schema={
                    "position_id": pl.Int64,
                    "factor_id_i": pl.Int32,
                    "factor_id_j": pl.Int32,
                    "cross_second_order": pl.Float64,
                },
            )
            book = compile_sensitivity_book(
                sensitivities=sens,
                factor_meta=meta,
                level_universe_factor_ids=ids,
                cross_sensitivities=cross,
                extra_position_uids=np.array([1], dtype=np.int64),
            )
            disp = np.ascontiguousarray(
                rng.normal(0.0, 0.02, size=(num_factors, 6)).astype(np.float32)
            )
            got = compute_sensitivity_pnl_from_displacement(
                book=book, displacement=disp
            )
            want = np.zeros(6)
            for e in range(book.nnz):
                x = disp[book.active_level_uix[book.factor_uix[e]]].astype(np.float64)
                want += book.first_order[e] * x + 0.5 * book.second_order[e] * x * x
            for e in range(book.nnz_cross):
                xi = disp[book.active_level_uix[book.cross_factor_uix_i[e]]]
                xj = disp[book.active_level_uix[book.cross_factor_uix_j[e]]]
                want += book.cross_second_order[e] * xi.astype(np.float64) * xj
            assert np.allclose(got[0], want, rtol=1e-5, atol=1e-5), (nnz_per, cross_per)


# =========================================================================
# LAYER 1 — prange work balance
# =========================================================================


def _simulated_thread_loads(cost: np.ndarray, num_threads: int) -> np.ndarray:
    """
    Numba partitions a prange into contiguous, EQUAL-COUNT slices. Verified:

        T=4, n=48  ->  000000000000111111111111222222222222333333333333

    Replicating it here lets the balance guarantee be asserted deterministically
    on any host, including a single-core CI box.
    """
    base, remainder = divmod(cost.shape[0], num_threads)
    loads, start = [], 0
    for t in range(num_threads):
        length = base + (1 if t < remainder else 0)
        loads.append(cost[start : start + length].sum())
        start += length
    return np.array(loads)


def _skewed_book(*, num_positions=4_000, num_factors=256, heavy_fraction=0.05, seed=17):
    """5% of positions carry 80 entries, the rest carry 4 — heavy rows CONTIGUOUS."""
    rng = np.random.default_rng(seed)
    ids = np.arange(num_factors, dtype=np.int32)
    entries = np.full(num_positions, 4)
    entries[: int(num_positions * heavy_fraction)] = 80
    position_ids = np.repeat(np.arange(1, num_positions + 1, dtype=np.int64), entries)
    factor_ids = np.concatenate(
        [rng.choice(ids, n, replace=False) for n in entries]
    ).astype(np.int32)
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
        sensitivities=pl.DataFrame(
            {
                "position_id": position_ids,
                "factor_id": factor_ids,
                "first_order": rng.normal(0.0, 1e3, position_ids.size),
                "second_order": rng.normal(0.0, 1e4, position_ids.size),
            }
        ),
        factor_meta=meta,
        level_universe_factor_ids=ids,
    )
    disp = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(num_factors, 64)).astype(np.float32)
    )
    return book, disp


@pytest.mark.parametrize("num_threads", [2, 4, 8, 12, 24, 64])
def test_balanced_chunks_equalise_skewed_work(num_threads: int) -> None:
    """
    The guarantee. Unbalanced, this shape collapses to a 0.17 scaling ceiling at
    12 threads and swings run to run depending only on where the heavy rows sit.
    """
    book, _ = _skewed_book()
    cost = row_costs(book)

    unbalanced = _simulated_thread_loads(cost, num_threads)
    unbalanced_ceiling = unbalanced.mean() / unbalanced.max()

    offsets = balanced_chunk_offsets(
        book=book,
        num_chunks=default_num_chunks(
            num_positions=book.num_positions, thread_ceiling=num_threads
        ),
    )
    chunk_cost = np.array(
        [cost[offsets[i] : offsets[i + 1]].sum() for i in range(offsets.shape[0] - 1)]
    )
    balanced = _simulated_thread_loads(
        chunk_cost, min(num_threads, chunk_cost.shape[0])
    )
    balanced_ceiling = balanced.mean() / balanced.max()

    # A row is the atom of work, so a thread can overshoot the mean by at most
    # one row's cost. Assert against that physical bound rather than a flat
    # number: at high thread counts on a small book the granularity term
    # dominates and a flat 0.95 would be asserting something unachievable.
    mean_thread_load = cost.sum() / num_threads
    attainable = min(1.0, mean_thread_load / (mean_thread_load + cost.max()))
    assert balanced_ceiling >= 0.97 * attainable, (
        num_threads,
        balanced_ceiling,
        attainable,
    )
    assert balanced_ceiling > unbalanced_ceiling


@pytest.mark.parametrize("num_chunks", [1, 2, 7, 64, 1_000, 10_000])
def test_balanced_chunk_offsets_structure(num_chunks: int) -> None:
    book, _ = _skewed_book(num_positions=500)
    offsets = balanced_chunk_offsets(book=book, num_chunks=num_chunks)
    assert offsets.dtype == np.int32 and offsets.flags.c_contiguous
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == book.num_positions
    # Strictly increasing: an empty chunk burns a prange slot for no work.
    assert bool(np.all(np.diff(offsets) > 0))
    assert offsets.shape[0] - 1 <= max(num_chunks, 1)


def test_row_costs_are_strictly_positive() -> None:
    """
    Zero-cost rows would make the cost prefix sum flat and the partition
    ill-defined. PER_ROW_SCATTER_COST guarantees a floor, including for
    positions carrying no sensitivities at all.
    """
    assert bool(np.all(row_costs(frozen_book()) > 0.0))


@pytest.mark.parametrize("partition", ["single", "per_row", "balanced", "ragged"])
def test_chunking_is_output_invariant(partition: str) -> None:
    """
    Chunking is a scheduling decision and must not be observable in the result.
    Bit-identical, not merely close.
    """
    book, disp = _skewed_book(num_positions=600)
    m = book.num_positions
    reference = compute_sensitivity_pnl_from_displacement(book=book, displacement=disp)

    offsets = {
        "single": np.array([0, m], dtype=np.int32),
        "per_row": np.arange(m + 1, dtype=np.int32),
        "balanced": balanced_chunk_offsets(book=book, num_chunks=37),
        "ragged": np.array([0, 1, 2, 3, m // 2, m - 1, m], dtype=np.int32),
    }[partition]

    grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=disp.shape[1]
    )
    engine = SensitivityPnlEngine(
        book=book,
        out=grid.block_view(0),
        batch_size=disp.shape[1],
        chunk_offsets=offsets,
    )
    engine.absorb_displacement(
        displacement=disp, batch_len=disp.shape[1], scenario_offset=0
    )
    assert np.array_equal(grid.pnl, reference)


def test_chunk_offsets_must_cover_every_row() -> None:
    """
    A partition short of M leaves those rows unwritten — and `out` comes from
    `np.empty`, so the grid would ship uninitialised memory as PnL.
    """
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    with pytest.raises(ValueError, match="expected M="):
        SensitivityPnlEngine(
            book=book,
            out=grid.block_view(0),
            batch_size=2,
            chunk_offsets=np.array([0, 2], dtype=np.int32),
        )


def test_chunk_offsets_must_be_strictly_increasing() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    with pytest.raises(ValueError, match="strictly increasing"):
        SensitivityPnlEngine(
            book=book,
            out=grid.block_view(0),
            batch_size=2,
            chunk_offsets=np.array([0, 1, 1, 3], dtype=np.int32),
        )


def test_chunk_offsets_must_start_at_zero() -> None:
    book = frozen_book()
    grid = allocate_pnl_grid(block_position_uids=[book.position_uids], num_scenarios=2)
    with pytest.raises(ValueError, match="must start at 0"):
        SensitivityPnlEngine(
            book=book,
            out=grid.block_view(0),
            batch_size=2,
            chunk_offsets=np.array([1, 3], dtype=np.int32),
        )
