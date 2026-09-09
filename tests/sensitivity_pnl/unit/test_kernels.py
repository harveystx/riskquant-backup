# tests/sensitivity_pnl/unit/test_kernels.py
"""
Direct unit tests for the four compiled kernels in
`riskquant.sensitivity_pnl.kernels`, called in complete isolation from both
the compiler and either streaming engine.

Every other test in this package exercises a kernel through several layers of
orchestration (compile a book, construct an engine, absorb a batch). That is
the right thing for an end-to-end guarantee, but it means a kernel-level
defect could in principle hide behind a coincidentally-correct combination of
compiler and engine behaviour. These tests build the kernel's raw CSR/scratch
arguments by hand and call `kernel_*` directly, so the kernel's own arithmetic
is pinned on its own terms.

All four kernels are `@parallel_njit`, so every call here pays a one-time JIT
compilation cost the first time a given signature is hit; the file is written
to reuse small, deliberately-simple fixtures rather than multiply the number
of distinct call shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_LOG_ID,
    DISPLACEMENT_RELATIVE_ID,
)
from riskquant.sensitivity_pnl.engine import (
    accumulator_row_width,
    numba_thread_ceiling,
)
from riskquant.sensitivity_pnl.kernels.displacement import kernel_build_displacement
from riskquant.sensitivity_pnl.kernels.gather import kernel_gather_displacement
from riskquant.sensitivity_pnl.kernels.taylor import kernel_taylor_pnl
from riskquant.sensitivity_pnl.kernels.taylor_validity import (
    kernel_taylor_validity_terms,
)


def _acc(batch_size: int) -> np.ndarray:
    """A correctly-sized per-thread accumulator scratch, as the engines build it."""
    return np.empty(
        (numba_thread_ceiling(), accumulator_row_width(batch_size)),
        dtype=np.float64,
        order="C",
    )


# =========================================================================
# kernel_build_displacement
# =========================================================================


class TestKernelBuildDisplacement:
    def test_absolute_convention_is_l_minus_l0(self) -> None:
        base = np.array([100.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[105.0, 90.0, 100.0]]))
        uix = np.array([0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_ABSOLUTE_ID)], dtype=np.int8)
        out = np.empty((1, 3), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 3, 0, out)
        assert np.allclose(out[0], [5.0, -10.0, 0.0], atol=1e-5)

    def test_relative_convention_is_l_over_l0_minus_one(self) -> None:
        base = np.array([100.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[110.0, 50.0, 100.0]]))
        uix = np.array([0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_RELATIVE_ID)], dtype=np.int8)
        out = np.empty((1, 3), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 3, 0, out)
        assert np.allclose(out[0], [0.10, -0.50, 0.0], atol=1e-5)

    def test_log_convention_is_ln_of_l_over_l0(self) -> None:
        base = np.array([100.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[100.0 * np.e, 100.0 / np.e, 100.0]]))
        uix = np.array([0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_LOG_ID)], dtype=np.int8)
        out = np.empty((1, 3), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 3, 0, out)
        assert np.allclose(out[0], [1.0, -1.0, 0.0], atol=1e-5)

    def test_multiple_factors_get_independent_conventions(self) -> None:
        base = np.array([100.0, 2.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[110.0], [2.0 * np.e]]))
        uix = np.array([0, 1], dtype=np.int32)
        conventions = np.array(
            [int(DISPLACEMENT_RELATIVE_ID), int(DISPLACEMENT_LOG_ID)], dtype=np.int8
        )
        out = np.empty((2, 1), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 1, 0, out)
        assert np.isclose(float(out[0, 0]), 0.10, atol=1e-5)
        assert np.isclose(float(out[1, 0]), 1.0, atol=1e-5)

    def test_active_level_uix_compacts_a_sparse_universe(self) -> None:
        """active_level_uix = [2, 0] means the book's active factor 0 maps to
        universe row 2 and active factor 1 maps to universe row 0 -- output
        row order follows the ACTIVE axis, not the universe axis."""
        base = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[11.0], [22.0], [33.0]]))
        uix = np.array([2, 0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_ABSOLUTE_ID)] * 2, dtype=np.int8)
        out = np.empty((2, 1), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 1, 0, out)
        assert np.isclose(float(out[0, 0]), 3.0, atol=1e-5)  # universe row 2: 33-30
        assert np.isclose(float(out[1, 0]), 1.0, atol=1e-5)  # universe row 0: 11-10

    def test_levels_offset_selects_the_right_column_window(self) -> None:
        base = np.array([100.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[100.0, 105.0, 110.0, 115.0]]))
        uix = np.array([0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_ABSOLUTE_ID)], dtype=np.int8)
        out = np.empty((1, 2), dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 2, 1, out)
        assert np.allclose(out[0], [5.0, 10.0], atol=1e-5)  # columns [1, 3)

    def test_batch_len_shorter_than_out_width_only_writes_the_prefix(self) -> None:
        base = np.array([100.0], dtype=np.float64)
        shocked = np.ascontiguousarray(np.array([[105.0, 999.0]]))
        uix = np.array([0], dtype=np.int32)
        conventions = np.array([int(DISPLACEMENT_ABSOLUTE_ID)], dtype=np.int8)
        out = np.full((1, 2), -999.0, dtype=np.float32, order="C")
        kernel_build_displacement(base, shocked, uix, conventions, 1, 0, out)
        assert np.isclose(float(out[0, 0]), 5.0, atol=1e-5)
        assert float(out[0, 1]) == -999.0  # untouched


# =========================================================================
# kernel_gather_displacement
# =========================================================================


class TestKernelGatherDisplacement:
    def test_pure_passthrough_when_active_axis_equals_source_axis(self) -> None:
        source = np.ascontiguousarray(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        uix = np.array([0, 1], dtype=np.int32)
        out = np.empty((2, 2), dtype=np.float32, order="C")
        kernel_gather_displacement(source, uix, 2, 0, out)
        assert np.array_equal(out, source)

    def test_compaction_selects_and_reorders_rows(self) -> None:
        source = np.ascontiguousarray(
            np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        )
        uix = np.array([2, 0], dtype=np.int32)  # book's factor 0 -> source row 2, etc
        out = np.empty((2, 1), dtype=np.float32, order="C")
        kernel_gather_displacement(source, uix, 1, 0, out)
        assert out[0, 0] == 30.0
        assert out[1, 0] == 10.0

    def test_source_offset_selects_the_right_column_window(self) -> None:
        source = np.ascontiguousarray(
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        )
        uix = np.array([0], dtype=np.int32)
        out = np.empty((1, 2), dtype=np.float32, order="C")
        kernel_gather_displacement(source, uix, 2, 2, out)
        assert np.array_equal(out[0], [3.0, 4.0])

    def test_a_row_the_book_never_touches_is_never_read(self) -> None:
        """active_level_uix omitting a source row must not cause any error --
        that row simply never appears in `out`."""
        source = np.ascontiguousarray(
            np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
        )
        uix = np.array([0, 2], dtype=np.int32)  # row 1 (NaN) is skipped entirely
        out = np.empty((2, 1), dtype=np.float32, order="C")
        kernel_gather_displacement(source, uix, 1, 0, out)
        assert bool(np.isfinite(out).all())


# =========================================================================
# kernel_taylor_pnl
# =========================================================================


def _run_taylor_pnl(
    *,
    position_offsets,
    factor_uix,
    first_order,
    second_order,
    cross_offsets=None,
    cross_factor_uix_i=None,
    cross_factor_uix_j=None,
    cross_second_order=None,
    displacement,
    chunk_offsets=None,
    batch_len=None,
):
    m = position_offsets.shape[0] - 1
    s = displacement.shape[1]
    batch_len = s if batch_len is None else batch_len
    chunk_offsets = (
        np.array([0, m], dtype=np.int32) if chunk_offsets is None else chunk_offsets
    )
    cross_offsets = (
        np.zeros(m + 1, dtype=np.int32) if cross_offsets is None else cross_offsets
    )
    cross_factor_uix_i = (
        np.empty(0, dtype=np.int32)
        if cross_factor_uix_i is None
        else cross_factor_uix_i
    )
    cross_factor_uix_j = (
        np.empty(0, dtype=np.int32)
        if cross_factor_uix_j is None
        else cross_factor_uix_j
    )
    cross_second_order = (
        np.empty(0, dtype=np.float64)
        if cross_second_order is None
        else cross_second_order
    )
    out = np.empty((m, s), dtype=np.float32, order="C")
    kernel_taylor_pnl(
        chunk_offsets,
        position_offsets,
        factor_uix,
        first_order,
        second_order,
        cross_offsets,
        cross_factor_uix_i,
        cross_factor_uix_j,
        cross_second_order,
        displacement,
        _acc(batch_len),
        batch_len,
        0,
        out,
    )
    return out


class TestKernelTaylorPnl:
    def test_pure_linear_single_entry(self) -> None:
        """PnL = d1 * x, gamma = 0."""
        displacement = np.ascontiguousarray(np.array([[0.5, -0.5]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([100.0]),
            second_order=np.array([0.0]),
            displacement=displacement,
        )
        assert np.allclose(out[0], [50.0, -50.0], atol=1e-3)

    def test_pure_quadratic_single_entry(self) -> None:
        """PnL = 0.5 * d2 * x^2."""
        displacement = np.ascontiguousarray(np.array([[2.0, -2.0]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([0.0]),
            second_order=np.array([10.0]),
            displacement=displacement,
        )
        # 0.5 * 10 * 4 = 20, same for +2 and -2 (even power)
        assert np.allclose(out[0], [20.0, 20.0], atol=1e-3)

    def test_mixed_delta_and_gamma(self) -> None:
        displacement = np.ascontiguousarray(np.array([[3.0]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([10.0]),
            second_order=np.array([4.0]),
            displacement=displacement,
        )
        # 10*3 + 0.5*4*9 = 30 + 18 = 48
        assert np.isclose(float(out[0, 0]), 48.0, atol=1e-3)

    def test_multiple_diagonal_entries_sum(self) -> None:
        displacement = np.ascontiguousarray(
            np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        )
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 3], dtype=np.int32),
            factor_uix=np.array([0, 1, 2], dtype=np.int32),
            first_order=np.array([1.0, 10.0, 100.0]),
            second_order=np.array([0.0, 0.0, 0.0]),
            displacement=displacement,
        )
        assert np.isclose(float(out[0, 0]), 111.0, atol=1e-3)

    def test_cross_term_is_the_full_mixed_partial_not_halved(self) -> None:
        """PnL = c * x_i * x_j, exactly, no 1/2 factor -- see contracts.py's
        pinned unit contract."""
        displacement = np.ascontiguousarray(np.array([[2.0], [3.0]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0),
            second_order=np.empty(0),
            cross_offsets=np.array([0, 1], dtype=np.int32),
            cross_factor_uix_i=np.array([0], dtype=np.int32),
            cross_factor_uix_j=np.array([1], dtype=np.int32),
            cross_second_order=np.array([5.0]),
            displacement=displacement,
        )
        assert np.isclose(float(out[0, 0]), 5.0 * 2.0 * 3.0, atol=1e-3)

    def test_a_row_with_no_entries_at_all_yields_exact_zero(self) -> None:
        """A declared position with no diagonal and no cross entries: the
        zero-seed / seed-from-first-entry logic must still yield exact 0.0."""
        displacement = np.ascontiguousarray(np.array([[7.0]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0),
            second_order=np.empty(0),
            displacement=displacement,
        )
        assert float(out[0, 0]) == 0.0

    def test_diagonal_and_cross_together(self) -> None:
        displacement = np.ascontiguousarray(np.array([[1.0], [2.0]], dtype=np.float32))
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([100.0]),
            second_order=np.array([0.0]),
            cross_offsets=np.array([0, 1], dtype=np.int32),
            cross_factor_uix_i=np.array([0], dtype=np.int32),
            cross_factor_uix_j=np.array([1], dtype=np.int32),
            cross_second_order=np.array([10.0]),
            displacement=displacement,
        )
        # delta: 100*1 = 100 ; cross: 10*1*2 = 20 ; total 120
        assert np.isclose(float(out[0, 0]), 120.0, atol=1e-3)

    @pytest.mark.parametrize("entry_count", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_diagonal_entry_blocking_boundary_sweep(self, entry_count: int) -> None:
        """The 4-wide entry-blocking optimisation in the kernel has a seed arm,
        a 4-wide main loop, and a remainder loop; this sweeps entry counts
        across and around that boundary (1 through 9) and checks the sum
        against a plain Python reference for each."""
        rng = np.random.default_rng(entry_count)
        first_order = rng.normal(0, 10, entry_count)
        second_order = rng.normal(0, 5, entry_count)
        x = 0.37
        displacement = np.ascontiguousarray(
            np.full((entry_count, 1), x, dtype=np.float32)
        )
        out = _run_taylor_pnl(
            position_offsets=np.array([0, entry_count], dtype=np.int32),
            factor_uix=np.arange(entry_count, dtype=np.int32),
            first_order=first_order,
            second_order=second_order,
            displacement=displacement,
        )
        expected = sum(
            first_order[i] * x + 0.5 * second_order[i] * x * x
            for i in range(entry_count)
        )
        assert np.isclose(float(out[0, 0]), expected, rtol=1e-4)

    @pytest.mark.parametrize("cross_count", [1, 2, 3, 4, 5])
    def test_cross_entry_blocking_boundary_sweep(self, cross_count: int) -> None:
        """Same idea for the cross block's 2-wide blocking."""
        rng = np.random.default_rng(100 + cross_count)
        coefficients = rng.normal(0, 5, cross_count)
        # cross_count pairs among 2*cross_count distinct factors
        displacement = np.ascontiguousarray(
            rng.normal(0, 1, (2 * cross_count, 1)).astype(np.float32)
        )
        out = _run_taylor_pnl(
            position_offsets=np.array([0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0),
            second_order=np.empty(0),
            cross_offsets=np.array([0, cross_count], dtype=np.int32),
            cross_factor_uix_i=np.arange(0, 2 * cross_count, 2, dtype=np.int32),
            cross_factor_uix_j=np.arange(1, 2 * cross_count, 2, dtype=np.int32),
            cross_second_order=coefficients,
            displacement=displacement,
        )
        expected = sum(
            coefficients[k] * displacement[2 * k, 0] * displacement[2 * k + 1, 0]
            for k in range(cross_count)
        )
        assert np.isclose(float(out[0, 0]), expected, rtol=1e-3)

    def test_result_is_independent_of_chunk_partition(self) -> None:
        """Bit-identical output regardless of how the row axis is chunked --
        the guarantee `scheduling.balanced_chunk_offsets` depends on."""
        rng = np.random.default_rng(7)
        m, k = 20, 6
        position_offsets = np.arange(0, m * 3 + 1, 3, dtype=np.int32)
        factor_uix = np.tile(np.arange(3, dtype=np.int32), m)
        first_order = rng.normal(0, 10, m * 3)
        second_order = rng.normal(0, 5, m * 3)
        displacement = np.ascontiguousarray(
            rng.normal(0, 0.1, (k, 4)).astype(np.float32)
        )
        reference = _run_taylor_pnl(
            position_offsets=position_offsets,
            factor_uix=factor_uix,
            first_order=first_order,
            second_order=second_order,
            displacement=displacement,
            chunk_offsets=np.array([0, m], dtype=np.int32),
        )
        per_row = _run_taylor_pnl(
            position_offsets=position_offsets,
            factor_uix=factor_uix,
            first_order=first_order,
            second_order=second_order,
            displacement=displacement,
            chunk_offsets=np.arange(m + 1, dtype=np.int32),
        )
        ragged = _run_taylor_pnl(
            position_offsets=position_offsets,
            factor_uix=factor_uix,
            first_order=first_order,
            second_order=second_order,
            displacement=displacement,
            chunk_offsets=np.array([0, 1, 2, 3, 10, 19, m], dtype=np.int32),
        )
        assert np.array_equal(reference, per_row)
        assert np.array_equal(reference, ragged)

    def test_result_is_independent_of_batch_len_within_capacity(self) -> None:
        """Evaluating a NARROWER batch_len over the same displacement buffer
        must reproduce a prefix of the full-batch result exactly."""
        rng = np.random.default_rng(3)
        displacement = np.ascontiguousarray(
            rng.normal(0, 0.2, (2, 10)).astype(np.float32)
        )
        full = _run_taylor_pnl(
            position_offsets=np.array([0, 2], dtype=np.int32),
            factor_uix=np.array([0, 1], dtype=np.int32),
            first_order=np.array([5.0, -3.0]),
            second_order=np.array([1.0, 2.0]),
            displacement=displacement,
            batch_len=10,
        )
        narrow = _run_taylor_pnl(
            position_offsets=np.array([0, 2], dtype=np.int32),
            factor_uix=np.array([0, 1], dtype=np.int32),
            first_order=np.array([5.0, -3.0]),
            second_order=np.array([1.0, 2.0]),
            displacement=displacement,
            batch_len=4,
        )
        assert np.array_equal(full[:, :4], narrow[:, :4])


# =========================================================================
# kernel_taylor_validity_terms
# =========================================================================


def _run_validity(
    *,
    position_offsets,
    factor_uix,
    first_order,
    second_order,
    cross_offsets=None,
    cross_factor_uix_i=None,
    cross_factor_uix_j=None,
    cross_second_order=None,
    displacement,
    chunk_offsets=None,
    batch_len=None,
):
    m = position_offsets.shape[0] - 1
    s = displacement.shape[1]
    batch_len = s if batch_len is None else batch_len
    chunk_offsets = (
        np.array([0, m], dtype=np.int32) if chunk_offsets is None else chunk_offsets
    )
    cross_offsets = (
        np.zeros(m + 1, dtype=np.int32) if cross_offsets is None else cross_offsets
    )
    cross_factor_uix_i = (
        np.empty(0, dtype=np.int32)
        if cross_factor_uix_i is None
        else cross_factor_uix_i
    )
    cross_factor_uix_j = (
        np.empty(0, dtype=np.int32)
        if cross_factor_uix_j is None
        else cross_factor_uix_j
    )
    cross_second_order = (
        np.empty(0, dtype=np.float64)
        if cross_second_order is None
        else cross_second_order
    )
    linear_out = np.empty((m, s), dtype=np.float32, order="C")
    quadratic_out = np.empty((m, s), dtype=np.float32, order="C")
    kernel_taylor_validity_terms(
        chunk_offsets,
        position_offsets,
        factor_uix,
        first_order,
        second_order,
        cross_offsets,
        cross_factor_uix_i,
        cross_factor_uix_j,
        cross_second_order,
        displacement,
        _acc(batch_len),
        _acc(batch_len),
        batch_len,
        0,
        linear_out,
        quadratic_out,
    )
    return linear_out, quadratic_out


class TestKernelTaylorValidityTerms:
    def test_pure_delta_gives_all_linear_zero_quadratic(self) -> None:
        displacement = np.ascontiguousarray(np.array([[2.0]], dtype=np.float32))
        linear, quadratic = _run_validity(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([15.0]),
            second_order=np.array([0.0]),
            displacement=displacement,
        )
        assert np.isclose(float(linear[0, 0]), 30.0, atol=1e-3)
        assert float(quadratic[0, 0]) == 0.0

    def test_pure_gamma_gives_all_quadratic_zero_linear(self) -> None:
        displacement = np.ascontiguousarray(np.array([[2.0]], dtype=np.float32))
        linear, quadratic = _run_validity(
            position_offsets=np.array([0, 1], dtype=np.int32),
            factor_uix=np.array([0], dtype=np.int32),
            first_order=np.array([0.0]),
            second_order=np.array([10.0]),
            displacement=displacement,
        )
        assert float(linear[0, 0]) == 0.0
        assert np.isclose(float(quadratic[0, 0]), 20.0, atol=1e-3)  # 0.5*10*4

    def test_cross_term_lands_entirely_in_quadratic(self) -> None:
        displacement = np.ascontiguousarray(np.array([[2.0], [3.0]], dtype=np.float32))
        linear, quadratic = _run_validity(
            position_offsets=np.array([0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0),
            second_order=np.empty(0),
            cross_offsets=np.array([0, 1], dtype=np.int32),
            cross_factor_uix_i=np.array([0], dtype=np.int32),
            cross_factor_uix_j=np.array([1], dtype=np.int32),
            cross_second_order=np.array([5.0]),
            displacement=displacement,
        )
        assert float(linear[0, 0]) == 0.0
        assert np.isclose(float(quadratic[0, 0]), 30.0, atol=1e-3)

    def test_empty_row_gives_exact_zero_for_both_terms(self) -> None:
        displacement = np.ascontiguousarray(np.array([[7.0]], dtype=np.float32))
        linear, quadratic = _run_validity(
            position_offsets=np.array([0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0),
            second_order=np.empty(0),
            displacement=displacement,
        )
        assert float(linear[0, 0]) == 0.0
        assert float(quadratic[0, 0]) == 0.0

    def test_sum_of_linear_and_quadratic_matches_kernel_taylor_pnl_exactly(
        self,
    ) -> None:
        """The identity `linear + quadratic == kernel_taylor_pnl(...)` is the
        core cross-check both this file and the golden master rely on; pinned
        here directly at the kernel level, with no engine or compiler in
        between either call."""
        rng = np.random.default_rng(11)
        m = 5
        entries_per_row = 3
        position_offsets = np.arange(
            0, (m + 1) * entries_per_row, entries_per_row, dtype=np.int32
        )
        factor_uix = np.tile(np.arange(entries_per_row, dtype=np.int32), m)
        first_order = rng.normal(0, 10, m * entries_per_row)
        second_order = rng.normal(0, 5, m * entries_per_row)
        cross_offsets = np.zeros(m + 1, dtype=np.int32)
        cross_offsets[1:] = np.arange(1, m + 1)  # one cross entry per row
        cross_i = np.zeros(m, dtype=np.int32)
        cross_j = np.ones(m, dtype=np.int32)
        cross_value = rng.normal(0, 3, m)
        displacement = np.ascontiguousarray(
            rng.normal(0, 0.3, (entries_per_row, 7)).astype(np.float32)
        )

        linear, quadratic = _run_validity(
            position_offsets=position_offsets,
            factor_uix=factor_uix,
            first_order=first_order,
            second_order=second_order,
            cross_offsets=cross_offsets,
            cross_factor_uix_i=cross_i,
            cross_factor_uix_j=cross_j,
            cross_second_order=cross_value,
            displacement=displacement,
        )
        pnl = _run_taylor_pnl(
            position_offsets=position_offsets,
            factor_uix=factor_uix,
            first_order=first_order,
            second_order=second_order,
            cross_offsets=cross_offsets,
            cross_factor_uix_i=cross_i,
            cross_factor_uix_j=cross_j,
            cross_second_order=cross_value,
            displacement=displacement,
        )
        recovered = linear.astype(np.float64) + quadratic.astype(np.float64)
        assert np.allclose(recovered, pnl.astype(np.float64), rtol=1e-3, atol=1e-3)
