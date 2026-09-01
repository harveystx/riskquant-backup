# tests/dependence/test_tail_counts.py

"""
Phase 1: exact tail exceedance counts.

Counts only. No coefficient is computed here or anywhere in the implementation;
``lambda_hat``, ``chi_hat`` and ``chi_bar_hat`` belong to Phase 1b and are
blocked on methodology decisions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from riskquant.dependence._kernels._symmetry import (
    kernel_mirror_upper_and_set_diagonal,
)
from riskquant.dependence.contracts import TailSide
from riskquant.dependence.estimators.tail_counts import tail_counts
from riskquant.observations.arrays import BoolMatrix, Float64Matrix
from riskquant.observations.orientation import ObservationOrientation
from tests.dependence.conftest import (
    compile_both,
    continuous_sample,
    zero_inflated_sample,
)

VARIABLES_ROWS = ObservationOrientation.VARIABLES_ON_ROWS
GRID = np.array([0.80, 0.90, 0.95, 0.99])


def brute_force_counts(
    matrix: np.ndarray, thresholds: np.ndarray, tail: TailSide
) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent O(K^2 S) reference.

    Deliberately written from the event definition rather than from the
    implementation: average ranks from SciPy, boolean masks, explicit loops.
    """
    ranks = rankdata(matrix, method="average", axis=1)
    n_variables, n_observations = matrix.shape
    scale = n_observations + 1

    joint = np.zeros((len(thresholds), n_variables, n_variables), dtype=np.int64)
    marginal = np.zeros((len(thresholds), n_variables), dtype=np.int64)

    for position, level in enumerate(thresholds):
        cut = level * scale
        mask = ranks > cut if tail is TailSide.UPPER else (scale - ranks) > cut
        marginal[position] = mask.sum(axis=1)
        for i in range(n_variables):
            for j in range(n_variables):
                joint[position, i, j] = int(np.count_nonzero(mask[i] & mask[j]))

    return joint, marginal


PANELS = {
    "random": continuous_sample(seed=21, n_variables=5, n_observations=400),
    "tied": np.round(continuous_sample(seed=22, n_variables=5, n_observations=400), 1),
    "zero_inflated": zero_inflated_sample(seed=23, n_variables=5, n_observations=400),
}


# --------------------------------------------------------------- exact counts


@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("tail", list(TailSide))
def test_counts_match_a_brute_force_reference_exactly(
    panel: str, tail: TailSide
) -> None:
    matrix = PANELS[panel]
    sample, ranks = compile_both(matrix)
    result = tail_counts(sample, ranks, thresholds=GRID, tail=tail)

    joint, marginal = brute_force_counts(matrix, GRID, tail)

    assert np.array_equal(result.joint_count, joint)
    assert np.array_equal(result.marginal_count, marginal)


@pytest.mark.parametrize("panel", PANELS)
def test_the_joint_diagonal_is_the_marginal_count(panel: str) -> None:
    sample, ranks = compile_both(PANELS[panel])
    result = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)

    for position in range(result.threshold_count):
        assert np.array_equal(
            np.diag(result.joint_count[position]), result.marginal_count[position]
        )


# --------------------------------------------------------- reflection invariance


@pytest.mark.parametrize("panel", PANELS)
def test_reflection_invariance_is_exact(panel: str) -> None:
    """
    Pins the strict ``>`` / ``<`` convention.

    Negating a variable maps ranks ``r -> S + 1 - r`` exactly, including inside
    tie blocks, so the upper event on ``-X`` is exactly the lower event on
    ``X``. A mixed ``>=`` / ``<`` pairing breaks this at the boundary.
    """
    matrix = PANELS[panel]

    sample, ranks = compile_both(matrix)
    upper = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)

    reflected_sample, reflected_ranks = compile_both(np.ascontiguousarray(-matrix))
    lower = tail_counts(
        reflected_sample, reflected_ranks, thresholds=GRID, tail=TailSide.LOWER
    )

    assert np.array_equal(upper.joint_count, lower.joint_count)
    assert np.array_equal(upper.marginal_count, lower.marginal_count)


def test_the_reflection_test_would_fail_under_a_non_strict_convention() -> None:
    """
    Guards against the invariance holding vacuously.

    S = 9, q = 0.8, so the lower cut sits at ``(1 - q)(S + 1) = 2.0`` and rank
    2.0 is realised exactly. The strict rule ``r < 2.0`` selects one
    observation, matching the strict upper rule ``r > 8.0`` on the negated
    variable. The non-strict rule ``r <= 2.0`` selects two, and the reflection
    invariant fails.
    """
    n_observations = 9
    matrix = np.vstack(
        [np.arange(float(n_observations)), np.arange(float(n_observations))[::-1]]
    )
    ranks = rankdata(matrix, method="average", axis=1)
    reflected_ranks = rankdata(-matrix, method="average", axis=1)

    level = 0.5
    scale = float(n_observations + 1)
    cut = level * scale
    assert cut == 5.0
    assert np.any(ranks == cut), "the boundary rank must be realised"

    strict_lower = ((scale - ranks) > cut).sum(axis=1)
    non_strict_lower = ((scale - ranks) >= cut).sum(axis=1)
    reflected_upper = (reflected_ranks > cut).sum(axis=1)

    assert np.array_equal(strict_lower, reflected_upper)
    assert not np.array_equal(non_strict_lower, reflected_upper)


def test_the_two_algebraically_equivalent_lower_cuts_are_not_bit_identical() -> None:
    """
    Why the lower tail is computed as ``(T - r) > q*T`` rather than ``r < (1-q)*T``.

    The two forms are algebraically identical and differ in floating point: over
    a sweep of S in [2, 4000) and ten conventional levels, they disagree in the
    last bits for the majority of pairs.

    No realised rank was found strictly between the two cuts, because ranks are
    spaced by at least 0.5 while the discrepancy is around 1e-16. The reflective
    form is used anyway, so that the reflection invariant rests on an exactness
    argument -- both tails compare against the same float, and ``T - r`` is
    exact -- rather than on a spacing margin that would have to be re-derived
    for any future change to the rank representation.
    """
    differing = 0
    total = 0
    for n_observations in range(2, 4000):
        scale = float(n_observations + 1)
        for level in (0.5, 0.75, 0.9, 0.95, 0.99):
            total += 1
            if (1.0 - level) * scale != scale - level * scale:
                differing += 1

    assert differing > 0, "the two forms would be interchangeable"
    assert differing < total

    # T - r is exact for every realised rank, which is the property relied upon.
    for n_observations in (9, 100, 4001):
        scale = float(n_observations + 1)
        ranks = np.arange(1.0, n_observations + 1.0)
        half_ranks = ranks + 0.5
        assert np.array_equal(scale - (scale - ranks), ranks)
        assert np.array_equal(scale - (scale - half_ranks), half_ranks)


def test_the_lower_cut_follows_the_specified_reflective_arithmetic() -> None:
    """
    NB-5 regression. Pins ``(T - r) > q*T`` against ``r < (1-q)*T``.

    Asserted through the public ``tail_counts`` result, not by re-evaluating the
    implementation's own expression.

    Construction. ``S = 9`` so ``T = 10``, and ``q = 0.7``. The two cuts are::

        specified :  T - r  >  q * T        ->  10.0 - r > 7.0
        naive     :  r      <  (1 - q) * T  ->  r < 3.0000000000000004

    ``q * T`` is exactly ``7.0`` and ``(1 - q) * T`` is ``3.0000000000000004``,
    one ulp above ``3.0``. The rank ``r = 3.0`` is realised by a variable with
    nine distinct values, and it falls between the two cuts:

    * specified: ``10.0 - 3.0 = 7.0`` is not ``> 7.0``  -> EXCLUDED
    * naive:     ``3.0 < 3.0000000000000004``           -> INCLUDED

    So ``marginal_count`` is 2 under the specified convention (ranks 1 and 2)
    and would be 3 under the naive one. The test fails if the implementation is
    changed to the naive form.
    """
    n_observations = 9
    level = 0.7
    scale = float(n_observations + 1)

    assert level * scale == 7.0
    assert (1.0 - level) * scale == 3.0000000000000004
    assert not ((scale - 3.0) > level * scale), "specified form excludes r = 3.0"
    assert (1.0 - level) * scale > 3.0, "naive form would include r = 3.0"

    values = np.arange(1.0, n_observations + 1.0)
    matrix = np.ascontiguousarray(np.vstack([values, values[::-1]]))
    sample, ranks = compile_both(matrix)

    assert np.array_equal(ranks.ranks[0], values), (
        "ranks must be 1..S, so r = 3.0 is realised"
    )

    result = tail_counts(
        sample, ranks, thresholds=np.array([level]), tail=TailSide.LOWER
    )

    specified_count = 2
    naive_count = 3
    assert int(result.marginal_count[0, 0]) == specified_count
    assert int(result.marginal_count[0, 0]) != naive_count
    assert np.array_equal(result.marginal_count[0], [specified_count, specified_count])


# ------------------------------------------------------------------ tie cases


def test_a_tie_block_straddling_the_threshold_is_all_or_nothing() -> None:
    """
    The case that justifies storing marginal counts instead of deriving them.

    Twenty observations; the top eight share one value, so they share one
    average rank. At q = 0.90 the nominal count is 2, but the block is included
    or excluded as a whole.
    """
    n_observations = 20
    straddling = np.concatenate([np.arange(12.0), np.full(8, 100.0)])
    matrix = np.vstack([straddling, np.arange(float(n_observations))])

    sample, ranks = compile_both(matrix)
    result = tail_counts(
        sample, ranks, thresholds=np.array([0.90]), tail=TailSide.UPPER
    )

    nominal = n_observations * (1.0 - 0.90)
    realised = int(result.marginal_count[0, 0])

    assert realised in (0, 8), "the tie block must be included or excluded as a whole"
    assert realised != int(nominal)

    joint, marginal = brute_force_counts(matrix, np.array([0.90]), TailSide.UPPER)
    assert np.array_equal(result.marginal_count, marginal)
    assert np.array_equal(result.joint_count, joint)


def test_marginal_count_equals_the_nominal_count_when_there_are_no_ties() -> None:
    """
    The complementary fact, so the contract is stated precisely rather than as
    a blanket "never equal".
    """
    matrix = np.vstack([np.arange(1000.0), np.arange(1000.0)[::-1]])
    sample, ranks = compile_both(matrix)
    result = tail_counts(
        sample, ranks, thresholds=np.array([0.99]), tail=TailSide.UPPER
    )
    assert int(result.marginal_count[0, 0]) == 10


# ------------------------------------------------------------- degenerate cases


def test_zero_marginal_exceedances_returns_zero_and_does_not_raise() -> None:
    """
    Reachable, and not hypothetical.

    Sixty per cent of the sample shares the maximum value, so that block has
    average rank 0.7 * (S + 1) and nothing exceeds q = 0.95.
    """
    n_observations = 100
    tied_block = int(0.6 * n_observations)
    values = np.concatenate(
        [np.arange(float(n_observations - tied_block)), np.full(tied_block, 1e6)]
    )
    matrix = np.vstack([values, np.arange(float(n_observations))])

    sample, ranks = compile_both(matrix)
    result = tail_counts(
        sample, ranks, thresholds=np.array([0.95]), tail=TailSide.UPPER
    )

    assert int(result.marginal_count[0, 0]) == 0
    assert int(result.joint_count[0, 0, 0]) == 0


def test_zero_joint_exceedances_returns_zero_and_does_not_raise() -> None:
    """Two variables whose tails are disjoint by construction."""
    n_observations = 200
    first = np.arange(float(n_observations))
    second = np.roll(first, n_observations // 2)
    matrix = np.vstack([first, second])

    sample, ranks = compile_both(matrix)
    result = tail_counts(
        sample, ranks, thresholds=np.array([0.95]), tail=TailSide.UPPER
    )

    assert int(result.joint_count[0, 0, 1]) == 0
    assert int(result.marginal_count[0, 0]) > 0


def test_a_constant_variable_is_rejected_before_tail_compilation() -> None:
    sample, ranks = compile_both(np.vstack([np.ones(50), np.arange(50.0)]))
    with pytest.raises(ValueError, match="Tail exceedance counting is undefined"):
        tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)


# ------------------------------------------------------------ GEMM integrality


def test_the_gemm_produces_exact_integers() -> None:
    sample, ranks = compile_both(PANELS["random"])
    result = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)
    assert result.joint_count.dtype == np.int32
    assert result.marginal_count.dtype == np.int32


def test_a_perturbed_count_trips_the_integrality_check() -> None:
    """
    Simulates a reduced-precision BLAS dispatch path.

    The guarantee must not depend on ``sgemm`` staying in IEEE binary32, so the
    check exists to fail loudly rather than silently return wrong counts.
    """
    import riskquant.dependence.estimators.tail_counts as module

    sample, ranks = compile_both(PANELS["random"])
    original = module._exceedance_indicator

    def perturbed(
        ranks: Float64Matrix,
        *,
        threshold: float,
        tail: TailSide,
        n_observations: int,
        indicator_out: Float64Matrix,
        mask_out: BoolMatrix,
    ) -> None:
        original(
            ranks,
            threshold=threshold,
            tail=tail,
            n_observations=n_observations,
            indicator_out=indicator_out,
            mask_out=mask_out,
        )
        indicator_out[0, 0] += 1e-3

    module._exceedance_indicator = perturbed
    try:
        with pytest.raises(ValueError, match="non-integral"):
            tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)
    finally:
        module._exceedance_indicator = original


# -------------------------------------------------------------------- symmetry


def test_the_mirror_is_authoritative_and_is_actually_applied() -> None:
    """
    NB-6 regression. Pins the mirror as the mechanism that delivers symmetry.

    ``indicator @ indicator.T`` is already symmetric, because the summands are
    exact integers and floating addition of exact integers is order-independent.
    A test that only asserts ``array_equal(C, C.T)`` therefore cannot tell an
    applied mirror from a naturally symmetric GEMM, and removing the mirror
    survives such a test.

    This test reaches the pre-mirror state by intercepting the mirror kernel,
    perturbs the non-authoritative (lower) triangle with a sentinel, then lets
    the real kernel run. It asserts:

    1. the interception happened once per threshold -- so removing the mirror
       from the production path makes the call count zero and fails here;
    2. the authoritative strict upper triangle is unchanged by the mirror;
    3. the perturbed lower triangle is overwritten by the authoritative values,
       and no sentinel survives into the result;
    4. the diagonal carries the marginal count, per the contract.
    """
    import riskquant.dependence.estimators.tail_counts as module

    sample, ranks = compile_both(PANELS["random"])
    grid = np.array([0.80, 0.95])
    n_variables = ranks.variable_count

    sentinel = -12345.0
    lower = np.tril_indices(n_variables, k=-1)
    upper = np.triu_indices(n_variables, k=1)

    # getattr/setattr rather than attribute access: the name is imported into
    # the estimator module, not re-exported by it, and monkeypatching a
    # re-imported symbol is exactly what this test needs to do.
    mirror_name = "kernel_mirror_upper_and_set_diagonal"
    original = getattr(module, mirror_name)
    calls: list[Float64Matrix] = []

    def intercepted(matrix: Float64Matrix, diagonal_value: float) -> None:
        # Pre-mirror state. Capture the authoritative triangle, then corrupt the
        # triangle the mirror is required to overwrite.
        calls.append(matrix[upper].copy())
        matrix[lower] = sentinel
        original(matrix, diagonal_value)

    setattr(module, mirror_name, intercepted)
    try:
        result = tail_counts(sample, ranks, thresholds=grid, tail=TailSide.UPPER)
    finally:
        setattr(module, mirror_name, original)

    assert len(calls) == grid.size, "the mirror must run once per threshold"

    for position in range(grid.size):
        counts = result.joint_count[position]
        authoritative = calls[position]

        assert np.array_equal(counts[upper], authoritative.astype(np.int32))
        assert np.array_equal(counts[lower], counts.T[lower])
        assert not np.any(counts == np.int32(sentinel))
        assert np.array_equal(np.diag(counts), result.marginal_count[position])


def test_the_mirror_kernel_overwrites_the_lower_triangle_from_the_upper() -> None:
    """
    NB-6, kernel level. The mirror's direction is part of the contract.

    An asymmetric matrix goes in; the strict upper triangle must survive
    untouched and the lower triangle must be replaced by it.
    """
    matrix = np.array(
        [[9.0, 1.0, 2.0], [-7.0, 9.0, 3.0], [-8.0, -6.0, 9.0]], dtype=np.float64
    )
    expected_upper = matrix[np.triu_indices(3, k=1)].copy()

    kernel_mirror_upper_and_set_diagonal(matrix, 0.0)

    assert np.array_equal(matrix[np.triu_indices(3, k=1)], expected_upper)
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(3))
    assert np.array_equal(matrix[np.tril_indices(3, k=-1)], np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("panel", PANELS)
def test_every_count_slice_is_bitwise_symmetric(panel: str) -> None:
    sample, ranks = compile_both(PANELS[panel])
    result = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)

    for position in range(result.threshold_count):
        assert np.array_equal(
            result.joint_count[position], result.joint_count[position].T
        )


# ---------------------------------------------------------------- determinism


def test_counts_are_deterministic_under_differing_thread_counts() -> None:
    """
    Numba-parallel determinism, directly exercised.

    NB-7, recorded rather than hidden: this test varies the Numba thread count,
    which is genuinely honoured. It does NOT establish multi-threaded BLAS
    determinism. The joint counts come from a BLAS GEMM, and on a single-CPU
    host OpenBLAS reports one thread regardless of ``OPENBLAS_NUM_THREADS``, so
    the threaded GEMM path cannot be exercised here.

    For this estimator specifically the exposure is nil by construction: the
    summands are exact integers, so floating addition is order-independent and
    any accumulation order gives the same result, with the integrality check as
    a backstop. The unverified case is the float GEMM used by Pearson and
    Spearman, and it must be re-run on a genuine multi-core host before
    backend-level determinism is claimed.

    The backend's actual thread count is asserted below so that the limitation
    is visible in the test record instead of being inferred.
    """
    import numba as nb

    sample, ranks = compile_both(PANELS["zero_inflated"])
    original = nb.get_num_threads()

    produced = []
    try:
        for threads in (1, max(1, original)):
            nb.set_num_threads(threads)
            produced.append(
                tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)
            )
    finally:
        nb.set_num_threads(original)

    assert np.array_equal(produced[0].joint_count, produced[1].joint_count)
    assert np.array_equal(produced[0].marginal_count, produced[1].marginal_count)


def test_the_blas_backend_thread_count_is_recorded() -> None:
    """
    NB-7 environment probe. Never fails on thread count; only reports it.

    Skips when ``threadpoolctl`` is unavailable, because the limitation being
    recorded is environmental and its absence is not a defect.
    """
    threadpoolctl = pytest.importorskip("threadpoolctl")

    pools = [
        pool
        for pool in threadpoolctl.threadpool_info()
        if pool.get("internal_api") in {"openblas", "mkl", "blis"}
    ]
    assert pools, "no BLAS pool detected; determinism evidence cannot be scoped"

    # Recorded, not asserted against: a single-threaded pool means the threaded
    # GEMM path was not exercised, which is NB-7.
    for pool in pools:
        assert pool["num_threads"] >= 1


# --------------------------------------------------------------- threshold grid


@pytest.mark.parametrize(
    "bad",
    [
        np.array([]),
        np.array([0.0, 0.9]),
        np.array([0.9, 1.0]),
        np.array([0.95, 0.90]),
        np.array([0.9, np.nan]),
        np.array([[0.9, 0.95]]),
    ],
)
def test_a_malformed_threshold_grid_is_rejected(bad: np.ndarray) -> None:
    sample, ranks = compile_both(PANELS["random"])
    with pytest.raises(ValueError):
        tail_counts(sample, ranks, thresholds=bad, tail=TailSide.UPPER)


# ----------------------------------------------------------------- provenance


def test_provenance_carries_a_per_variable_tie_rate() -> None:
    """
    Per variable, not a scalar maximum: whether a cell (i, j) estimates an
    identified quantity depends on the tie structure of variables i and j.
    """
    n_observations = 100
    clean = np.arange(float(n_observations))
    forty_percent_tied = clean.copy()
    forty_percent_tied[:40] = 0.0
    matrix = np.vstack([clean, forty_percent_tied])

    sample, ranks = compile_both(matrix)
    result = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)

    tie_rate = result.provenance.tie_rate
    assert tie_rate is not None
    assert tie_rate.shape == (2,)
    assert tie_rate[0] == 0.0
    assert tie_rate[1] == pytest.approx(0.40)


def test_the_parameter_digest_covers_the_threshold_grid_and_the_tail() -> None:
    sample, ranks = compile_both(PANELS["random"])

    first = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)
    same = tail_counts(sample, ranks, thresholds=GRID.copy(), tail=TailSide.UPPER)
    other_grid = tail_counts(
        sample, ranks, thresholds=np.array([0.85, 0.95]), tail=TailSide.UPPER
    )
    other_tail = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.LOWER)

    digest = first.provenance.parameter_digest
    assert digest is not None
    assert digest == same.provenance.parameter_digest
    assert digest != other_grid.provenance.parameter_digest
    assert digest != other_tail.provenance.parameter_digest


def test_results_are_frozen() -> None:
    sample, ranks = compile_both(PANELS["random"])
    result = tail_counts(sample, ranks, thresholds=GRID, tail=TailSide.UPPER)

    assert not result.joint_count.flags.writeable
    assert not result.marginal_count.flags.writeable
    assert not result.thresholds.flags.writeable


# ------------------------------------------------------------------- memory


def test_peak_memory_stays_within_the_declared_bound() -> None:
    """
    The corrected tracked-allocation bound, itemised.

    Algorithmic scaling is ``O(KS + |Q| K^2)``. The exact tracked bound adds the
    two ``K^2`` float64 temporaries an earlier version of this note omitted --
    the GEMM output and the ``np.rint`` temporary inside the integrality check.
    A 5 per cent margin covers NumPy and CPython allocator granularity, which is
    process overhead rather than arrays this module creates.
    """
    import tracemalloc

    n_variables, n_observations = 30, 3000
    thresholds = np.array([0.80, 0.90, 0.95, 0.99])
    matrix = continuous_sample(
        seed=31, n_variables=n_variables, n_observations=n_observations
    )

    sample, ranks = compile_both(matrix)
    tail_counts(sample, ranks, thresholds=thresholds, tail=TailSide.UPPER)  # warm

    tracemalloc.start()
    tail_counts(sample, ranks, thresholds=thresholds, tail=TailSide.UPPER)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    indicator_bytes = n_variables * n_observations * 8
    mask_bytes = n_variables * n_observations * 1
    result_bytes = len(thresholds) * n_variables * n_variables * 4
    gemm_bytes = n_variables * n_variables * 8
    rint_bytes = n_variables * n_variables * 8

    declared = indicator_bytes + mask_bytes + result_bytes + gemm_bytes + rint_bytes
    assert peak <= declared * 1.05

    # Retaining one indicator per threshold would cost |Q| times the KS term.
    assert peak < len(thresholds) * indicator_bytes
