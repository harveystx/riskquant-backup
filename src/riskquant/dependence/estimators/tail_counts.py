# src/riskquant/dependence/estimators/tail_counts.py

"""
Exact tail exceedance counts. Counts only.

What this module deliberately does not do
------------------------------------------
It does not compute ``lambda_hat``, ``chi_hat`` or ``chi_bar_hat``. Every tail
quantity of interest is a function of the counts produced here, but turning
counts into a coefficient requires deciding what to do when a count is zero
(``log 0`` diverges) and what count is credible enough to report. Those are
policy decisions, and kernels contain computation, not policy.

The split also removes the zero-exceedance failure mode entirely: a count of
zero is valid data, and only the derived coefficient is undefined -- at the
layer that has a policy for it. A consumer may compute ``joint / marginal``
outside this module at its own risk; that division is not part of this contract.

Event definition
----------------
Let ``r`` be average ranks in ``[1, S]`` and ``u = r / (S + 1)`` the conceptual
pseudo-observations. At level ``q`` in ``(0, 1)``::

    UPPER(q) = { s : u[s] >  q     }
    LOWER(q) = { s : u[s] <  1 - q }

Both inequalities are **strict**, and that pairing is forced rather than chosen.
Negating a variable maps ranks ``r -> S + 1 - r`` exactly, including within tie
blocks, hence ``u -> 1 - u``. The upper event on ``-X`` is then exactly the
lower event on ``X``, giving the invariant::

    joint_upper(X, Y; q) == joint_lower(-X, -Y; q)

as integer arrays. A mixed ``>=`` / ``<`` pairing breaks that at the boundary.

``u`` is never materialised, which is why pseudo-observations are not a stored
artifact. Writing ``T = S + 1``, the two events are applied as::

    UPPER(q):        r  > q * T
    LOWER(q):  (T - r)  > q * T

The lower form is algebraically ``r < (1 - q) * T`` and is deliberately not
computed that way. ``(1 - q) * T`` and ``T - q * T`` are not bit-identical: at
``q = 0.8`` and ``S = 9`` the first evaluates to ``1.9999999999999996`` and the
second to ``2.0``, and over a sweep of S and conventional levels they disagree
in the last bits for most pairs.

No realised rank has been found strictly between the two cuts -- ranks are
spaced by at least 0.5 while the discrepancy is around 1e-16 -- so the naive
form is not known to produce a wrong count. The form above is used anyway,
because it makes the reflection invariant rest on an exactness argument rather
than on that spacing margin: both tails compare against the *same* float
``q * T``, and ``T - r`` is exact since ranks are integers or half-integers
bounded by ``T`` with ``2T <= 2**53``. The invariant then holds by construction
for every ``q``, and does not have to be re-derived if the rank representation
ever changes.

Marginal counts are stored, not derived
---------------------------------------
Without ties, ``marginal_count = S - floor(q(S+1))``, which equals ``S(1-q)``
whenever ``q * S`` is an integer -- true of the common configurations. With
ties, a block straddling the nominal quantile shares one average rank and is
therefore included or excluded as a whole, so the realised count moves in steps
of the block size. The discrepancy is bounded only by the largest straddling
block, and it appears precisely on the zero-inflated data where the tail
estimate is least reliable. Deriving the count would be silently wrong exactly
where accuracy matters most.

Kernel
------
The joint-count matrix at one threshold is ``M @ M.T`` for the ``(K, S)`` 0/1
indicator ``M``: a single GEMM, BLAS-optimal, with no bit manipulation.

The indicator is built in **float64**, not float32. The float32 argument -- that
every partial sum of an inner product of 0/1 vectors is an integer bounded by
``S``, hence exactly representable while ``S <= 2**24`` under any summation
order -- is mathematically sound but assumes IEEE binary32 arithmetic
throughout, and that is not portable: a BLAS may dispatch ``sgemm`` to a
reduced-precision path such as TF32 or bfloat16, where counts above a few
thousand would be silently wrong. Correctness must not depend on an uncontrolled
dispatch path. float64 is exact to ``2**53`` regardless, and an integrality
check before the cast catches any future surprise loudly.

Because the summands are exact integers, floating addition is order-independent,
so the counts are deterministic under any thread count.

Complexity
    ``O(|Q| K^2 S)`` GEMM flops.

Memory
    Three statements, deliberately distinguished.

    *Algorithmic scaling*: ``O(KS + |Q| K^2)`` beyond the shared artifacts. The
    indicator is built once and reused for every threshold, so the ``KS`` term
    does not scale with ``|Q|``.

    *Tracked allocation bound*: the exact arithmetic is

    .. code-block:: text

        K*S*8   indicator, float64, reused across the grid
      + K*S*1   boolean mask scratch, reused across the grid
      + Q*K*K*4 joint_count, int32, the mandatory result
      + K*K*8   the float64 GEMM output for one threshold
      + K*K*8   the np.rint temporary inside the integrality check

    The last two terms are ``K^2`` per threshold and transient. An earlier
    version of this note omitted them, which understated the bound by
    ``2 K^2 * 8`` bytes -- 0.16 MB at K = 100. Measured peaks at K = 100,
    S = 5000 are 4.876 / 5.078 / 5.482 MB for Q = 5 / 10 / 20 on both tails,
    against 4.860 / 5.060 / 5.460 MB from the corrected formula.

    *Process allocator overhead*: the residual few kilobytes above the corrected
    formula are NumPy and CPython allocator granularity, not arrays this module
    creates.

    Growth from Q = 5 to Q = 20 was measured at 0.606 MB against 0.600 MB for
    the 15 additional ``K^2`` int32 slices; retaining one indicator per
    threshold would have cost 60.0 MB. There is no architectural memory
    violation.
"""

from __future__ import annotations

import hashlib

import numpy as np

from riskquant.dependence._kernels._symmetry import (
    kernel_mirror_upper_and_set_diagonal,
)
from riskquant.dependence.contracts import (
    DependenceProvenance,
    TailCountResult,
    TailSide,
)
from riskquant.observations._kernels import kernel_sum_of_squares, kernel_tie_rate
from riskquant.observations._validate import reject_constant_variables
from riskquant.observations.arrays import BoolMatrix, Float64Matrix, Float64Vector
from riskquant.observations.compile import CompiledSample, RankRepresentation


def _validate_thresholds(thresholds: Float64Vector) -> Float64Vector:
    """Reject a malformed threshold grid; return it canonicalised."""
    grid: Float64Vector = np.ascontiguousarray(thresholds, dtype=np.float64)

    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(
            f"thresholds must be a non-empty 1-D array; got shape={grid.shape}."
        )
    if not np.isfinite(grid).all():
        raise ValueError("thresholds must be finite.")
    if grid.min() <= 0.0 or grid.max() >= 1.0:
        raise ValueError(
            "thresholds must lie strictly inside (0, 1); got range "
            f"[{grid.min()!r}, {grid.max()!r}]."
        )
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("thresholds must be strictly ascending.")

    return grid


def _parameter_digest(thresholds: Float64Vector, tail: TailSide) -> str:
    """
    Stable digest of the threshold grid and the tail side.

    Built from the exact float64 bit patterns, so two grids that print the same
    but differ in the last bit produce different digests.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(tail.value.encode("ascii"))
    hasher.update(thresholds.tobytes())
    return hasher.hexdigest()


def _exceedance_indicator(
    ranks: Float64Matrix,
    *,
    threshold: float,
    tail: TailSide,
    n_observations: int,
    indicator_out: Float64Matrix,
    mask_out: BoolMatrix,
) -> None:
    """
    Fill ``indicator_out`` with the ``(K, S)`` float64 0/1 exceedance indicator.

    float64 for the reason given in the module docstring: the exactness
    guarantee must not depend on the BLAS dispatch path.

    Both buffers are supplied by the caller and reused across the whole
    threshold grid. Returning a fresh array instead would keep the previous
    iteration's ``(K, S)`` alive while the next one is built -- name rebinding
    happens only after the right-hand side has been evaluated -- so the measured
    peak was two indicators, not one. Reuse restores the declared
    ``O(KS + |Q| K^2)`` bound.

    The lower tail needs ``T - r`` before the comparison. It is written into
    ``indicator_out`` itself, which is overwritten by the result two lines
    later, so no third ``(K, S)`` buffer exists.

    Parameters
    ----------
    ranks : Float64Matrix
        ``(K, S)`` average ranks. Never mutated.

    threshold : float
        Level ``q`` in ``(0, 1)``.

    tail : TailSide

    n_observations : int
        ``S``.

    indicator_out : Float64Matrix
        ``(K, S)`` float64 destination. Also used as scratch by the lower tail.

    mask_out : BoolMatrix
        ``(K, S)`` boolean scratch.
    """
    scale = float(n_observations + 1)
    cut = threshold * scale

    if tail is TailSide.UPPER:
        np.greater(ranks, cut, out=mask_out)
    elif tail is TailSide.LOWER:
        # (T - r) > q * T, algebraically r < (1 - q) * T but compared against
        # the same float cut as the upper tail, which makes the reflection
        # invariant exact for every q. See the module docstring.
        np.subtract(scale, ranks, out=indicator_out)
        np.greater(indicator_out, cut, out=mask_out)
    else:
        raise ValueError(f"Unsupported TailSide: {tail!r}.")

    np.copyto(indicator_out, mask_out)


def tail_counts(
    sample: CompiledSample,
    ranks: RankRepresentation,
    *,
    thresholds: Float64Vector,
    tail: TailSide,
) -> TailCountResult:
    """
    Exact joint and marginal exceedance counts over a threshold grid.

    Parameters
    ----------
    sample : CompiledSample
        Supplies the degeneracy gate and provenance.

    ranks : RankRepresentation
        Supplies the average ranks and the shared sort order.

    thresholds : Float64Vector
        ``(Q,)`` strictly ascending levels in ``(0, 1)``.

    tail : TailSide
        ``UPPER`` or ``LOWER``. One tail per call: the lower joint counts are
        not recoverable from the upper ones, so the two require separate
        indicators and separate GEMMs.

    Returns
    -------
    result : TailCountResult
        Counts only. No coefficient is derived.

    Raises
    ------
    ValueError
        If any variable is constant, if the threshold grid is malformed, or if
        the GEMM returns a non-integral value.
    """
    grid = _validate_thresholds(thresholds)

    rank_sum_of_squares = np.empty(ranks.variable_count, dtype=np.float64)
    kernel_sum_of_squares(ranks.ranks, rank_sum_of_squares)
    reject_constant_variables(
        rank_sum_of_squares,
        estimator_name="Tail exceedance counting",
    )

    n_variables = ranks.variable_count
    n_observations = ranks.observation_count
    n_thresholds = int(grid.size)

    joint_count = np.empty((n_thresholds, n_variables, n_variables), dtype=np.int32)
    marginal_count = np.empty((n_thresholds, n_variables), dtype=np.int32)

    # One indicator and one boolean scratch, reused for every threshold. Peak
    # beyond the shared artifacts stays O(KS + |Q| K^2) and is flat in Q.
    indicator = np.empty((n_variables, n_observations), dtype=np.float64)
    mask_scratch = np.empty((n_variables, n_observations), dtype=np.bool_)

    for position in range(n_thresholds):
        _exceedance_indicator(
            ranks.ranks,
            threshold=float(grid[position]),
            tail=tail,
            n_observations=n_observations,
            indicator_out=indicator,
            mask_out=mask_scratch,
        )

        marginal = indicator.sum(axis=1)
        counts: Float64Matrix = indicator @ indicator.T

        if not np.array_equal(counts, np.rint(counts)):
            raise ValueError(
                "the exceedance GEMM returned a non-integral value at threshold "
                f"{grid[position]!r}. Counts are sums of exact integers and must "
                "be exact; a non-integral result indicates a reduced-precision "
                "BLAS dispatch path and the counts cannot be trusted."
            )

        # The GEMM populates both triangles, and exact-integer summation makes
        # them equal in any order. The mirror is applied anyway: exactly one
        # value per unordered pair is retained, uniformly across the engine.
        kernel_mirror_upper_and_set_diagonal(counts, 0.0)
        np.fill_diagonal(counts, marginal)

        joint_count[position] = counts.astype(np.int32)
        marginal_count[position] = marginal.astype(np.int32)

    tie_rate = np.empty(n_variables, dtype=np.float64)
    kernel_tie_rate(sample.observations, ranks.sort_order, tie_rate)

    grid.setflags(write=False)
    joint_count.setflags(write=False)
    marginal_count.setflags(write=False)
    tie_rate.setflags(write=False)

    return TailCountResult(
        thresholds=grid,
        tail=tail,
        joint_count=joint_count,
        marginal_count=marginal_count,
        provenance=DependenceProvenance(
            observation_count=sample.observation_count,
            tie_rate=tie_rate,
            parameter_digest=_parameter_digest(grid, tail),
        ),
    )
