# src/riskquant/observations/compile.py

"""
Compilation of an empirical observation matrix into immutable execution artifacts.

Two artifacts, deliberately separate:

* :class:`CompiledSample` -- the canonical observations and the per-variable
  centred sum of squares. Everything an estimator on raw observations needs.
* :class:`RankRepresentation` -- the average ranks, the stable ascending sort
  order, and the tied-pair counts. Built from a compiled sample, and shared by
  every rank estimator.

They are separate rather than one object with optional fields so that the cost
is visible at the call site: a Pearson-only caller allocates 4 MB at K=100,
S=5000 and never touches the rank pipeline; a caller wanting Spearman, Kendall
and the tail block pays for one sort, once, and shares it across all three.

Ownership
---------
:func:`compile_sample` **always** copies, including when the caller's array is
already canonical. The copy costs one ``(K, S)`` allocation, once. It buys
snapshot semantics: two estimates from one compiled sample are guaranteed to
describe the same data even if the caller mutates their own array in between.
For an artifact whose purpose is to be retained across many estimator calls,
aliasing a live caller buffer is indefensible. Every retained array is marked
non-writeable.

Neither artifact carries dependence semantics. ``sort_order`` in particular is a
property of the sample under an ordering, which is why
:mod:`riskquant.dependence_stress` can consume it without importing anything
from :mod:`riskquant.dependence`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from riskquant.observations import _validate
from riskquant.observations._kernels import (
    kernel_sum_of_squares,
    kernel_tied_pair_counts,
)
from riskquant.observations.arrays import Float64Matrix, Float64Vector, Int32Matrix
from riskquant.observations.orientation import ObservationOrientation


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSample:
    """
    An immutable, validated empirical observation matrix.

    Attributes
    ----------
    observations : Float64Matrix
        ``(K, S)`` float64 C-contiguous, finite, non-writeable. Variables on
        rows: every pairwise estimator consumes one complete variable at a time,
        so row-major variables makes each variable a unit-stride slice.

    sum_of_squares : Float64Vector
        ``(K,)`` float64 non-writeable, ``sum((x - mean(x))^2)`` per variable.
        The degeneracy gate for estimators on raw observations. Exactly ``0.0``
        for a constant variable.
    """

    observations: Float64Matrix
    sum_of_squares: Float64Vector

    @property
    def variable_count(self) -> int:
        """Number of variables, ``K``."""
        return int(self.observations.shape[0])

    @property
    def observation_count(self) -> int:
        """Number of observations, ``S``."""
        return int(self.observations.shape[1])


@dataclass(frozen=True, slots=True, kw_only=True)
class RankRepresentation:
    """
    Rank-derived artifacts shared by every rank estimator.

    Attributes
    ----------
    ranks : Float64Matrix
        ``(K, S)`` float64 C-contiguous non-writeable. Average ranks in
        ``[1, S]``, taken along the observation axis. Ties receive the mean of
        the ranks they span, which is what keeps Spearman equal to the Pearson
        correlation of the ranks and keeps Kendall's tie correction consistent
        with the rank representation.

    sort_order : Int32Matrix
        ``(K, S)`` int32 C-contiguous non-writeable. Stable ascending argsort of
        each variable. int32 halves the footprint of NumPy's native int64 result
        and matches the kernel signatures.

    tie_pairs : Float64Vector
        ``(K,)`` float64 non-writeable. ``sum over tie blocks of t(t-1)/2``.
    """

    ranks: Float64Matrix
    sort_order: Int32Matrix
    tie_pairs: Float64Vector

    @property
    def variable_count(self) -> int:
        """Number of variables, ``K``."""
        return int(self.ranks.shape[0])

    @property
    def observation_count(self) -> int:
        """Number of observations, ``S``."""
        return int(self.ranks.shape[1])


def _freeze(array: NDArray[np.generic]) -> None:
    """Mark an owned array non-writeable."""
    array.setflags(write=False)


def compile_sample(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
) -> CompiledSample:
    """
    Validate, canonicalise and take ownership of an observation matrix.

    Parameters
    ----------
    observations : NDArray
        Real numeric 2-D array of already-cleaned empirical observations. Any
        sample repair belongs upstream, in
        :mod:`riskquant.observations.preprocess`.

    orientation : ObservationOrientation
        Declares which axis carries the variables. Never inferred, never
        silently transposed.

    Returns
    -------
    sample : CompiledSample
        Owns its storage. Mutating ``observations`` afterwards has no effect on
        the compiled sample or on any estimate derived from it.

    Raises
    ------
    TypeError
        Unsupported dtype kind, or an orientation that is not an enum member.

    ValueError
        Wrong rank, fewer than two variables or observations, ``S >= 2**31``,
        or any non-finite cell (reported with its cell address).
    """
    if not isinstance(orientation, ObservationOrientation):
        raise TypeError(
            "orientation must be an ObservationOrientation member; got "
            f"{type(orientation).__name__}."
        )

    _validate.reject_wrong_rank(observations)
    _validate.reject_unsupported_dtype(observations)

    oriented = (
        observations
        if orientation is ObservationOrientation.VARIABLES_ON_ROWS
        else observations.T
    )

    n_variables, n_observations = oriented.shape
    _validate.reject_degenerate_shape(n_variables, n_observations)
    _validate.reject_excessive_observation_count(n_observations)

    # Unconditional copy: snapshot ownership, not a canonicalisation shortcut.
    canonical: Float64Matrix = np.array(
        oriented, dtype=np.float64, order="C", copy=True
    )

    _validate.reject_non_finite(canonical)

    sum_of_squares = np.empty(n_variables, dtype=np.float64)
    kernel_sum_of_squares(canonical, sum_of_squares)

    _freeze(canonical)
    _freeze(sum_of_squares)

    return CompiledSample(observations=canonical, sum_of_squares=sum_of_squares)


def compile_ranks(sample: CompiledSample) -> RankRepresentation:
    """
    Build the rank artifacts shared by every rank estimator.

    Parameters
    ----------
    sample : CompiledSample

    Returns
    -------
    ranks : RankRepresentation
        Owns its storage; every array is non-writeable.

    Notes
    -----
    One sort supports all three consumers. ``scipy.stats.rankdata`` with
    ``axis=1`` already returns a float64 C-contiguous array, so no
    re-canonicalisation is performed; that layout is asserted in the test suite
    rather than defended with a copy.

    ``np.argsort(..., kind="stable")`` is used rather than a sort inside a
    kernel: NumPy's is the well-tuned path, and doing it per variable inside a
    ``prange`` would allocate once per variable.
    """
    observations = sample.observations

    ranks: Float64Matrix = rankdata(observations, method="average", axis=1)

    sort_order: Int32Matrix = np.ascontiguousarray(
        np.argsort(observations, axis=1, kind="stable"),
        dtype=np.int32,
    )

    tie_pairs = np.empty(sample.variable_count, dtype=np.float64)
    kernel_tied_pair_counts(observations, sort_order, tie_pairs)

    _freeze(ranks)
    _freeze(sort_order)
    _freeze(tie_pairs)

    return RankRepresentation(
        ranks=ranks,
        sort_order=sort_order,
        tie_pairs=tie_pairs,
    )
