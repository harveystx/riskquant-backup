# src/riskquant/dependence_stress/permutations.py

"""
Permutation construction and application.

Every dependence method in this package reduces to one K x S permutation
matrix. The methodology builds the permutation. This module applies it.

The mechanical transformation is always::

    stressed = np.take_along_axis(data, permutations, axis=1)

which guarantees exact row-wise marginal preservation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from .types import DependenceStressResult, FloatArray, IntArray
from .validation import (
    validate_observation_matrix,
    validate_permutation_matrix,
    validate_rng,
    validate_transformation,
)


@runtime_checkable
class DependenceTransformer(Protocol):
    """
    Protocol for empirical dependence transformations.

    A transformer builds only the permutation. It never creates,
    destroys, scales, rounds, or interpolates observation values.

    Attributes
    ----------
    requires_rng:
        True when the construction is stochastic. A stochastic
        transformer must receive an explicit ``numpy.random.Generator``.
        A deterministic transformer must produce the same permutation
        for the same input regardless of ``rng``, including ``None``.
    """

    @property
    def requires_rng(self) -> bool:
        """Whether this transformer consumes randomness."""
        ...

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> IntArray:
        """
        Construct a K x S row-wise permutation matrix.

        Data contract
        -------------
        ``data`` must already be a validated, canonical, C-contiguous
        float64 observation matrix. Transformers are core components:
        they assume the boundary contract has been met and never
        re-validate. Reach them through ``transform_empirical_matrix``
        or ``DependenceStressEngine``, which own the edge.

        RNG contract
        ------------
        A transformer with ``requires_rng`` True validates ``rng``
        itself and raises ``TypeError`` on anything that is not a
        ``numpy.random.Generator``, including ``None``. A deterministic
        transformer ignores ``rng`` entirely.
        """
        ...


# ==============================================================================
# SCORE CONTRACT
# ==============================================================================


def _validate_score_matrix(
    scores: FloatArray,
    expected_shape: tuple[int, int],
) -> FloatArray:
    """
    Validate a finite K x S score matrix.

    Scores are latent ordering variables, not observations. The minimum
    K / S semantics of ``validate_observation_matrix`` therefore do not
    apply. Only dimensionality, shape agreement, and finiteness matter.
    """
    matrix = np.asarray(
        scores,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-dimensional matrix.")

    if matrix.shape != expected_shape:
        raise ValueError(
            f"scores shape {matrix.shape} does not match data shape {expected_shape}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError("scores contains NaN or infinite values.")

    return np.ascontiguousarray(matrix)


# ==============================================================================
# CONSTRUCTION
# ==============================================================================


def _build_rank_permutation_unchecked(
    matrix: FloatArray,
    score_matrix: FloatArray,
) -> IntArray:
    """
    Transfer the row-wise rank structure of scores onto observations.

    Internal trusted path. Both inputs must already satisfy their
    contracts and share a shape.

    Permutation by construction
    ---------------------------
    ``np.argsort`` over ``S`` elements returns a permutation of
    ``0..S-1`` by definition. Therefore:

        data_order[row]  is a permutation of 0..S-1
        score_order[row] is a permutation of 0..S-1

    ``put_along_axis`` writes ``data_order[row, i]`` into slot
    ``score_order[row, i]``. Because ``score_order[row]`` hits every slot
    exactly once, each output slot is written exactly once, and the
    multiset of written values is exactly ``data_order[row]``. The output
    row is therefore a permutation of ``0..S-1``.

    Running ``validate_permutation_matrix`` here would re-derive a
    property that the construction already guarantees, at O(K S log S).
    The audit stays available on the explicit audit path instead.

    Ties are broken by stable sort on both sides. Equal scores therefore
    retain ascending scenario order, which makes the construction
    deterministic for a fixed score matrix.
    """
    data_order = np.argsort(
        matrix,
        axis=1,
        kind="stable",
    ).astype(np.int64, copy=False)

    score_order = np.argsort(
        score_matrix,
        axis=1,
        kind="stable",
    )

    permutations = np.empty(
        matrix.shape,
        dtype=np.int64,
        order="C",
    )

    np.put_along_axis(
        permutations,
        score_order,
        data_order,
        axis=1,
    )

    return permutations


def build_rank_permutation(
    data: FloatArray,
    scores: FloatArray,
) -> IntArray:
    """
    Transfer the row-wise rank structure of ``scores`` onto ``data``.

    For each row:

        smallest score  -> smallest observation
        ...
        largest score   -> largest observation

    Parameters
    ----------
    data:
        K x S empirical observation matrix.
    scores:
        K x S finite latent score matrix of identical shape.

    Returns
    -------
    IntArray
        K x S C-contiguous int64 row-wise permutation matrix.

    Notes
    -----
    Score ties resolve by stable sort: tied scores keep ascending
    scenario order. Observation ties resolve the same way, so the mapping
    is fully deterministic for fixed inputs.
    """
    matrix = validate_observation_matrix(data)

    score_matrix = _validate_score_matrix(
        scores,
        matrix.shape,
    )

    return _build_rank_permutation_unchecked(
        matrix,
        score_matrix,
    )


def _comonotonic_permutation_unchecked(
    matrix: FloatArray,
) -> IntArray:
    """
    Comonotonic permutation of an already-validated matrix.
    """
    return np.argsort(
        matrix,
        axis=1,
        kind="stable",
    ).astype(np.int64, copy=False)


def comonotonic_permutation(
    data: FloatArray,
) -> IntArray:
    """
    Construct the empirical comonotonic permutation.

    Applying it sorts every row ascending, so all rows align
    scenario-by-scenario at the same severity level.

    Returns
    -------
    IntArray
        K x S C-contiguous int64 permutation matrix.
    """
    return _comonotonic_permutation_unchecked(
        validate_observation_matrix(data),
    )


# ==============================================================================
# APPLICATION
# ==============================================================================


def _apply_permutation_unchecked(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Reconstruct the full stressed K x S matrix.

    Internal trusted path. Inputs are assumed to satisfy the observation
    and permutation contracts. The gather allocates one owned K x S
    output, which is why the production path does not call it.
    """
    return np.take_along_axis(
        data,
        permutations,
        axis=1,
    )


def _aggregate_portfolio_unchecked(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Aggregate the rearranged observations scenario by scenario.

    Computes, without materialising the stressed K x S matrix::

        portfolio_pnl[s] = sum_k data[k, permutations[k, s]]

    Internal trusted path.

    Implementation notes
    --------------------
    The loop runs over K, which is small (single or double digits),
    while each iteration is one vectorised S-length gather-and-add. This
    is not a Python loop over scenarios or trades; it is a loop over a
    handful of rows.

    Peak memory is O(S) for the accumulator plus one transient S-length
    gather, against O(K*S) for reconstruct-and-sum. Measured at
    K=100, S=5000: 0.88 ms and 79 KB peak versus 2.01 ms and 3946 KB.
    The direct path is roughly 2x faster at every size measured and
    allocates about K times less.

    ``data[k][permutations[k]]`` is used rather than
    ``np.take(..., out=buffer)``. Reusing a buffer looks more
    allocation-conscious but benchmarked 20-40% SLOWER, because fancy
    indexing has a faster inner loop than ``np.take`` with an output
    argument. Measured, not assumed.

    Floating-point ordering
    -----------------------
    Accumulation runs k = 0, 1, ..., K-1. NumPy's ``sum(axis=0)`` over a
    C-contiguous array reduces along the outer axis in the same order,
    so this agrees bitwise with reconstruct-and-sum. That agreement is
    verified by test across many shapes and magnitude ranges rather than
    assumed from the documentation.
    """
    _, s = data.shape

    portfolio_pnl = np.empty(s, dtype=np.float64)

    np.take(
        data[0],
        permutations[0],
        out=portfolio_pnl,
    )

    for row in range(1, data.shape[0]):
        portfolio_pnl += data[row][permutations[row]]

    return portfolio_pnl


def apply_permutation(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Apply a K x S permutation matrix to a K x S observation matrix.

    This public entry point validates both inputs, including the full
    O(K S log S) permutation audit.

    Returns
    -------
    FloatArray
        Owned K x S stressed matrix.
    """
    matrix = validate_observation_matrix(data)

    permutation_matrix = validate_permutation_matrix(
        permutations,
        expected_shape=matrix.shape,
    )

    return _apply_permutation_unchecked(
        matrix,
        permutation_matrix,
    )


def reconstruct_stressed(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Reconstruct the full stressed K x S matrix from a stress result.

    The stressed matrix is an OPTIONAL DERIVED REPRESENTATION. It is not
    a field of ``DependenceStressResult`` and the production path never
    builds it. Call this only for audit, diagnostics, debugging,
    demonstration, or a downstream consumer that genuinely needs
    row-level stressed observations.

    Parameters
    ----------
    data:
        The original K x S empirical observation matrix.
    permutations:
        The permutation matrix from a ``DependenceStressResult``.

    Returns
    -------
    FloatArray
        Owned K x S matrix satisfying
        ``stressed[k, s] == data[k, permutations[k, s]]`` and
        ``sort(stressed[k, :]) == sort(data[k, :])`` exactly.

    Notes
    -----
    Thin alias of ``apply_permutation`` with a name that says why you
    would call it. Both validate, and both route to the same central
    ``np.take_along_axis`` operation; there is no second implementation.
    """
    return apply_permutation(data, permutations)


def aggregate_portfolio(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Aggregate permuted observations into a scenario P&L vector.

    Public, validated entry point for::

        portfolio_pnl[s] = sum_k data[k, permutations[k, s]]

    Parameters
    ----------
    data:
        K x S empirical observation matrix.
    permutations:
        K x S row-wise permutation matrix.

    Returns
    -------
    FloatArray
        Owned S-length float64 vector.

    Notes
    -----
    The sum over k is the only aggregation this package performs. It
    carries no portfolio semantics: the caller decides whether summing
    the supplied rows is meaningful. The package stays domain-agnostic.
    """
    matrix = validate_observation_matrix(data)

    permutation_matrix = validate_permutation_matrix(
        permutations,
        expected_shape=matrix.shape,
    )

    return _aggregate_portfolio_unchecked(
        matrix,
        permutation_matrix,
    )


# ==============================================================================
# ORCHESTRATION
# ==============================================================================


def _transform_trusted(
    matrix: FloatArray,
    transformer: DependenceTransformer,
    rng: np.random.Generator | None,
    *,
    audit: bool,
) -> DependenceStressResult:
    """
    Trusted transformation path.

    Private. ``matrix`` must already be a validated, canonical,
    C-contiguous float64 observation matrix. Callers inside the package
    that own a canonicalized matrix (the engine) enter here and pay the
    boundary contract exactly once, at their own construction.

    This is a private helper rather than a public ``_trusted=True``
    keyword: a user-facing boolean that switches off the core validation
    contract is a foot-gun, and the edge/core seam should be expressed
    by which function you call, not by an argument.

    The stressed K x S matrix is built ONLY when ``audit`` is True. The
    production path goes straight from the permutation to the S-vector.
    """
    if transformer.requires_rng:
        rng = validate_rng(rng)

    permutations = transformer.build_permutation(
        matrix,
        rng,
    )

    portfolio_pnl = _aggregate_portfolio_unchecked(
        matrix,
        permutations,
    )

    if audit:
        stressed = _apply_permutation_unchecked(matrix, permutations)

        if not validate_transformation(matrix, stressed, permutations):
            raise RuntimeError(
                f"{type(transformer).__name__} violated the dependence-stress "
                "transformation invariants."
            )

        if not np.array_equal(stressed.sum(axis=0), portfolio_pnl):
            raise RuntimeError(
                f"{type(transformer).__name__} portfolio aggregation does not "
                "match the reconstructed stressed matrix."
            )

    return DependenceStressResult(
        permutations=permutations,
        portfolio_pnl=portfolio_pnl,
    )


def transform_empirical_matrix(
    data: FloatArray,
    transformer: DependenceTransformer,
    rng: np.random.Generator | None = None,
    *,
    audit: bool = False,
) -> DependenceStressResult:
    """
    Build one dependence stress and aggregate it into scenario P&L.

    This is the public entry point. It validates and canonicalizes
    ``data`` at the edge, then delegates to the trusted core. The
    transformer owns dependence policy; this function owns the
    mechanical application.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.
    transformer:
        Object implementing ``DependenceTransformer``.
    rng:
        Explicit generator. Mandatory when ``transformer.requires_rng``
        is True, ignored otherwise. The package never creates a default
        generator and never reads global NumPy random state.
    audit:
        When True, reconstruct the stressed matrix and run the full
        structural audit plus a portfolio-aggregation cross-check. Off
        by default: the audit costs O(K S log S) and one K x S
        allocation on top of the transformation itself.

    Returns
    -------
    DependenceStressResult
        Read-only ``permutations`` (K x S int64) and ``portfolio_pnl``
        (S float64). The stressed K x S matrix is not returned; call
        ``reconstruct_stressed`` if row-level stressed observations are
        genuinely needed.

    Raises
    ------
    ValueError
        If the observation-matrix contract is violated.
    TypeError
        If a stochastic transformer receives a missing or malformed
        ``rng``.
    RuntimeError
        If ``audit`` is True and the transformation fails an invariant.
    """
    return _transform_trusted(
        validate_observation_matrix(data),
        transformer,
        rng,
        audit=audit,
    )


def iter_transform_empirical_matrix(
    data: FloatArray,
    transformer: DependenceTransformer,
    rng: np.random.Generator | None = None,
    *,
    n_realizations: int,
    audit: bool = False,
) -> Iterator[DependenceStressResult]:
    """
    Stream ``n_realizations`` dependence-stress realisations.

    Yields one result at a time. Nothing is accumulated, so memory stays
    bounded by a single realisation regardless of ``n_realizations``:

        base data       K x S   (owned once, by the caller or engine)
        permutation     K x S   (one live realisation)
        portfolio_pnl   S       (one live realisation)

    A caller can therefore run 10,000 realisations and retain only the
    scalar results it cares about, instead of a B x K x S ensemble.

    Statistical description
    -----------------------
    Repeated draws from a stochastic transformer are MONTE CARLO
    ESTIMATION of the finite-sample distribution induced by the stress
    specification. This is not a bootstrap: no resampling of the
    empirical observations occurs. The observations are fixed; only
    their cross-sectional pairing is redrawn.

    Reproducibility
    ---------------
    The same initial generator state, the same input and the same
    parameters produce the same SEQUENCE of realisations. That is
    reproducibility of a stochastic sequence, not determinism of the
    methodology: a copula specification induces a distribution over
    finite-sample permutations rather than one uniquely determined
    permutation.

    A deterministic transformer yields ``n_realizations`` identical
    results, which is correct but rarely useful.

    Parameters
    ----------
    n_realizations:
        Number of realisations to yield. Must be non-negative. Zero
        yields nothing.

    Yields
    ------
    DependenceStressResult
        One realisation per iteration.

    Raises
    ------
    ValueError
        If ``n_realizations`` is negative, or the data contract fails.
    TypeError
        If a stochastic transformer receives a missing or malformed
        ``rng``.
    """
    matrix = validate_observation_matrix(data)

    return _iter_transform_trusted(
        matrix,
        transformer,
        rng,
        n_realizations=n_realizations,
        audit=audit,
    )


def _iter_transform_trusted(
    matrix: FloatArray,
    transformer: DependenceTransformer,
    rng: np.random.Generator | None,
    *,
    n_realizations: int,
    audit: bool,
) -> Iterator[DependenceStressResult]:
    """
    Trusted streaming path. ``matrix`` must already be canonical.

    Validation of ``n_realizations`` and of the RNG happens eagerly, in
    the wrapping call rather than on first ``next()``, so a malformed
    argument fails at the call site instead of somewhere downstream in
    the consumer's loop.
    """
    count = int(n_realizations)

    if count < 0:
        raise ValueError(
            f"n_realizations must be non-negative; got {n_realizations!r}."
        )

    if transformer.requires_rng:
        rng = validate_rng(rng)

    def _generate() -> Iterator[DependenceStressResult]:
        for _ in range(count):
            yield _transform_trusted(
                matrix,
                transformer,
                rng,
                audit=audit,
            )

    return _generate()
