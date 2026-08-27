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
    Apply an already-validated permutation.

    Internal trusted path. Inputs are assumed to satisfy the observation
    and permutation contracts. The gather allocates exactly one owned
    K x S output.
    """
    return np.take_along_axis(
        data,
        permutations,
        axis=1,
    )


def apply_permutation(
    data: FloatArray,
    permutations: IntArray,
) -> FloatArray:
    """
    Apply a K x S permutation matrix to a K x S observation matrix.

    This public entry point validates both inputs, including the full
    O(K S log S) permutation audit. Use ``transform_empirical_matrix``
    for the production path.

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
    """
    if transformer.requires_rng:
        rng = validate_rng(rng)

    permutations = transformer.build_permutation(
        matrix,
        rng,
    )

    stressed = _apply_permutation_unchecked(
        matrix,
        permutations,
    )

    if audit and not validate_transformation(
        matrix,
        stressed,
        permutations,
    ):
        raise RuntimeError(
            f"{type(transformer).__name__} violated the dependence-stress "
            "transformation invariants."
        )

    return DependenceStressResult(
        stressed=stressed,
        permutations=permutations,
    )


def transform_empirical_matrix(
    data: FloatArray,
    transformer: DependenceTransformer,
    rng: np.random.Generator | None = None,
    *,
    audit: bool = False,
) -> DependenceStressResult:
    """
    Build and apply one empirical dependence transformation.

    This is the public entry point. It validates and canonicalizes
    ``data`` at the edge, then delegates to the trusted core. The
    transformer owns dependence policy; this function owns the
    mechanical application of the resulting permutation.

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
        When True, run the full ``validate_transformation`` audit on the
        result. Off by default because the audit costs O(K S log S) on
        top of the transformation itself.

    Returns
    -------
    DependenceStressResult
        Read-only stressed matrix and the exact permutation that
        produced it.

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
