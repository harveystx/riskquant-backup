# src/riskquant/dependence_stress/validation.py

"""
Public input/output contracts for the dependence-stress package.

This module is the single boundary between untrusted caller data and the
trusted computational core. Functions here canonicalize to float64 /
int64 C-contiguous arrays and raise loudly on any violation.

Nothing in this module repairs, clips, projects, or transposes input.
Repair is a policy decision and does not belong in a contract gate.
"""

from __future__ import annotations

import numpy as np

from .types import FloatArray, IntArray

# ==============================================================================
# PACKAGE-LOCAL NUMERICAL TOLERANCES
# ==============================================================================
# These are package-specific numerical contracts. They stay local on purpose:
# riskquant.constants carries estate-wide sentinels, not per-package linear
# algebra tolerances.

#: Absolute tolerance for the correlation symmetry check ``R == R.T``.
CORRELATION_SYMMETRY_TOL: float = 1e-10

#: Absolute tolerance for the unit-diagonal check ``diag(R) == 1``.
CORRELATION_DIAGONAL_TOL: float = 1e-10

#: Slack above 1.0 permitted on ``|R_ij|`` before rejection.
CORRELATION_BOUND_TOL: float = 1e-10

#: Strict lower bound on the minimum eigenvalue. Strict positive definiteness
#: is required because the copulas take a Cholesky factor at construction.
CORRELATION_MIN_EIGENVALUE: float = 1e-12


# ==============================================================================
# SCALAR CONTRACTS
# ==============================================================================


def validate_probability(
    probability: float,
) -> float:
    """
    Validate a probability in the open interval (0, 1).

    Returns
    -------
    float
        The canonical float value.

    Raises
    ------
    ValueError
        If the value is not finite or not strictly inside (0, 1).
    """
    value = float(probability)

    if not np.isfinite(value):
        raise ValueError(f"probability must be finite; got {probability!r}.")

    if not 0.0 < value < 1.0:
        raise ValueError(
            f"probability must lie strictly between 0 and 1; got {probability!r}."
        )

    return value


def validate_rng(
    rng: object,
) -> np.random.Generator:
    """
    Validate an explicit NumPy random generator.

    Reproducibility belongs to the caller. The package never creates a
    default generator and never touches global NumPy RNG state.

    Raises
    ------
    TypeError
        If ``rng`` is not a ``numpy.random.Generator``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an instance of numpy.random.Generator; "
            f"got {type(rng).__name__}."
        )

    return rng


def validate_disjoint_tail_counts(
    n_lower: int,
    n_upper: int,
    s: int,
) -> None:
    """
    Validate that discrete lower and upper tail blocks fit disjointly.

    The lower block occupies slots ``0 .. n_lower-1`` and the upper block
    occupies slots ``s - n_upper .. s-1``. The two blocks tile the scenario
    space without overlap if and only if ``n_lower + n_upper <= s``. Equality
    is valid: the blocks exactly fill the ``s`` slots and leave no middle
    block. A strict inequality leaves a middle block.

    This contract is distinct from the probability-level sum check on
    ``lower_probability + upper_probability``. Discrete tail sizes derive from
    ``max(1, ceil(p * s))``, so the ceiling and the ``max(1, ...)`` floor can
    push ``n_lower + n_upper`` above ``s`` even when the probabilities sum to
    strictly less than one. ``s`` is known only at transformation time, which
    is why this check lives here and not in a dataclass ``__post_init__``.

    Parameters
    ----------
    n_lower : int
        Discrete lower tail block size.

    n_upper : int
        Discrete upper tail block size.

    s : int
        Number of scenarios, the observation-axis length.

    Raises
    ------
    ValueError
        If ``n_lower + n_upper > s``. The counts are never repaired, clipped,
        or shrunk; overlap is a hard contract violation.
    """
    if n_lower + n_upper > s:
        raise ValueError(
            "discretized lower and upper tail sizes overlap; "
            f"got n_lower={n_lower}, n_upper={n_upper}, S={s}."
        )


# ==============================================================================
# MATRIX CONTRACTS
# ==============================================================================


def validate_observation_matrix(
    data: FloatArray,
) -> FloatArray:
    """
    Validate and canonicalize a K x S empirical observation matrix.

    Rows are domain-agnostic. They may hold P&Ls, returns, shocks,
    spreads, or any other empirical variable.

    Contract
    --------
    - exactly 2 dimensions
    - K >= 2
    - S >= 2
    - finite values only
    - float64 representation
    - C-contiguous storage

    Returns
    -------
    FloatArray
        C-contiguous float64 view or copy of the input.

    Raises
    ------
    ValueError
        On any contract violation. The axis convention is never inferred
        and the input is never transposed.
    """
    matrix = np.asarray(
        data,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"data must be a 2-dimensional K x S matrix; got shape {matrix.shape}."
        )

    k, s = matrix.shape

    if k < 2:
        raise ValueError(f"data must contain at least 2 rows; got K={k}.")

    if s < 2:
        raise ValueError(f"data must contain at least 2 observations; got S={s}.")

    if not np.isfinite(matrix).all():
        raise ValueError("data contains NaN or infinite values.")

    return np.ascontiguousarray(matrix)


def canonicalize_owned_matrix(
    data: FloatArray,
    *,
    read_only: bool = True,
) -> FloatArray:
    """
    Validate an observation matrix and return an owned copy.

    Use at ownership boundaries such as engine construction. A frozen
    dataclass does not freeze array contents, so the engine must hold
    memory that no caller can mutate.

    Parameters
    ----------
    read_only:
        If True, clear the write flag on the returned copy.
    """
    matrix = validate_observation_matrix(data)

    owned = np.array(
        matrix,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    if read_only:
        owned.setflags(write=False)

    return owned


def validate_correlation(
    correlation: FloatArray,
    k: int | None = None,
) -> FloatArray:
    """
    Validate a strictly positive-definite K x K correlation matrix.

    Contract
    --------
    - 2-dimensional and square
    - shape ``(k, k)`` when ``k`` is supplied
    - finite
    - symmetric within ``CORRELATION_SYMMETRY_TOL``
    - unit diagonal within ``CORRELATION_DIAGONAL_TOL``
    - entries inside ``[-1, 1]`` within ``CORRELATION_BOUND_TOL``
    - minimum eigenvalue strictly above ``CORRELATION_MIN_EIGENVALUE``

    Canonicalisation
    ----------------
    The symmetry check accepts a small asymmetry. Everything downstream
    then runs on ``0.5 * (R + R.T)``: the diagonal, bound, and
    eigenvalue checks, the returned matrix, and therefore the Cholesky
    factor the copulas build from it. Without this step the accepted
    matrix and the factorized matrix would differ, and
    ``np.linalg.eigvalsh`` / ``np.linalg.cholesky`` would silently read
    only the lower triangle of an input whose upper triangle was never
    validated.

    Symmetrizing an input that already *passed* the symmetry contract is
    canonicalisation, not repair. A matrix that fails the contract is
    rejected outright. This function never clips eigenvalues, never
    projects to the nearest correlation matrix, and never regularizes.
    Those are policy decisions owned by the caller.

    Returns
    -------
    FloatArray
        Canonical symmetric C-contiguous float64 matrix.
    """
    matrix = np.asarray(
        correlation,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"correlation must be a 2-dimensional square matrix; "
            f"got {matrix.ndim} dimensions."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"correlation must be square; got shape {matrix.shape}.")

    if k is not None and matrix.shape != (k, k):
        raise ValueError(f"expected correlation shape {(k, k)}; got {matrix.shape}.")

    if not np.isfinite(matrix).all():
        raise ValueError("correlation contains NaN or infinite values.")

    if not np.allclose(
        matrix,
        matrix.T,
        atol=CORRELATION_SYMMETRY_TOL,
        rtol=0.0,
    ):
        raise ValueError("correlation matrix must be symmetric.")

    # Canonical form. Every subsequent check, and the returned matrix,
    # operate on exactly the matrix the copulas will factorize.
    canonical = np.ascontiguousarray(
        0.5 * (matrix + matrix.T),
        dtype=np.float64,
    )

    if not np.allclose(
        np.diag(canonical),
        1.0,
        atol=CORRELATION_DIAGONAL_TOL,
        rtol=0.0,
    ):
        raise ValueError("correlation matrix diagonal must equal 1.")

    if np.any(np.abs(canonical) > 1.0 + CORRELATION_BOUND_TOL):
        raise ValueError("correlation coefficients must lie in [-1, 1].")

    eigenvalues = np.linalg.eigvalsh(canonical)
    min_eigenvalue = float(eigenvalues[0])

    if min_eigenvalue <= CORRELATION_MIN_EIGENVALUE:
        raise ValueError(
            "correlation matrix must be strictly positive definite; "
            f"minimum eigenvalue={min_eigenvalue:.6g} "
            f"<= {CORRELATION_MIN_EIGENVALUE:.6g}."
        )

    return canonical


def validate_permutation_matrix(
    permutations: IntArray,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> IntArray:
    """
    Validate a row-wise permutation matrix.

    Every row must contain every integer in ``0 .. S-1`` exactly once.

    This is an explicit audit operation with O(K S log S) cost. The
    trusted production path does not call it. Construction routines in
    ``permutations.py`` produce valid permutations by construction.

    Returns
    -------
    IntArray
        C-contiguous int64 matrix.
    """
    matrix = np.asarray(permutations)

    if matrix.ndim != 2:
        raise ValueError("permutations must be a 2-dimensional matrix.")

    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError(
            f"permutations must contain integer indices; got dtype {matrix.dtype}."
        )

    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValueError(
            f"expected permutation shape {expected_shape}; got {matrix.shape}."
        )

    k, s = matrix.shape

    if k < 1 or s < 1:
        raise ValueError("permutations must have at least one row and one scenario.")

    if np.any(matrix < 0) or np.any(matrix >= s):
        raise ValueError(f"permutation indices must lie in [0, {s - 1}].")

    expected = np.arange(
        s,
        dtype=matrix.dtype,
    )

    if not np.array_equal(
        np.sort(matrix, axis=1),
        np.broadcast_to(expected, matrix.shape),
    ):
        raise ValueError(
            "each permutation row must contain every scenario index exactly once."
        )

    return np.ascontiguousarray(
        matrix,
        dtype=np.int64,
    )


# ==============================================================================
# AUDIT CONTRACTS
# ==============================================================================


def validate_marginal_preservation(
    base: FloatArray,
    stressed: FloatArray,
) -> bool:
    """
    Verify exact row-wise empirical marginal preservation.

    The comparison is exact. A stressed observation is one of the
    original float64 values, so no tolerance is appropriate.

    This is an audit operation with O(K S log S) cost. Do not run it in
    the production path.
    """
    base_matrix = validate_observation_matrix(base)
    stressed_matrix = validate_observation_matrix(stressed)

    if stressed_matrix.shape != base_matrix.shape:
        raise ValueError(
            f"expected stressed shape {base_matrix.shape}; got {stressed_matrix.shape}."
        )

    return bool(
        np.array_equal(
            np.sort(base_matrix, axis=1),
            np.sort(stressed_matrix, axis=1),
        )
    )


def validate_transformation(
    base: FloatArray,
    stressed: FloatArray,
    permutations: IntArray,
) -> bool:
    """
    Verify a complete dependence transformation.

    Checks, in order:

    1. ``stressed`` and ``base`` share a shape;
    2. ``permutations`` is a valid row-wise permutation of that shape;
    3. row-wise marginals are preserved exactly;
    4. ``stressed == take_along_axis(base, permutations, axis=1)`` exactly.

    Returns
    -------
    bool
        True when every check passes.

    Raises
    ------
    ValueError
        When a shape or permutation contract is violated. Value-level
        failures return False rather than raising, so a caller can
        report them.

    Notes
    -----
    This is the audit/debug entry point. It costs O(K S log S) plus a
    full K x S gather. Do not run it on every production transformation.
    """
    base_matrix = validate_observation_matrix(base)
    stressed_matrix = validate_observation_matrix(stressed)

    if stressed_matrix.shape != base_matrix.shape:
        raise ValueError(
            f"expected stressed shape {base_matrix.shape}; got {stressed_matrix.shape}."
        )

    permutation_matrix = validate_permutation_matrix(
        permutations,
        expected_shape=base_matrix.shape,
    )

    if not validate_marginal_preservation(base_matrix, stressed_matrix):
        return False

    expected = np.take_along_axis(
        base_matrix,
        permutation_matrix,
        axis=1,
    )

    return bool(np.array_equal(stressed_matrix, expected))
