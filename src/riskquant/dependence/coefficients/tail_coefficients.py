# src/riskquant/dependence/coefficients/tail_coefficients.py

"""
Deterministic tail coefficients derived from exact exceedance counts.

Implements ``PHASE_1B_A_MATHEMATICAL_SPECIFICATION.md`` exactly. Every
convention below is frozen by that document; none may be changed here.

Estimators
----------
With ``S`` the observation count, ``m_i = marginal_count[q, i]``,
``n_ij = joint_count[q, i, j]``, ``p_i = m_i / S`` and ``p_ij = n_ij / S``::

    lambda_hat[q, i, j]  = n_ij / m_i
    chi_hat[q, i, j]     = 2 - log(p_ij) / log(p_i)
    chi_bar_hat[q, i, j] = 2 * log(p_i) / log(p_ij) - 1

**The row index is the conditioning variable.** All three statistics are
directional and the results are not symmetric in general.

Frozen conventions, each with its reason
----------------------------------------
*Probability denominator is* ``S``, *never* ``S + 1``. ``S`` is the number of
trials: each of the ``S`` observations either exceeds or does not, so
``count / S`` is the empirical frequency. The ``S + 1`` in the Phase 1 threshold
arithmetic states where the event boundary sits on the pseudo-observation scale,
not how many trials there are. It also does not rescue the ``p_i == 1``
degeneracy, it disguises it: under ``S + 1`` a variable exceeding everywhere
gives ``log(S/(S+1)) ~ -1/S``, a denominator near zero returning a wild finite
number instead of an honest domain exclusion.

*The marginal is* ``m_i / S``, never ``m_j / S`` and never the nominal
``1 - q``. In the Coles-Heffernan-Tawn derivation the marginal belongs to the
conditioning variable. The nominal ``1 - q`` was rejected because it mixes a
population quantity into a sample estimator, and because under ties the realised
marginal genuinely departs from it.

*No symmetrisation.* ``max``, ``min`` and the geometric mean of the two
marginals were all considered. Two are bounds rather than estimators; the
geometric form is the Ochiai coefficient, a different statistic. Only the
directional form estimates a conditional probability.

*No clipping anywhere.* ``lambda_hat`` is in ``[0, 1]`` by exact integer
arithmetic and cannot excurse. ``chi_hat <= 1`` and ``chi_bar_hat`` in
``[-1, 1]`` hold by construction. Any value outside these is an implementation
defect and must surface. The Phase 0 numerical-excursion mechanism addresses
floating-point error from a GEMM and must not be repurposed to hide estimator
behaviour: ``chi_hat < 0`` is a legitimate finite-threshold value under negative
tail association, and is reported unmodified.

*No epsilon, pseudocount, continuity correction or minimum credible count.*
None is authorised. Credibility is made visible rather than enforced: the
effective conditioning count travels on the result so the policy can live
downstream, where the architecture already places policy.

Zero counts, resolved separately per statistic
----------------------------------------------
The same input produces three different dispositions, and that is the point:

===================  ==========================  ===========================
``n_ij == 0``        disposition                 reason
===================  ==========================  ===========================
``lambda_hat``       defined, exactly ``0.0``    a valid estimate
``chi_hat``          undefined                   ``log(0)`` in the numerator
                                                 gives ``-inf``, a limit
                                                 carrying nothing beyond the
                                                 count itself
``chi_bar_hat``      defined, exactly ``-1.0``   ``log(0)`` in the denominator
                                                 gives a finite limit inside
                                                 the attainable range
===================  ==========================  ===========================

No logarithm of zero is ever evaluated. Every branch is taken on the counts
before any transcendental is applied, so the module produces no floating-point
warning and the suite passes under ``-W error``.

Complexity
    ``O(Q K^2)`` time and memory. No ``(K, S)`` array is created or retained;
    no rank, indicator or count is recomputed. There is no BLAS call, no
    reduction across threads and no RNG, so the result is a pure deterministic
    function of the input and is unaffected by thread count.
"""

from __future__ import annotations

import hashlib
from typing import TypeVar

import numpy as np

from riskquant.dependence.contracts import (
    TAIL_COEFFICIENT_METHODOLOGY_VERSION,
    TailCoefficientProvenance,
    TailCoefficientResult,
    TailCountResult,
)
from riskquant.observations.arrays import (
    BoolMatrix,
    Float64Matrix,
    Float64Vector,
    Int32Matrix,
)

#: Value-restricted so ``_owned_copy`` returns the dtype it was given, without
#: widening either call site to ``Any``.
_ArrayT = TypeVar("_ArrayT", Float64Vector, Int32Matrix)

#: Smallest joint or marginal count at which a logarithm may be evaluated.
#: Not a credibility threshold: it is the domain boundary, forced by
#: ``log(0)`` diverging, and it is the only threshold this layer applies.
_MINIMUM_POSITIVE_COUNT: int = 1


def _derived_digest(upstream_digest: str | None, methodology_version: str) -> str:
    """
    Chain the upstream counting digest with the methodology version.

    Parameters
    ----------
    upstream_digest : str or None
        ``DependenceProvenance.parameter_digest`` from the counts. ``None`` is
        encoded as empty bytes, so a parameterless upstream and an upstream with
        an empty digest are indistinguishable by construction -- which is
        correct, since neither carries a parameter.

    methodology_version : str

    Returns
    -------
    digest : str
        32 hex characters.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"" if upstream_digest is None else upstream_digest.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(methodology_version.encode("utf-8"))
    return hasher.hexdigest()


def _owned_copy(array: _ArrayT) -> _ArrayT:
    """
    Non-writeable copy, so the result never aliases its input.

    The Phase 1 contract already guarantees C-contiguity, and ``copy()``
    preserves order, so no re-canonicalisation is needed.
    """
    copied: _ArrayT = array.copy()
    copied.setflags(write=False)
    return copied


def tail_coefficients(counts: TailCountResult) -> TailCoefficientResult:
    """
    Derive the three directional tail statistics from exact exceedance counts.

    A pure function of ``counts``. The input is never mutated and is never
    aliased by the result.

    Parameters
    ----------
    counts : TailCountResult
        Accepted Phase 1 exceedance counts.

    Returns
    -------
    result : TailCoefficientResult
        ``lambda_hat``, ``chi_hat`` and ``chi_bar_hat`` with their three
        independent estimability masks, the effective conditioning counts, and
        a provenance chained to the upstream counting digest.

    Raises
    ------
    ValueError
        If the upstream observation count is below one, which no accepted
        ``TailCountResult`` can carry, or if any count violates the Phase 1
        invariants this layer relies on.

    Notes
    -----
    Values are meaningful only where the corresponding mask is ``True``.
    Elsewhere the array holds NaN, chosen deliberately: no in-band sentinel
    exists, because ``lambda_hat`` attains every value in ``[0, 1]`` and
    ``chi_bar_hat`` every value in ``[-1, 1]``. NaN is contagious, so a consumer
    who ignores the mask gets a loud failure rather than a plausible wrong
    number.
    """
    n_observations = int(counts.provenance.observation_count)
    joint = counts.joint_count
    marginal = counts.marginal_count

    _reject_inconsistent_counts(
        joint=joint,
        marginal=marginal,
        n_observations=n_observations,
    )

    n_thresholds, n_variables, _ = joint.shape
    shape = (n_thresholds, n_variables, n_variables)

    # (Q, K, 1): broadcasting over the trailing axis makes the ROW index the
    # conditioning variable, which is the whole content of the directional
    # decision. Transposing this to (Q, 1, K) would silently condition on the
    # column instead.
    conditioning_count = marginal[:, :, None]

    joint_positive = joint >= _MINIMUM_POSITIVE_COUNT

    estimable_lambda: BoolMatrix = np.empty(shape, dtype=np.bool_)
    np.copyto(estimable_lambda, conditioning_count >= _MINIMUM_POSITIVE_COUNT)

    estimable_chi_bar: BoolMatrix = np.empty(shape, dtype=np.bool_)
    np.copyto(
        estimable_chi_bar,
        (conditioning_count >= _MINIMUM_POSITIVE_COUNT)
        & (conditioning_count <= n_observations - 1),
    )

    estimable_chi: BoolMatrix = estimable_chi_bar & joint_positive

    lambda_hat = _lambda_hat(
        joint=joint,
        conditioning_count=conditioning_count,
        estimable=estimable_lambda,
        shape=shape,
    )

    # log(p_i) depends on i only, so it is held at (Q, K, 1) rather than
    # (Q, K, K): one fewer full-size temporary, and it broadcasts identically.
    log_marginal_probability = _log_probability(
        count=conditioning_count,
        n_observations=n_observations,
        estimable=estimable_chi_bar,
    )
    log_joint_probability = _log_probability(
        count=joint,
        n_observations=n_observations,
        estimable=estimable_chi,
    )

    chi_hat = _chi_hat(
        log_joint_probability=log_joint_probability,
        log_marginal_probability=log_marginal_probability,
        estimable=estimable_chi,
        shape=shape,
    )
    chi_bar_hat = _chi_bar_hat(
        log_joint_probability=log_joint_probability,
        log_marginal_probability=log_marginal_probability,
        estimable=estimable_chi_bar,
        joint_positive=joint_positive,
        shape=shape,
    )

    for array in (lambda_hat, chi_hat, chi_bar_hat):
        array.setflags(write=False)
    for mask in (estimable_lambda, estimable_chi, estimable_chi_bar):
        mask.setflags(write=False)

    provenance = TailCoefficientProvenance(
        observation_count=n_observations,
        upstream_digest=counts.provenance.parameter_digest,
        methodology_version=TAIL_COEFFICIENT_METHODOLOGY_VERSION,
        derived_digest=_derived_digest(
            counts.provenance.parameter_digest,
            TAIL_COEFFICIENT_METHODOLOGY_VERSION,
        ),
    )

    return TailCoefficientResult(
        thresholds=_owned_copy(counts.thresholds),
        tail=counts.tail,
        marginal_count=_owned_copy(marginal),
        lambda_hat=lambda_hat,
        chi_hat=chi_hat,
        chi_bar_hat=chi_bar_hat,
        estimable_lambda=estimable_lambda,
        estimable_chi=estimable_chi,
        estimable_chi_bar=estimable_chi_bar,
        provenance=provenance,
    )


def _reject_inconsistent_counts(
    *,
    joint: Int32Matrix,
    marginal: Int32Matrix,
    n_observations: int,
) -> None:
    """
    Fail loud on counts that violate the Phase 1 invariants this layer assumes.

    The bounds proved in the specification -- ``lambda_hat`` in ``[0, 1]``,
    ``chi_hat <= 1``, ``chi_bar_hat`` in ``[-1, 1]`` -- all rest on
    ``n_ij <= min(m_i, m_j) <= S``. Checking the premise here means a violation
    surfaces as a stated contract breach rather than as a silently
    out-of-range coefficient.
    """
    if n_observations < 1:
        raise ValueError(
            f"observation_count must be >= 1; got {n_observations}."
        )
    if int(marginal.min(initial=0)) < 0 or int(marginal.max(initial=0)) > n_observations:
        raise ValueError(
            "marginal_count must lie in [0, observation_count]; got range "
            f"[{int(marginal.min())}, {int(marginal.max())}] with "
            f"observation_count={n_observations}."
        )
    if int(joint.min(initial=0)) < 0:
        raise ValueError("joint_count must be non-negative.")
    if bool((joint > marginal[:, :, None]).any()):
        raise ValueError(
            "joint_count[q, i, j] must not exceed marginal_count[q, i]; the "
            "joint exceedance set is a subset of each marginal exceedance set."
        )


def _lambda_hat(
    *,
    joint: Int32Matrix,
    conditioning_count: Int32Matrix,
    estimable: BoolMatrix,
    shape: tuple[int, int, int],
) -> Float64Matrix:
    """
    ``n_ij / m_i``, NaN where ``m_i == 0``.

    ``where=`` rather than a guarded divide: the division is never evaluated at
    an undefined cell, so no divide-by-zero warning can be raised and the suite
    holds under ``-W error``.
    """
    result: Float64Matrix = np.full(shape, np.nan, dtype=np.float64)
    np.divide(joint, conditioning_count, out=result, where=estimable)
    return result


def _log_probability(
    *,
    count: Int32Matrix,
    n_observations: int,
    estimable: BoolMatrix,
) -> Float64Matrix:
    """
    ``log(count / S)`` where estimable, NaN elsewhere.

    The probability denominator is ``S``. Both operations are gated by
    ``where=``, so ``log(0)`` and ``log`` of a NaN are never evaluated.

    The output takes the broadcast shape of ``count``, which is ``(Q, K, 1)``
    for the marginal and ``(Q, K, K)`` for the joint. Keeping the marginal at
    ``(Q, K, 1)`` avoids a full-size temporary for a quantity that does not
    depend on the column index.
    """
    result: Float64Matrix = np.full(count.shape, np.nan, dtype=np.float64)
    gate = np.broadcast_to(estimable, np.broadcast_shapes(estimable.shape, count.shape))
    if gate.shape != count.shape:
        # The marginal gate is (Q, K, K) while the marginal count is (Q, K, 1);
        # a cell is loggable when its own count is interior, which does not
        # depend on j, so any column of the gate serves.
        gate = gate[:, :, :1]
    np.divide(count, n_observations, out=result, where=gate)
    np.log(result, out=result, where=gate)
    return result


def _chi_hat(
    *,
    log_joint_probability: Float64Matrix,
    log_marginal_probability: Float64Matrix,
    estimable: BoolMatrix,
    shape: tuple[int, int, int],
) -> Float64Matrix:
    """``2 - log(p_ij) / log(p_i)``, NaN outside the domain."""
    result: Float64Matrix = np.full(shape, np.nan, dtype=np.float64)
    np.divide(
        log_joint_probability,
        log_marginal_probability,
        out=result,
        where=estimable,
    )
    np.subtract(2.0, result, out=result, where=estimable)
    return result


def _chi_bar_hat(
    *,
    log_joint_probability: Float64Matrix,
    log_marginal_probability: Float64Matrix,
    estimable: BoolMatrix,
    joint_positive: BoolMatrix,
    shape: tuple[int, int, int],
) -> Float64Matrix:
    """
    ``2 * log(p_i) / log(p_ij) - 1``, and exactly ``-1.0`` at a zero joint count.

    The zero-joint branch is taken on the count, before any logarithm. Computing
    the formula and letting IEEE produce ``-1.0`` from ``finite / -inf`` would
    give the same number while raising a divide-by-zero warning, which fails
    under ``-W error``. It would also make the value an accident of floating
    point rather than a stated rule.
    """
    result: Float64Matrix = np.full(shape, np.nan, dtype=np.float64)

    logarithmic = estimable & joint_positive
    np.divide(
        log_marginal_probability,
        log_joint_probability,
        out=result,
        where=logarithmic,
    )
    np.multiply(result, 2.0, out=result, where=logarithmic)
    np.subtract(result, 1.0, out=result, where=logarithmic)

    np.copyto(result, -1.0, where=estimable & ~joint_positive)
    return result
