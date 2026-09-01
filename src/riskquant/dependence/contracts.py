# src/riskquant/dependence/contracts.py

"""
Result, provenance and tail-count containers.

Three fields on a result and six on provenance. Every field is justified by a
consumer that reads it; nothing is retained because it might one day be useful.

Fields deliberately absent, with reasons, so they are not reintroduced:

``family``
    No runtime consumer. ``psd_status`` and ``value_range`` carry every
    operational distinction; the taxonomy survives as documentation.

``variable_count``
    Exactly ``matrix.shape[0]``.

``effective_observation_count``
    Equal to ``observation_count`` for every estimator in this phase. It earns a
    field when an estimator actually conditions on a subsample.

``maximum_tie_rate``
    Structurally wrong: identification of a copula functional is a *pairwise*
    property, so a sample-wide scalar cannot localise which cells are affected.
    Replaced by the per-variable ``tie_rate`` vector.

``is_signed``, ``symmetry``, ``is_stochastic``
    Derivable, constant, or made an engine invariant. Exact symmetry is
    guaranteed for every estimator by the unconditional mirror, so a field
    carrying one value is not information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from riskquant.dependence.taxonomy import DependenceContract
from riskquant.observations.arrays import (
    BoolMatrix,
    Float64Matrix,
    Float64Vector,
    Int32Matrix,
)

#: Engine version stamped onto every result. The Kendall kernel is the component
#: most likely to change, and this is the only field that lets a persisted
#: matrix be reconciled after such a change.
ENGINE_VERSION: str = "riskquant.dependence/0.1.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class DependenceProvenance:
    """
    What was estimated, from how much data, under what parameters.

    Attributes
    ----------
    observation_count : int
        ``S``.

    tie_rate : Float64Vector or None
        ``(K,)`` fraction of observations in a tie block of size two or more,
        per variable. Present only when a rank representation was compiled, and
        absent for Pearson, whose estimand is not a copula functional and for
        which the field would carry no meaning.

        Per-variable rather than a scalar maximum, because whether a *cell*
        ``(i, j)`` estimates an identified quantity depends on the tie structure
        of variables ``i`` and ``j``, not on the sample's worst variable.

    parameter_digest : str or None
        Stable digest of the estimator's parameters. ``None`` for parameterless
        estimators: a digest of an empty parameter set is noise.

    engine_version : str
    """

    observation_count: int
    tie_rate: Float64Vector | None = None
    parameter_digest: str | None = None
    engine_version: str = ENGINE_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class DependenceResult:
    """
    A dependence matrix with its guarantees and its origin.

    Attributes
    ----------
    matrix : Float64Matrix
        ``(K, K)`` float64 C-contiguous, non-writeable, exactly symmetric.
        Row and column ``k`` correspond to variable ``k`` in the caller's
        declared ordering. Labels are never carried through the numerical layer.

    contract : DependenceContract
        Interned reference to the static table entry.

    provenance : DependenceProvenance
    """

    matrix: Float64Matrix
    contract: DependenceContract
    provenance: DependenceProvenance


class TailSide(Enum):
    """
    Which tail an exceedance count refers to.

    Attributes
    ----------
    UPPER
        ``u > q``.

    LOWER
        ``u < 1 - q``.

    Both inequalities are strict, which is what makes the two exact reflections
    of one another: negating a variable maps ranks ``r -> S + 1 - r`` exactly,
    hence ``u -> 1 - u``, hence the upper event on ``-X`` is the lower event on
    ``X``. A mixed ``>=`` / ``<`` pairing would break that at the boundary.
    """

    UPPER = "upper"
    LOWER = "lower"


@dataclass(frozen=True, slots=True, kw_only=True)
class TailCountResult:
    """
    Exact exceedance counts. Counts only; no coefficient is derived here.

    Every tail quantity of interest is a function of these counts, but turning
    counts into ``lambda_hat``, ``chi_hat`` or ``chi_bar_hat`` requires deciding
    what to do when a count is zero (``log 0`` diverges) and what count is
    credible. Those are policy, and kernels contain computation, not policy.
    The split also removes the zero-exceedance failure mode entirely: a count of
    zero is valid data, and only the derived coefficient is undefined, at the
    layer that has a policy for it.

    Attributes
    ----------
    thresholds : Float64Vector
        ``(Q,)`` float64 non-writeable, ascending, each in ``(0, 1)``.

    tail : TailSide

    joint_count : Int32Matrix
        ``(Q, K, K)`` int32 non-writeable. ``joint_count[q, i, j]`` is the exact
        number of observations at which variables ``i`` and ``j`` both exceed
        the ``q``-th threshold. Exactly symmetric in ``(i, j)``; the diagonal is
        the marginal count of that variable.

    marginal_count : Int32Matrix
        ``(Q, K)`` int32 non-writeable.

        Stored, not derived from ``S * (1 - q)``. Without ties the two agree
        whenever ``q * S`` is an integer, but a tie block straddling the nominal
        quantile shares one average rank and is therefore included or excluded
        as a whole, so the realised count moves in steps of the block size. The
        discrepancy is bounded only by the largest straddling block, and it
        appears precisely on the zero-inflated data where the tail estimate is
        least reliable.

    provenance : DependenceProvenance
    """

    thresholds: Float64Vector
    tail: TailSide
    joint_count: Int32Matrix
    marginal_count: Int32Matrix
    provenance: DependenceProvenance

    @property
    def variable_count(self) -> int:
        """Number of variables, ``K``."""
        return int(self.joint_count.shape[1])

    @property
    def threshold_count(self) -> int:
        """Number of thresholds, ``Q``."""
        return int(self.thresholds.shape[0])


#: Complete identifier of the frozen Phase 1b-A methodology. Every convention in
#: the coefficient layer -- directional conditioning, the ``S`` probability
#: denominator, the three domain rules, the absence of a minimum credible count,
#: and the mask-plus-NaN representation -- is fixed by the specification this
#: string names. Because none of them is a free parameter, this string is the
#: exact and complete summary of the methodology, and changing any convention
#: requires bumping it.
TAIL_COEFFICIENT_METHODOLOGY_VERSION: str = "riskquant.dependence.tail_coefficients/1bA.1.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class TailCoefficientProvenance:
    """
    Provenance of a derived tail-coefficient result.

    A separate type rather than an extension of :class:`DependenceProvenance`.
    Extending the accepted schema would alter a frozen contract and would stop
    the upstream counting digest being reproducible in isolation; chaining
    leaves both intact and makes the derivation explicit.

    Attributes
    ----------
    observation_count : int
        ``S``, copied from upstream. Required by the coefficient arithmetic
        itself, since the probability denominator is ``S``.

    upstream_digest : str or None
        The upstream ``DependenceProvenance.parameter_digest``, verbatim.

    methodology_version : str
        :data:`TAIL_COEFFICIENT_METHODOLOGY_VERSION`.

    derived_digest : str
        ``blake2b(upstream_digest_bytes + b"|" + methodology_version_bytes,
        digest_size=16).hexdigest()``, with ``b""`` standing in for a ``None``
        upstream digest, both operands UTF-8.

        Nothing else is hashed. The coefficient layer has no free parameter, so
        encoding its conventions individually would hash a tuple of constants
        that cannot vary within a methodology version. The consequence is the
        point: the derived digest changes if and only if the upstream counting
        parameters change or the methodology version changes.

    engine_version : str
    """

    observation_count: int
    upstream_digest: str | None
    methodology_version: str
    derived_digest: str
    engine_version: str = ENGINE_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class TailCoefficientResult:
    """
    Directional threshold-indexed tail statistics derived from exact counts.

    NOT SYMMETRIC. All three statistics condition on the **row** variable, so
    ``M[q, i, j] != M[q, j, i]`` is valid and expected wherever the realised
    marginal counts differ, which happens under ties. The engine-wide
    ``array_equal(M, M.T)`` invariant is **inapplicable** to this type and must
    not be asserted against it. Under continuous margins the arrays come out
    symmetric because ``m_i == m_j``, but that is a numerical consequence, not a
    contract.

    None of these is an asymptotic coefficient. Each is a finite-sample estimate
    at a declared threshold, which is why every array carries the threshold
    axis and why no field is named for a limit.

    Attributes
    ----------
    thresholds : Float64Vector
        ``(Q,)`` float64 non-writeable, copied from the counts. Axis 0 of every
        coefficient array indexes this vector in the same order.

    tail : TailSide

    marginal_count : Int32Matrix
        ``(Q, K)`` int32 non-writeable, copied from the counts.

        This is the **effective conditioning count** for the cell ``(q, i, j)``,
        read at ``[q, i]``. It is carried because no minimum credible count is
        imposed: an estimate resting on one exceedance and an estimate resting
        on three hundred are both reported, and this is how a consumer tells
        them apart. Not duplicated to ``(Q, K, K)``, which would allocate ``K``
        copies of information already present.

    lambda_hat : Float64Matrix
        ``(Q, K, K)`` float64 C-contiguous non-writeable.
        ``n_ij / m_i``, estimating ``P(U_j > q | U_i > q)``. In ``[0, 1]``
        exactly, by integer arithmetic. NaN where not estimable.

    chi_hat : Float64Matrix
        ``(Q, K, K)``. ``2 - log(p_ij) / log(p_i)``. Bounded above by ``1``
        exactly; unbounded below. Negative values are legitimate at finite ``q``
        and are reported unmodified. NaN where not estimable.

    chi_bar_hat : Float64Matrix
        ``(Q, K, K)``. ``2 * log(p_i) / log(p_ij) - 1``, and exactly ``-1.0``
        where the joint count is zero. In ``[-1, 1]`` by construction. NaN where
        not estimable.

    estimable_lambda, estimable_chi, estimable_chi_bar : BoolMatrix
        ``(Q, K, K)`` bool C-contiguous non-writeable.

        ``True`` means exactly one thing: **the estimator is mathematically
        defined at this cell**. It says nothing about statistical sufficiency,
        which is represented separately by ``marginal_count``. The two concepts
        are held apart deliberately rather than collapsed into one flag.

        Three masks rather than one, because the three domains genuinely differ:
        ``chi_hat`` excludes a zero joint count where ``chi_bar_hat`` does not,
        and both exclude ``m_i == S`` where ``lambda_hat`` does not. A shared
        mask would have to take the intersection and would suppress values that
        are defined.

    provenance : TailCoefficientProvenance
    """

    thresholds: Float64Vector
    tail: TailSide
    marginal_count: Int32Matrix
    lambda_hat: Float64Matrix
    chi_hat: Float64Matrix
    chi_bar_hat: Float64Matrix
    estimable_lambda: BoolMatrix
    estimable_chi: BoolMatrix
    estimable_chi_bar: BoolMatrix
    provenance: TailCoefficientProvenance

    @property
    def variable_count(self) -> int:
        """Number of variables, ``K``."""
        return int(self.lambda_hat.shape[1])

    @property
    def threshold_count(self) -> int:
        """Number of thresholds, ``Q``."""
        return int(self.thresholds.shape[0])
