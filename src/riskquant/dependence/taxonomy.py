# src/riskquant/dependence/taxonomy.py

"""
Mathematically accurate taxonomy of the supported estimators.

Organised by what the output *is* and what may legally be done with it, because
that is the axis on which consumers make mistakes. Technique is secondary.

The distinction that matters most is inside the association branch. Pearson and
Spearman are Gram matrices of real vectors, so positive semi-definiteness is a
theorem. Kendall tau-b is assembled pair by pair and has no such guarantee. That
decides whether a matrix may be handed to a factorisation, and it is invisible
in a taxonomy organised by "rank versus linear".

The contract table below is static. A :class:`DependenceContract` is interned
once per estimator and referenced by every result, so a result carries its
guarantees at zero per-result allocation. It is also the input to the
conformance test generator: declaring a guarantee generates the test that proves
it, so an estimator cannot claim a property it does not have.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from riskquant.constants import CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND


class DependenceEstimator(Enum):
    """
    Matrix estimators available in this phase.

    Standard ``Enum`` semantics. Equality is identity-based and members are
    hashable, so they work as dictionary keys.

    Attributes
    ----------
    PEARSON
        Pearson product-moment correlation of the raw observations.

    SPEARMAN
        Pearson correlation of average ranks.

    KENDALL_TAU_B
        Tie-corrected Kendall tau-b.
    """

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL_TAU_B = "kendall_tau_b"


class PsdStatus(Enum):
    """
    Positive semi-definiteness status of an estimator's output.

    Three states, not a boolean: a boolean would force an estimator whose output
    is not a correlation matrix at all to answer a question that does not apply,
    and ``False`` would be read as "not PSD" rather than "not that kind of
    object". This is the field a future simulation seam gates on, so it must be
    incapable of expressing a falsehood.

    Attributes
    ----------
    PSD_BY_CONSTRUCTION
        Two claims, both tested. The estimand is PSD in exact arithmetic,
        because the matrix is a Gram matrix of real vectors; and the returned
        floating-point matrix satisfies the numerical criterion
        ``lambda_min >= -tau_psd`` with ``tau_psd = K * eps * max(lambda)``.

        This does **not** imply Cholesky eligibility. A PSD matrix can be
        singular, and singularity is guaranteed whenever ``K > S``, since the
        Gram matrix then has rank at most ``S < K``. PSD, positive definite and
        Cholesky-eligible are three distinct properties, and eligibility is a
        dynamic property of a particular matrix reported by the spectrum
        diagnostic, never a static property of an estimator.

    PSD_NOT_GUARANTEED
        The output is a symmetric unit-diagonal matrix with no PSD guarantee.

    NOT_A_CORRELATION_MATRIX
        PSD is not a meaningful question for this output.
    """

    PSD_BY_CONSTRUCTION = "psd_by_construction"
    PSD_NOT_GUARANTEED = "psd_not_guaranteed"
    NOT_A_CORRELATION_MATRIX = "not_a_correlation_matrix"


class Invariance(Enum):
    """
    Declared invariance properties, each of which generates a conformance test.

    The purpose of this enum is to make an untested guarantee impossible to
    declare. Two booleans could not express the structure: Pearson is invariant
    under positive affine maps but *equivariant* under reflection, and
    collapsing that to "scale invariant" would lose the distinction that drives
    two different tests.

    Attributes
    ----------
    JOINT_PERMUTATION
        Permuting observations jointly across all variables leaves the estimate
        unchanged. Asserted exactly for pairwise integer kernels and within the
        numerical tolerance for GEMM-derived estimators.

    VARIABLE_PERMUTATION_EQUIVARIANT
        ``estimate(X[P]) == estimate(X)[P][:, P]``.

        Asserted **within the numerical tolerance**, not bitwise, for
        GEMM-derived estimators. Measured deviation is 4.2e-17 at K=13, S=977:
        BLAS blocking depends on the matrix dimensions and alignment, so
        reordering rows can change the accumulation order inside a dot product.
        Equivariance is the mathematical claim; bitwise equality is not a
        mathematical property and is not claimed here.

    POSITIVE_AFFINE_MARGINAL
        ``x -> a*x + b`` with ``a > 0`` applied to any single variable leaves
        the estimate unchanged.

    STRICTLY_INCREASING_MARGINAL
        Any strictly increasing map applied to any single variable leaves the
        estimate unchanged. Strictly stronger than
        ``POSITIVE_AFFINE_MARGINAL``, so rank estimators declare only this one.

    SIGN_EQUIVARIANT_UNDER_REFLECTION
        Negating variable ``k`` negates row and column ``k`` off the diagonal.
        Asserted **bitwise**: IEEE negation is exact and the sum of negated
        terms is the exact negation of the sum, in any accumulation order.
    """

    JOINT_PERMUTATION = "joint_permutation"
    VARIABLE_PERMUTATION_EQUIVARIANT = "variable_permutation_equivariant"
    POSITIVE_AFFINE_MARGINAL = "positive_affine_marginal"
    STRICTLY_INCREASING_MARGINAL = "strictly_increasing_marginal"
    SIGN_EQUIVARIANT_UNDER_REFLECTION = "sign_equivariant_under_reflection"


@dataclass(frozen=True, slots=True, kw_only=True)
class DependenceContract:
    """
    The static mathematical guarantees of one estimator.

    Interned in :data:`ESTIMATOR_CONTRACT` and referenced, never rebuilt, by
    every result.

    Attributes
    ----------
    estimator : DependenceEstimator
        Discriminator.

    value_range : tuple[float, float]
        Closed interval containing every value the estimator can return in exact
        arithmetic, diagonal included. ``+/-inf`` endpoints are permitted for
        unbounded estimators.

        A finite range does **not** by itself license clamping. See
        :func:`riskquant.dependence._matrix.finalise_symmetric` for the rule
        that does.

    diagonal_value : float
        The analytically known diagonal. Asserted, never estimated. Verified by
        a non-circular oracle: a duplicated variable's *off-diagonal* cell must
        equal this value, which exercises the pairwise computation rather than
        checking that a constant was copied.

    psd_status : PsdStatus

    invariances : frozenset[Invariance]
    """

    estimator: DependenceEstimator
    value_range: tuple[float, float]
    diagonal_value: float
    psd_status: PsdStatus
    invariances: frozenset[Invariance] = field(default_factory=frozenset)


_CORRELATION_RANGE: tuple[float, float] = (
    CORRELATION_LOWER_BOUND,
    CORRELATION_UPPER_BOUND,
)

_COMMON_INVARIANCES: frozenset[Invariance] = frozenset(
    {
        Invariance.JOINT_PERMUTATION,
        Invariance.VARIABLE_PERMUTATION_EQUIVARIANT,
        Invariance.SIGN_EQUIVARIANT_UNDER_REFLECTION,
    }
)

#: Static contract table. The single source of truth for every declared
#: guarantee, and the input to the conformance test generator.
ESTIMATOR_CONTRACT: Mapping[DependenceEstimator, DependenceContract] = MappingProxyType(
    {
        DependenceEstimator.PEARSON: DependenceContract(
            estimator=DependenceEstimator.PEARSON,
            value_range=_CORRELATION_RANGE,
            diagonal_value=1.0,
            psd_status=PsdStatus.PSD_BY_CONSTRUCTION,
            invariances=_COMMON_INVARIANCES | {Invariance.POSITIVE_AFFINE_MARGINAL},
        ),
        DependenceEstimator.SPEARMAN: DependenceContract(
            estimator=DependenceEstimator.SPEARMAN,
            value_range=_CORRELATION_RANGE,
            diagonal_value=1.0,
            psd_status=PsdStatus.PSD_BY_CONSTRUCTION,
            invariances=_COMMON_INVARIANCES | {Invariance.STRICTLY_INCREASING_MARGINAL},
        ),
        DependenceEstimator.KENDALL_TAU_B: DependenceContract(
            estimator=DependenceEstimator.KENDALL_TAU_B,
            value_range=_CORRELATION_RANGE,
            diagonal_value=1.0,
            # Assembled pairwise; no Gram representation, hence no PSD
            # guarantee. Consumers must not factorise this matrix.
            psd_status=PsdStatus.PSD_NOT_GUARANTEED,
            invariances=_COMMON_INVARIANCES | {Invariance.STRICTLY_INCREASING_MARGINAL},
        ),
    }
)
