# src/riskquant/observations/_validate.py

"""
Validation of an empirical observation matrix. PRIVATE.

The public entry point is :func:`riskquant.observations.compile.compile_sample`.
This module holds the rejection rules only; it never repairs, imputes, drops,
standardises, or infers an orientation. Sample-altering policy lives in
:mod:`riskquant.observations.preprocess` and is applied by the caller, upstream,
before compilation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant.constants import (
    MAX_DEPENDENCE_OBSERVATIONS,
    MIN_DEPENDENCE_OBSERVATIONS,
    MIN_DEPENDENCE_VARIABLES,
)
from riskquant.observations._kernels import kernel_first_non_finite_flat_index
from riskquant.observations.arrays import Float64Matrix

_ACCEPTED_DTYPE_KINDS: frozenset[str] = frozenset({"f", "i", "u"})


def reject_unsupported_dtype(observations: NDArray[np.generic]) -> None:
    """Reject object, complex, boolean, string and datetime payloads."""
    if observations.dtype.kind not in _ACCEPTED_DTYPE_KINDS:
        raise TypeError(
            "observations must hold a real numeric dtype (floating, signed or "
            f"unsigned integer); got dtype={observations.dtype!r} "
            f"(kind={observations.dtype.kind!r})."
        )


def reject_wrong_rank(observations: NDArray[np.generic]) -> None:
    """Reject anything that is not a 2-D array."""
    if observations.ndim != 2:
        raise ValueError(
            "observations must be 2-dimensional (K variables x S observations); "
            f"got ndim={observations.ndim} with shape={observations.shape}."
        )


def reject_degenerate_shape(n_variables: int, n_observations: int) -> None:
    """Reject matrices too small for any pairwise estimator."""
    if n_variables < MIN_DEPENDENCE_VARIABLES:
        raise ValueError(
            f"observations must contain at least {MIN_DEPENDENCE_VARIABLES} "
            f"variables; got {n_variables}."
        )

    if n_observations < MIN_DEPENDENCE_OBSERVATIONS:
        raise ValueError(
            f"observations must contain at least {MIN_DEPENDENCE_OBSERVATIONS} "
            f"observations; got {n_observations}."
        )


def reject_excessive_observation_count(n_observations: int) -> None:
    """
    Reject ``S >= 2**31``.

    Two independent reasons, both hard:

    * ``sort_order`` is ``np.int32``, so a position index must fit;
    * average ranks are integers and half-integers in ``[1, S]``, which are
      exactly representable in float64 only while ``2S <= 2**53``. The int32
      bound is the tighter of the two and subsumes it.

    Exceeding the bound would corrupt the tie detection that the Kendall
    equivalence proof and the tail counting contract both rest on, so the
    failure is loud rather than silent.
    """
    if n_observations >= MAX_DEPENDENCE_OBSERVATIONS:
        raise ValueError(
            f"observations must contain fewer than {MAX_DEPENDENCE_OBSERVATIONS} "
            f"observations (sort_order is int32); got {n_observations}."
        )


def reject_non_finite(canonical: Float64Matrix) -> None:
    """
    Reject the first non-finite cell, reporting its (variable, observation) address.

    Failing with a cell address is the difference between a five-second fix and
    a bisect over a 5000-column matrix.
    """
    flat_index = int(kernel_first_non_finite_flat_index(canonical))

    if flat_index < 0:
        return

    n_observations = canonical.shape[1]
    variable_index, observation_index = divmod(flat_index, n_observations)

    raise ValueError(
        "observations must be finite; first non-finite cell at "
        f"variable_index={variable_index}, observation_index={observation_index} "
        f"(value={canonical[variable_index, observation_index]!r}). "
        "Repair or exclude the observation upstream: see "
        "riskquant.observations.preprocess."
    )


def reject_constant_variables(
    sum_of_squares: NDArray[np.float64],
    *,
    estimator_name: str,
) -> None:
    """
    Reject variables with exactly zero dispersion.

    The test is exact equality with ``0.0``, not a tolerance. A constant
    variable makes the estimator mathematically undefined (a zero denominator);
    a variable with dispersion ``1e-30`` is merely ill-conditioned, and deciding
    where "ill-conditioned" starts is caller policy, not estimator policy.

    Parameters
    ----------
    sum_of_squares : Float64Vector
        ``(K,)`` centred sums of squares of the representation the estimator
        actually consumes -- raw observations for Pearson, average ranks for the
        rank estimators. The two differ only under float64 underflow, where a
        variable can have a non-degenerate rank vector and a sum of squares that
        rounds to zero.

    estimator_name : str
        Used in the error message only.

    Raises
    ------
    ValueError
        If any variable is constant. The message lists the offending indices.
    """
    degenerate = np.flatnonzero(sum_of_squares == 0.0)

    if degenerate.size == 0:
        return

    raise ValueError(
        f"{estimator_name} is undefined for constant variables (zero "
        f"dispersion). Offending variable_index values: {degenerate.tolist()}."
    )
