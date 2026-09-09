# src/riskquant/scenario_metrics/_computer.py

from __future__ import annotations

import numpy as np

from riskquant.scenario_metrics._kernels._tail_metrics import (
    _kernel_lower_tail_contribution,
    _kernel_lower_tail_var_es,
)
from riskquant.scenario_metrics._result import ContributionResult, VarEsResult
from riskquant.scenario_metrics._types import Float64Matrix, Float64Vector


def _compute_lower_tail_parameters(
    scenario_count: int,
    confidence_level: float,
) -> tuple[int, int, float, float]:
    """
    Compile the exact empirical lower-tail geometry.

    Parameters
    ----------
    scenario_count
        Number of historical observations.

    confidence_level
        Confidence level in the open interval (0, 1).

    Returns
    -------
    var_index
        Zero-based index of the empirical lower-tail VaR order statistic.

    full_tail_count
        Number of complete empirical observations contained in the ES tail.

    fractional_weight
        Fraction of the next observation required to complete the exact
        empirical tail mass. Zero when the tail mass is integral.

    tail_mass
        Exact empirical tail mass, expressed as ``alpha * scenario_count``.

    Notes
    -----
    Let

        alpha = 1 - confidence_level
        m     = alpha * scenario_count.

    The empirical VaR is the ceil(m)-th smallest observation.

    ES integrates exactly ``m`` observations' worth of empirical probability
    mass. Therefore, when ``m`` is non-integral, the next order statistic
    receives only the fractional residual weight.
    """

    alpha = 1.0 - confidence_level
    tail_mass = alpha * float(scenario_count)

    nearest_integer = np.rint(tail_mass)

    # Guard against binary floating-point noise when the mathematically
    # expected tail mass is integral, e.g. 0.99 * 100 -> 1.0000000000000009.
    # Binary64 cannot represent every decimal confidence level exactly.
    # Snap only floating-point noise around an integer tail mass; this protects
    # canonical cases such as 0.99 * 100 without changing ordinary fractional
    # tail masses.
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(tail_mass))

    if abs(tail_mass - nearest_integer) <= tolerance:
        tail_mass = nearest_integer

    full_tail_count = int(np.floor(tail_mass))
    fractional_weight = tail_mass - float(full_tail_count)

    if fractional_weight == 0.0:
        var_index = max(0, full_tail_count - 1)
    else:
        var_index = full_tail_count

    return (
        var_index,
        full_tail_count,
        fractional_weight,
        tail_mass,
    )


def _compute_tail_var_es(
    group_value_matrix: Float64Matrix,
    var_index: int,
    full_tail_count: int,
    fractional_weight: float,
    tail_mass: float,
) -> VarEsResult:
    """
    Allocate result and workspace arrays, then execute the numerical kernel.

    All policy decisions have already been resolved by the caller.
    """

    n_groups, n_scenarios = group_value_matrix.shape

    var_out = np.empty(
        n_groups,
        dtype=np.float64,
    )

    es_out = np.empty(
        n_groups,
        dtype=np.float64,
    )

    # Selection is performed in-place in a compact working matrix. The kernel
    # does not mutate the caller-owned scenario matrix.
    scratchpad = np.empty(
        (n_groups, n_scenarios),
        dtype=np.float64,
        order="C",
    )

    _kernel_lower_tail_var_es(
        group_value_matrix=group_value_matrix,
        scratchpad=scratchpad,
        var_index=var_index,
        full_tail_count=full_tail_count,
        fractional_weight=fractional_weight,
        tail_mass=tail_mass,
        var_out=var_out,
        es_out=es_out,
    )

    var_out.setflags(write=False)
    es_out.setflags(write=False)

    return VarEsResult(
        var=var_out,
        es=es_out,
    )


def _compute_tail_contribution(
    component_matrix: Float64Matrix,
    total_row: Float64Vector,
    var_index: int,
    full_tail_count: int,
    fractional_weight: float,
    tail_mass: float,
) -> ContributionResult:
    """
    Allocate result and workspace arrays, then execute the contribution kernel.

    All policy decisions have already been resolved by the caller. The tail
    scenario set is defined by ``total_row``; each component is gathered at
    those scenario indices so the contributions foot to the headline VaR/ES.
    """

    n_components, n_scenarios = component_matrix.shape

    var_contribution_out = np.empty(
        n_components,
        dtype=np.float64,
    )

    es_contribution_out = np.empty(
        n_components,
        dtype=np.float64,
    )

    # int64 index workspace for the argselect on the total row. Overwritten by
    # the kernel; contents on entry are irrelevant.
    index_scratch = np.empty(
        n_scenarios,
        dtype=np.int64,
    )

    _kernel_lower_tail_contribution(
        component_matrix=component_matrix,
        total_row=total_row,
        index_scratch=index_scratch,
        var_index=var_index,
        full_tail_count=full_tail_count,
        fractional_weight=fractional_weight,
        tail_mass=tail_mass,
        var_contribution_out=var_contribution_out,
        es_contribution_out=es_contribution_out,
    )

    var_contribution_out.setflags(write=False)
    es_contribution_out.setflags(write=False)

    return ContributionResult(
        var_contribution=var_contribution_out,
        es_contribution=es_contribution_out,
    )
