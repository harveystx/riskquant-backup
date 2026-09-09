# src/riskquant/scenario_metrics/api/var_es.py

from __future__ import annotations

from riskquant.scenario_metrics._computer import (
    _compute_lower_tail_parameters,
    _compute_tail_var_es,
)
from riskquant.scenario_metrics._result import VarEsResult
from riskquant.scenario_metrics._types import Float64Matrix
from riskquant.scenario_metrics._validate import (
    _validate_confidence_level,
    _validate_group_value_matrix,
)


def compute_var_es(
    group_value_matrix: Float64Matrix,
    confidence_level: float,
) -> VarEsResult:
    """
    Compute empirical lower-tail historical VaR and ES for each group.

    Parameters
    ----------
    group_value_matrix
        C-contiguous float64 matrix with shape
        ``(n_groups, n_scenarios)``.

        Rows are independent empirical risk variables/groups.
        Columns are historical scenarios.

    confidence_level
        Confidence level in ``(0, 1)``.

    Returns
    -------
    VarEsResult
        Empirical lower-tail VaR and ES for each group.

    Mathematical convention
    ------------------------
    Let ``alpha = 1 - confidence_level`` and ``m = alpha * S``.

    VaR is the empirical ``ceil(m)``-th lower-tail order statistic.

    ES integrates exactly ``m`` observations' worth of empirical probability
    mass, using a fractional weight on the boundary observation when ``m`` is
    non-integral.
    """

    _validate_group_value_matrix(
        group_value_matrix,
    )

    canonical_confidence_level = _validate_confidence_level(
        confidence_level,
    )

    (
        var_index,
        full_tail_count,
        fractional_weight,
        tail_mass,
    ) = _compute_lower_tail_parameters(
        scenario_count=group_value_matrix.shape[1],
        confidence_level=canonical_confidence_level,
    )

    return _compute_tail_var_es(
        group_value_matrix=group_value_matrix,
        var_index=var_index,
        full_tail_count=full_tail_count,
        fractional_weight=fractional_weight,
        tail_mass=tail_mass,
    )
