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
    Compute lower-tail VaR and ES for each group.

    Parameters
    ----------
    group_value_matrix : Float64Matrix
        Shape (n_groups, n_scenarios).

    confidence_level : float
        Example: 0.99.

    Returns
    -------
    var_es_result : VarEsResult
        Contains the VaR and ES for each group.
    """

    _validate_group_value_matrix(
        group_value_matrix,
    )

    _validate_confidence_level(
        confidence_level,
    )

    kth_index, tail_count = _compute_lower_tail_parameters(
        scenario_count=group_value_matrix.shape[1],
        confidence_level=confidence_level,
    )

    return _compute_tail_var_es(
        group_value_matrix=group_value_matrix,
        kth_index=kth_index,
        tail_count=tail_count,
    )
