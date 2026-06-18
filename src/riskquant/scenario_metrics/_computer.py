# src/riskquant/scenario_metrics/_computer.py

from __future__ import annotations

import numpy as np

from riskquant.scenario_metrics._kernels._tail_metrics import (
    _kernel_lower_tail_var_es,
)
from riskquant.scenario_metrics._result import VarEsResult
from riskquant.scenario_metrics._types import Float64Matrix


def _compute_lower_tail_parameters(
    scenario_count: int,
    confidence_level: float,
) -> tuple[int, int]:
    """
    Convert confidence level into lower-tail selection parameters.

    Returns
    -------
    kth_index
        VaR order statistic index.

    tail_count
        Number of observations included in ES.
    """

    alpha = 1.0 - confidence_level

    tail_count = max(
        1,
        int(alpha * scenario_count),
    )

    kth_index = tail_count - 1

    return kth_index, tail_count


def _compute_tail_var_es(
    group_value_matrix: Float64Matrix,
    kth_index: int,
    tail_count: int,
) -> VarEsResult:
    """
    Internal allocation layer.
    """

    if not group_value_matrix.flags.c_contiguous:
        group_value_matrix = np.ascontiguousarray(
            group_value_matrix,
        )

    n_groups = group_value_matrix.shape[0]

    var_out = np.empty(
        n_groups,
        dtype=np.float64,
    )

    es_out = np.empty(
        n_groups,
        dtype=np.float64,
    )

    scratchpad = np.empty_like(
        group_value_matrix,
        order="C",
    )

    _kernel_lower_tail_var_es(
        group_value_matrix=group_value_matrix,
        scratchpad=scratchpad,
        kth_index=kth_index,
        tail_count=tail_count,
        var_out=var_out,
        es_out=es_out,
    )

    return VarEsResult(var=var_out, es=es_out)
