# src/riskquant/scenario_metrics/_validate.py

from __future__ import annotations

from riskquant.scenario_metrics._types import Float64Matrix


def _validate_group_value_matrix(
    group_value_matrix: Float64Matrix,
) -> None:

    if group_value_matrix.ndim != 2:
        raise ValueError("group_value_matrix must be 2-dimensional.")

    if group_value_matrix.shape[0] == 0:
        raise ValueError("group_value_matrix must contain at least one group.")

    if group_value_matrix.shape[1] == 0:
        raise ValueError("group_value_matrix must contain at least one scenario.")


def _validate_confidence_level(
    confidence_level: float,
) -> None:

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1.")
