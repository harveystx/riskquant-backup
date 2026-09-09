# src/riskquant/scenario_metrics/_validate.py

from __future__ import annotations

import numpy as np

from riskquant.scenario_metrics._types import Float64Matrix, Float64Vector


def _validate_group_value_matrix(
    group_value_matrix: Float64Matrix,
) -> None:
    """
    Validate the canonical scenario-metrics input contract.

    Contract
    --------
    * 2-dimensional
    * at least one group
    * at least one scenario
    * float64
    * C-contiguous
    * every value finite

    The computational kernel deliberately relies on these invariants and
    therefore performs no defensive validation.
    """

    if group_value_matrix.ndim != 2:
        raise ValueError(
            "group_value_matrix must be 2-dimensional; "
            f"got ndim={group_value_matrix.ndim}."
        )

    n_groups, n_scenarios = group_value_matrix.shape

    if n_groups == 0:
        raise ValueError("group_value_matrix must contain at least one group.")

    if n_scenarios == 0:
        raise ValueError("group_value_matrix must contain at least one scenario.")

    if group_value_matrix.dtype != np.dtype(np.float64):
        raise TypeError(
            "group_value_matrix must have dtype float64; "
            f"got {group_value_matrix.dtype}."
        )

    if not group_value_matrix.flags.c_contiguous:
        raise ValueError("group_value_matrix must be C-contiguous.")

    if not np.isfinite(group_value_matrix).all():
        raise ValueError("group_value_matrix must contain only finite values.")


def _validate_component_matrix_and_total(
    component_matrix: Float64Matrix,
    total_row: Float64Vector,
) -> None:
    """
    Validate the contribution input contract.

    Contract
    --------
    * ``component_matrix`` is a ``(K, S)`` finite, C-contiguous float64 matrix
      with at least one component and one scenario.
    * ``total_row`` is a ``(S,)`` finite, C-contiguous float64 vector whose
      length matches the component matrix's scenario axis.

    The footing identity ``sum_k contribution[k] == metric(total_row)`` only
    holds when ``total_row`` is the column sum of ``component_matrix``. That is
    the caller's responsibility (the total is frequently accumulated online,
    independently of any materialised component matrix), so it is documented
    rather than re-derived and checked here.
    """

    _validate_group_value_matrix(component_matrix)

    if total_row.ndim != 1:
        raise ValueError(
            f"total_row must be 1-dimensional; got ndim={total_row.ndim}."
        )

    if total_row.dtype != np.dtype(np.float64):
        raise TypeError(
            f"total_row must have dtype float64; got {total_row.dtype}."
        )

    if not total_row.flags.c_contiguous:
        raise ValueError("total_row must be C-contiguous.")

    n_scenarios = component_matrix.shape[1]

    if total_row.shape[0] != n_scenarios:
        raise ValueError(
            "total_row length must match the component scenario axis; "
            f"got total_row S={total_row.shape[0]} vs component S={n_scenarios}."
        )

    if not np.isfinite(total_row).all():
        raise ValueError("total_row must contain only finite values.")


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    """
    Validate the scalar confidence level.

    The accepted domain is the strict open interval (0, 1).
    """

    try:
        value = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise TypeError("confidence_level must be a real scalar.") from exc

    if not np.isfinite(value):
        raise ValueError("confidence_level must be finite.")

    if not 0.0 < value < 1.0:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1.")

    return value
