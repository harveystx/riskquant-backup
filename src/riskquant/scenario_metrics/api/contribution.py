# src/riskquant/scenario_metrics/api/contribution.py

from __future__ import annotations

from riskquant.scenario_metrics._computer import (
    _compute_lower_tail_parameters,
    _compute_tail_contribution,
)
from riskquant.scenario_metrics._result import ContributionResult
from riskquant.scenario_metrics._types import Float64Matrix, Float64Vector
from riskquant.scenario_metrics._validate import (
    _validate_component_matrix_and_total,
    _validate_confidence_level,
)


def compute_var_es_contribution(
    component_matrix: Float64Matrix,
    total_row: Float64Vector,
    confidence_level: float,
) -> ContributionResult:
    """
    Euler lower-tail VaR and ES contributions for additive components.

    This is the single, shared home for tail-risk allocation. Market-risk and
    credit-risk callers both route through it (after mapping into the lower-tail
    P&L convention), so contribution VaR/ES use exactly the same tail geometry
    and kernel as the headline ``compute_var_es``.

    Parameters
    ----------
    component_matrix
        C-contiguous float64 matrix with shape ``(n_components, n_scenarios)``.
        Rows are additive components (obligors, desks, risk factors, …) whose
        column sum is the portfolio total.

    total_row
        C-contiguous float64 vector with shape ``(n_scenarios,)``. Must equal
        the column sum of ``component_matrix`` for the footing identity to hold
        (the total is often accumulated online, so it is passed explicitly
        rather than re-derived).

    confidence_level
        Confidence level in ``(0, 1)``.

    Returns
    -------
    ContributionResult
        ``var_contribution`` and ``es_contribution``, each shape
        ``(n_components,)``, aligned to the component row order.

    Footing
    -------
    Both allocations are evaluated over the tail scenario set defined by
    ``total_row`` at ``confidence_level``, so

        sum(var_contribution) == VaR(total_row)
        sum(es_contribution)  == ES(total_row)

    to within floating-point summation error, matching ``compute_var_es`` on
    the same total row.

    Convention
    ----------
    Lower-tail (P&L) convention: the worst outcomes are the most negative
    values, identical to ``compute_var_es``. Loss-oriented callers negate on
    the way in and out (see ``credit.aggregate.loss_metrics``).
    """

    _validate_component_matrix_and_total(
        component_matrix,
        total_row,
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
        scenario_count=component_matrix.shape[1],
        confidence_level=canonical_confidence_level,
    )

    return _compute_tail_contribution(
        component_matrix=component_matrix,
        total_row=total_row,
        var_index=var_index,
        full_tail_count=full_tail_count,
        fractional_weight=fractional_weight,
        tail_mass=tail_mass,
    )
