# src/riskquant/scenario_metrics/_result.py

from __future__ import annotations

from dataclasses import dataclass

from riskquant.scenario_metrics._types import Float64Vector


@dataclass(frozen=True, slots=True, kw_only=True)
class VarEsResult:
    """
    Immutable container for empirical VaR and ES vectors.

    The arrays are owned by the computation layer and returned read-only.
    """

    var: Float64Vector
    es: Float64Vector


@dataclass(frozen=True, slots=True, kw_only=True)
class ContributionResult:
    """
    Immutable container for per-component Euler VaR and ES contributions.

    Both vectors are aligned to the component (row) order of the input matrix
    and are returned read-only. By construction:

        sum(var_contribution) == VaR(total)
        sum(es_contribution)  == ES(total)

    to within floating-point summation error, because both are evaluated over
    the same tail scenario set the headline VaR/ES uses.
    """

    var_contribution: Float64Vector
    es_contribution: Float64Vector
