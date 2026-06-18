# src/riskquant/scenario_metrics/_result.py
from __future__ import annotations

from dataclasses import dataclass

from riskquant.scenario_metrics._types import Float64Vector


@dataclass(frozen=True, slots=True, kw_only=True)
class VarEsResult:
    var: Float64Vector
    es: Float64Vector
