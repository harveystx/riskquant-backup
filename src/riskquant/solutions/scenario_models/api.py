# src/riskquant/solutions/scenario_models/api.py
"""
Public surface of the scenario-model seam: parse a typed config, then build the
concrete ``ScenarioGenerator`` it selects.

The seam adds NO numerical logic and introduces NO parallel interface: it
returns the existing ``ScenarioGenerator`` protocol, which the downstream
runner already consumes unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.base import ScenarioGenerator
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.solutions.scenario_models.schemas import (
    HistoricalGeneratorConfig,
    ScenarioModelConfig,
)

__all__ = ["build_scenario_generator", "parse_scenario_model_config"]

_ADAPTER: TypeAdapter[ScenarioModelConfig] = TypeAdapter(ScenarioModelConfig)


def parse_scenario_model_config(data: Mapping[str, Any]) -> ScenarioModelConfig:
    """
    Validate an untyped mapping into a typed scenario-model config. Fails fast
    on an unknown ``model``, a missing or mistyped parameter, or a parameter
    that belongs to another model.
    """
    return _ADAPTER.validate_python(data)


def build_scenario_generator(
    config: ScenarioModelConfig, *, matrix: RiskFactorMatrix
) -> ScenarioGenerator:
    """
    Construct the concrete ``ScenarioGenerator`` selected by ``config``.

    Data (the risk-factor matrix) is supplied here, separate from the model
    config. The generator's own constructor validation runs unchanged; no
    model parameter is injected that the selected model does not declare.
    """
    if isinstance(config, HistoricalGeneratorConfig):
        return HistoricalGenerator(matrix=matrix, horizon=config.horizon)
    raise TypeError(f"Unsupported scenario-model config: {type(config).__name__}")
