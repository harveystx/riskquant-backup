# src/riskquant/solutions/scenario_models/api.py
"""
Public surface of the scenario-model seam: parse a typed config, then build the
concrete ``ScenarioGenerator`` it selects from a typed construction-data object.

The seam adds NO numerical logic and introduces NO parallel interface: it
returns the existing ``ScenarioGenerator`` protocol, which the downstream runner
already consumes unchanged. Configuration and construction data are separate
arguments; a config paired with the wrong data type fails fast here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.base import ScenarioGenerator
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator
from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.simlab.scenarios.parametric_data import (
    ForecastPeriodUnit,
    ParametricRiskData,
)
from riskquant.solutions.scenario_models.schemas import (
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    RegimeSwitchingDistribution,
    SbbGeneratorConfig,
    ScenarioModelConfig,
    StudentTDistribution,
    VarianceGammaDistribution,
)

__all__ = [
    "build_scenario_generator",
    "parse_scenario_model_config",
    "resolve_parametric_binding_horizon",
]

# The typed construction-data object required to build a generator. Each config
# is paired with exactly one data type; a mismatch is rejected in build.
ScenarioModelData = RiskFactorMatrix | ParametricRiskData

_ADAPTER: TypeAdapter[ScenarioModelConfig] = TypeAdapter(ScenarioModelConfig)


def parse_scenario_model_config(data: Mapping[str, Any]) -> ScenarioModelConfig:
    """
    Validate an untyped mapping into a typed scenario-model config. Fails fast
    on an unknown ``model``, a missing or mistyped parameter, or a parameter
    that belongs to another model.
    """
    return _ADAPTER.validate_python(data)


def _require_matrix(
    data: ScenarioModelData, config: ScenarioModelConfig
) -> RiskFactorMatrix:
    if not isinstance(data, RiskFactorMatrix):
        raise TypeError(
            f"{type(config).__name__} requires a RiskFactorMatrix, got "
            f"{type(data).__name__}."
        )
    return data


def _require_parametric_data(
    data: ScenarioModelData, config: ScenarioModelConfig
) -> ParametricRiskData:
    if not isinstance(data, ParametricRiskData):
        raise TypeError(
            f"{type(config).__name__} requires ParametricRiskData, got "
            f"{type(data).__name__}."
        )
    return data


def _build_parametric(
    config: ParametricGeneratorConfig, pdata: ParametricRiskData
) -> ParametricGenerator:
    """Map the nested distribution variant onto the generator's flat parameters.
    The generator re-validates every input; no numerical logic lives here."""
    dist = config.distribution
    kwargs: dict[str, Any] = {
        "available_factor_ids": pdata.factor_ids,
        "cholesky_matrix": pdata.cholesky_matrix,
        "volatilities": pdata.volatilities,
        "total_scenarios": config.total_scenarios,
        "distribution_kind": dist.kind,
        "seed": config.seed,
    }
    if isinstance(dist, StudentTDistribution):
        kwargs["student_t_df"] = dist.df
    elif isinstance(dist, VarianceGammaDistribution):
        kwargs["vg_nu"] = dist.nu
    elif isinstance(dist, RegimeSwitchingDistribution):
        kwargs["prob_crisis"] = dist.prob_crisis
        kwargs["crisis_vol_multipliers"] = dist.crisis_vol_multipliers
    return ParametricGenerator(**kwargs)


def resolve_parametric_binding_horizon(
    config: ParametricGeneratorConfig, data: ScenarioModelData
) -> int:
    """
    D090 composition gate for a parametric world.

    The requested period (``config.forecast_horizon``) and the effective period
    declared by the data must be equal: same count and same unit. Equality is
    exact; no unit or count conversion exists. Only after the check passes is
    the period reduced to the integer horizon ``h = count(H_W)``, the one value
    the ``(d, τ, h)`` binder accepts.

    Parameters
    ----------
    config
        The parametric scenario configuration (declares ``H_requested``).
    data
        The construction data (declares ``H_effective``); must be
        ``ParametricRiskData``.

    Returns
    -------
    int
        ``h = count(H_W)``, a plain ``int``.

    Raises
    ------
    TypeError
        ``data`` is not ``ParametricRiskData``.
    ValueError
        The requested and effective periods differ.
    """
    pdata = _require_parametric_data(data, config)
    requested = config.forecast_horizon
    requested_unit = int(ForecastPeriodUnit[requested.unit])
    effective_count = int(pdata.effective_period_count)
    effective_unit = int(pdata.effective_period_unit)
    if requested.count != effective_count or requested_unit != effective_unit:
        raise ValueError(
            "Parametric forecast period mismatch (D090): requested "
            f"({requested.count}, {requested.unit}), effective "
            f"({effective_count}, unit code {effective_unit}). "
            "The periods must be equal; no conversion exists."
        )
    return requested.count


def build_scenario_generator(
    config: ScenarioModelConfig, *, data: ScenarioModelData
) -> ScenarioGenerator:
    """
    Construct the concrete ``ScenarioGenerator`` selected by ``config`` from its
    construction ``data``. Matrix-backed models require a ``RiskFactorMatrix``;
    the parametric model requires a ``ParametricRiskData``. A config paired with
    the wrong data type raises ``TypeError``. The generator's own constructor
    validation runs unchanged; no undeclared model parameter is injected.
    """
    if isinstance(config, HistoricalGeneratorConfig):
        return HistoricalGenerator(
            matrix=_require_matrix(data, config), horizon=config.horizon
        )
    if isinstance(config, FilteredHistoricalGeneratorConfig):
        return FilteredHistoricalGenerator(
            matrix=_require_matrix(data, config),
            horizon=config.horizon,
            decay_factor=config.decay_factor,
            max_vol_ratio=config.max_vol_ratio,
        )
    if isinstance(config, SbbGeneratorConfig):
        return StationaryBlockBootstrapGenerator(
            matrix=_require_matrix(data, config),
            total_scenarios=config.total_scenarios,
            expected_block_size=config.expected_block_size,
            horizon=config.horizon,
            seed=config.seed,
        )
    if isinstance(config, FilteredBootstrapGeneratorConfig):
        return FilteredBootstrapGenerator(
            matrix=_require_matrix(data, config),
            total_scenarios=config.total_scenarios,
            horizon=config.horizon,
            decay_factor=config.decay_factor,
            max_vol_ratio=config.max_vol_ratio,
            seed=config.seed,
        )
    if isinstance(config, ParametricGeneratorConfig):
        pdata = _require_parametric_data(data, config)
        # D090: an independently built generator obeys the same period gate.
        resolve_parametric_binding_horizon(config, pdata)
        return _build_parametric(config, pdata)
    raise TypeError(f"Unsupported scenario-model config: {type(config).__name__}")
