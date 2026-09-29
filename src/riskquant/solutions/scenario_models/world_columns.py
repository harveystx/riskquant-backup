# src/riskquant/solutions/scenario_models/world_columns.py
"""
World-column declaration and binding for the Proxy V1 seam (D088 Slice 3).

A scenario world is one horizon ``h`` and a set of columns keyed
``(d, tau, h)`` (D088 §4). The world, not the proxy model, owns ``h``
(D088 §12.5, §13.5). This module answers two questions:

1. ``declare_world_columns``: which ``(d, tau)`` columns does this world
   supply, and at which ``h``?

   - Parametric world: exactly the columns the data declares,
     ``(ParametricRiskData.factor_ids[i], ParametricRiskData.chart_ids[i])``
     (D091), at ``h = resolve_parametric_binding_horizon(config, data)``
     (the D090 equality gate, reused unchanged).
   - Matrix world: a level-backed engine builds ``T_tau`` of any factor in
     its matrix, so it supplies each requested pair whose driver is a matrix
     factor, at ``h = config.horizon`` (the number of observation intervals of
     the world's own date axis). A requested driver that is not in the matrix
     fails; there is no substitute factor, tenor or tier.

2. ``bind_operator_to_world``: bind the lowered operator's terms to world
   columns. The request is first bound against the declared world keys, so a
   column the world does not supply fails before any scenario is generated;
   the terms are then bound against the request, so each term indexes the
   request-ordered columns a generator's ``iter_batches`` returns. Both steps
   use ``bind_term_columns``, the one ``(d, tau, h)`` binder.

Nothing here selects a source, ranks availability, converts a chart, infers a
horizon or weights a column. The world request is not an identity artifact:
world identity remains ``scenario_config`` plus the data digest, and the
proxy operator's identity remains ``proxy_model_digest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.axis_operator import AxisOperator, world_request
from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.solutions.scenario_models.api import resolve_parametric_binding_horizon
from riskquant.solutions.scenario_models.schemas import (
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
    ScenarioModelConfig,
)

__all__ = [
    "WorldBinding",
    "WorldColumns",
    "bind_operator_to_world",
    "declare_world_columns",
]

_G = TypeVar("_G", bound=np.generic)

_MATRIX_CONFIGS = (
    HistoricalGeneratorConfig,
    SbbGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    FilteredBootstrapGeneratorConfig,
)
_PREVIEW = 10


def _frozen(arr: npt.NDArray[_G]) -> npt.NDArray[_G]:
    out = np.array(arr, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class WorldColumns:
    """The ``(d, tau)`` columns a world declares, and its horizon ``h``."""

    driver_rf_ids: npt.NDArray[np.int32]
    chart_ids: npt.NDArray[np.int8]
    horizon: int


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class WorldBinding:
    """
    The operator bound to one world.

    ``request_driver_rf_ids`` / ``request_chart_ids`` are the unique
    ``(d, tau)`` pairs to stream, in first-seen order; ``horizon`` is the
    world's ``h``; ``term_column_uix[t]`` is the stream column of operator
    term ``t``, in canonical term order.
    """

    request_driver_rf_ids: npt.NDArray[np.int32]
    request_chart_ids: npt.NDArray[np.int8]
    horizon: int
    term_column_uix: npt.NDArray[np.int32]


def declare_world_columns(
    config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    *,
    request_driver_rf_ids: npt.NDArray[np.int32],
    request_chart_ids: npt.NDArray[np.int8],
) -> WorldColumns:
    """
    Declare the world's columns and horizon for a request.

    Raises
    ------
    TypeError
        Config and data kinds do not pair, or an unsupported config.
    ValueError
        D090 period mismatch (parametric), or a requested driver absent from
        the matrix (matrix world).
    """
    if isinstance(config, ParametricGeneratorConfig):
        # D090: requested and effective periods are equal; h = count(H_W).
        h = resolve_parametric_binding_horizon(config, data)
        if not isinstance(data, ParametricRiskData):  # the gate already refused it
            raise TypeError("a parametric world requires ParametricRiskData.")
        return WorldColumns(
            driver_rf_ids=_frozen(data.factor_ids),
            chart_ids=_frozen(data.chart_ids),
            horizon=h,
        )
    if isinstance(config, _MATRIX_CONFIGS):
        if not isinstance(data, RiskFactorMatrix):
            raise TypeError(
                f"{type(config).__name__} requires a RiskFactorMatrix, got "
                f"{type(data).__name__}."
            )
        absent = ~np.isin(request_driver_rf_ids, data.factor_ids)
        if np.any(absent):
            missing = [
                (int(d), int(c))
                for d, c in zip(
                    request_driver_rf_ids[absent][:_PREVIEW],
                    request_chart_ids[absent][:_PREVIEW],
                    strict=True,
                )
            ]
            raise ValueError(
                f"{int(absent.sum())} requested world column(s) (d, tau) name a "
                f"driver that is not in the matrix world: {missing}. No other "
                "factor, tenor or source is substituted (D088 §9)."
            )
        return WorldColumns(
            driver_rf_ids=_frozen(request_driver_rf_ids),
            chart_ids=_frozen(request_chart_ids),
            horizon=int(config.horizon),
        )
    raise TypeError(f"Unsupported scenario-model config: {type(config).__name__}")


def bind_operator_to_world(
    operator: AxisOperator,
    config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
) -> WorldBinding:
    """
    Bind the lowered operator to the world, before any scenario is generated.

    Parameters
    ----------
    operator
        ``lower_axis_operator`` output for this run.
    config, data
        The scenario world.

    Returns
    -------
    WorldBinding

    Raises
    ------
    ValueError
        A requested ``(d, tau, h)`` column the world does not supply, a horizon
        mismatch, or any ``declare_world_columns`` failure.
    """
    req_d, req_tau = world_request(operator)
    world = declare_world_columns(
        config, data, request_driver_rf_ids=req_d, request_chart_ids=req_tau
    )
    # The proxy model carries no horizon: its (d, tau) pairs are bound at the
    # world's declared h (D088 §12.5). For a parametric world the D090 gate has
    # already made the requested period equal to the data's effective period;
    # a matrix world declares h once, in its configuration.
    # Every requested world variable must exist at that h.
    bind_term_columns(
        term_driver_rf_ids=req_d,
        term_chart_ids=req_tau,
        requested_horizon=world.horizon,
        world_column_driver_rf_ids=world.driver_rf_ids,
        world_column_chart_ids=world.chart_ids,
        world_horizon=world.horizon,
    )
    # Each operator term to its column in the request-ordered stream.
    term_column_uix = bind_term_columns(
        term_driver_rf_ids=operator.term_driver_rf_ids,
        term_chart_ids=operator.term_chart_ids,
        requested_horizon=world.horizon,
        world_column_driver_rf_ids=req_d,
        world_column_chart_ids=req_tau,
        world_horizon=world.horizon,
    )
    return WorldBinding(
        request_driver_rf_ids=req_d,
        request_chart_ids=req_tau,
        horizon=world.horizon,
        term_column_uix=_frozen(term_column_uix.astype(np.int32, copy=False)),
    )
