# src/riskquant/solutions/scenario_models/schemas.py
"""
Typed configuration for selecting and parameterising a SimLab
``ScenarioGenerator``.

This is the SOLUTION-tier seam: strings and Pydantic live here, never in the
NumPy/Numba core. Each model has its OWN config carrying ONLY its parameters
(no giant nullable blob); ``model`` is the discriminator and ``extra="forbid"``
rejects any parameter belonging to another model. Data (the risk-factor matrix)
is NOT configuration — it is supplied at build time from market data.

Only the deterministic ``HistoricalGenerator`` is wired today. When a second
model is added this alias becomes a Pydantic discriminated union
``Annotated[HistoricalGeneratorConfig | <Next>, Field(discriminator="model")]``;
the ``ScenarioModelConfig`` alias keeps callers stable across that change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


class HistoricalGeneratorConfig(BaseModel):
    """
    Selects the deterministic historical rolling-window scenario model.

    ``horizon`` is the only model parameter: a strict positive integer h-day
    window length. ``StrictInt`` rejects float and bool (no silent ``1.0`` or
    ``True``); ``gt=0`` rejects non-positive. The generator's own constructor
    validation still runs at build time — this contract does not duplicate the
    model's ``T - horizon`` scenario-count logic.
    """

    model_config = _STRICT

    model: Literal["historical"]
    horizon: StrictInt = Field(gt=0, description="h-day rolling-window length.")


# Single member today. Promote to
# ``Annotated[HistoricalGeneratorConfig | <Next>, Field(discriminator="model")]``
# when the second model is wired; callers import this alias, not the concrete
# class, so the promotion is source-compatible.
ScenarioModelConfig = HistoricalGeneratorConfig
