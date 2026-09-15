# src/riskquant/solutions/scenario_models/schemas.py
"""
Typed configuration for selecting and parameterising a SimLab
``ScenarioGenerator``.

This is the SOLUTION-tier seam: strings and Pydantic live here, never in the
NumPy/Numba core. Each model has its OWN config carrying ONLY its parameters
(no giant nullable blob); ``model`` is the discriminator and ``extra="forbid"``
rejects any parameter belonging to another model. Data (the risk-factor matrix)
is NOT configuration — it is supplied at build time from market data.

Two models are wired: the deterministic ``HistoricalGenerator`` and the
stochastic ``StationaryBlockBootstrapGenerator``. ``ScenarioModelConfig`` is a
Pydantic discriminated union tagged by ``model``; a third model is a one-line
extension of that union.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

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


class SbbGeneratorConfig(BaseModel):
    """
    Selects the Stationary Block Bootstrap model.

    Model-specific parameters only. ``horizon`` and ``total_scenarios`` are
    strict positive integers; ``expected_block_size`` is a CONTINUOUS mean block
    length (the geometric block distribution's mean) — a finite real >= 1.0,
    accepting int or float but rejecting bool and non-finite; ``seed`` is a
    strict optional integer (``None`` selects OS entropy, no reproducibility
    claim, per D029). The generator's ``__post_init__`` remains authoritative
    for runtime/model invariants (float64 matrix, T >= 2, canonical persistence).
    """

    model_config = _STRICT

    model: Literal["sbb"]
    horizon: StrictInt = Field(gt=0, description="h-day path length.")
    total_scenarios: StrictInt = Field(gt=0, description="number of paths to draw.")
    expected_block_size: float = Field(
        default=1.0,
        ge=1.0,
        allow_inf_nan=False,
        description="geometric mean block length (continuous, >= 1.0).",
    )
    seed: StrictInt | None = Field(
        default=None, description="RNG seed; None = OS entropy (not reproducible)."
    )

    @field_validator("expected_block_size", mode="before")
    @classmethod
    def _reject_bool_block_size(cls, value: object) -> object:
        # bool is a subclass of int and would coerce to 1.0/0.0; a block length
        # is a real number, never a boolean.
        if isinstance(value, bool):
            raise ValueError("expected_block_size must be a real number, not bool.")
        return value


# Two real models now, so this is a genuine Pydantic discriminated union tagged
# by ``model`` — not a speculative abstraction. Callers import this alias, so
# adding the third model is a one-line union extension here.
ScenarioModelConfig = Annotated[
    HistoricalGeneratorConfig | SbbGeneratorConfig,
    Field(discriminator="model"),
]
