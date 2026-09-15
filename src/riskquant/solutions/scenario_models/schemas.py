# src/riskquant/solutions/scenario_models/schemas.py
"""
Typed configuration for selecting and parameterising a SimLab
``ScenarioGenerator``.

This is the SOLUTION-tier seam: strings and Pydantic live here, never in the
NumPy/Numba core. Each model has its OWN config carrying ONLY its parameters
(no giant nullable blob); ``model`` is the discriminator and ``extra="forbid"``
rejects any parameter belonging to another model. Data (the risk-factor matrix)
is NOT configuration — it is supplied at build time from market data.

Three models are wired: the deterministic ``HistoricalGenerator`` and
``FilteredHistoricalGenerator``, and the stochastic
``StationaryBlockBootstrapGenerator``. ``ScenarioModelConfig`` is a Pydantic
discriminated union tagged by ``model``; a further model is a one-line
extension of that union. All three share the ``RiskFactorMatrix`` construction
data contract.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO

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


class FilteredHistoricalGeneratorConfig(BaseModel):
    """
    Selects the Filtered Historical Simulation (FHS) model: deterministic
    rolling windows over AR(1)/EWMA-filtered, vol-rescaled returns.

    Model-specific parameters only. ``horizon`` is a strict positive integer.
    ``decay_factor`` is the EWMA memory ``lambda`` — a finite real in the open
    interval (0, 1) (the valid range for an exponentially-weighted moving
    average; the generator does not itself bound it, so the seam fails fast
    here). ``max_vol_ratio`` is the vol-amplification cap — a finite real
    >= 1.0. Both reject bool and non-finite; both default to the repository
    constants so seam construction matches the generator's own defaults.
    The generator's ``__post_init__`` stays authoritative for matrix/history
    invariants; this contract does not reproduce the EWMA/AR(1) math.
    """

    model_config = _STRICT

    model: Literal["filtered_historical"]
    horizon: StrictInt = Field(gt=0, description="h-day rolling-window length.")
    decay_factor: float = Field(
        default=EWMA_DECAY_FACTOR,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description="EWMA memory lambda in (0, 1).",
    )
    max_vol_ratio: float = Field(
        default=MAX_EWMA_VOL_RATIO,
        ge=1.0,
        allow_inf_nan=False,
        description="vol-amplification cap (>= 1.0).",
    )

    @field_validator("decay_factor", "max_vol_ratio", mode="before")
    @classmethod
    def _reject_bool_real(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a real number, not bool.")
        return value


class FilteredBootstrapGeneratorConfig(BaseModel):
    """
    Selects the Filtered Bootstrap model: stochastic bootstrap resampling of
    AR(1)/EWMA-filtered, vol-rescaled residuals.

    Model-specific parameters only, and the exact `FilteredBootstrapGenerator`
    constructor contract: `horizon` and `total_scenarios` are strict positive
    integers; `decay_factor` (EWMA memory lambda, open interval (0, 1)) and
    `max_vol_ratio` (vol-amplification cap, >= 1.0) are finite reals the
    generator does not itself bound, so the seam fails fast here (same as
    `filtered_historical`, D033); `seed` is an optional strict integer
    (None = OS entropy, no reproducibility claim, D029). The generator's
    `__post_init__` stays authoritative for matrix float64 / history-length /
    positive-int invariants; this contract does not reproduce the EWMA/AR(1) or
    bootstrap math.
    """

    model_config = _STRICT

    model: Literal["filtered_bootstrap"]
    horizon: StrictInt = Field(gt=0, description="h-day path length.")
    total_scenarios: StrictInt = Field(gt=0, description="number of paths to draw.")
    decay_factor: float = Field(
        default=EWMA_DECAY_FACTOR,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description="EWMA memory lambda in (0, 1).",
    )
    max_vol_ratio: float = Field(
        default=MAX_EWMA_VOL_RATIO,
        ge=1.0,
        allow_inf_nan=False,
        description="vol-amplification cap (>= 1.0).",
    )
    seed: StrictInt | None = Field(
        default=None, description="RNG seed; None = OS entropy (not reproducible)."
    )

    @field_validator("decay_factor", "max_vol_ratio", mode="before")
    @classmethod
    def _reject_bool_real(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a real number, not bool.")
        return value


# Parametric distribution configuration — a nested discriminated union tagged by
# ``kind``. Each variant carries ONLY its own parameters, so invalid parameter
# combinations (e.g. standard_normal with a df, or student_t without one) are
# unrepresentable rather than merely rejected.


def _require_real(value: object, name: str) -> object:
    """Strict real: accept int or float (integer-valued reals allowed for these
    continuous parameters), reject bool and non-numeric (incl. str). Finiteness
    is enforced by ``allow_inf_nan=False`` on the field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number (int or float).")
    return value


class StandardNormalDistribution(BaseModel):
    model_config = _STRICT
    kind: Literal["standard_normal"]


class LaplaceDistribution(BaseModel):
    model_config = _STRICT
    kind: Literal["laplace"]


class StudentTDistribution(BaseModel):
    model_config = _STRICT
    kind: Literal["student_t"]
    df: float = Field(
        gt=2.0, allow_inf_nan=False, description="degrees of freedom > 2."
    )

    @field_validator("df", mode="before")
    @classmethod
    def _real(cls, v: object) -> object:
        return _require_real(v, "df")


class VarianceGammaDistribution(BaseModel):
    model_config = _STRICT
    kind: Literal["variance_gamma"]
    nu: float = Field(gt=0.0, allow_inf_nan=False, description="VG variance rate > 0.")

    @field_validator("nu", mode="before")
    @classmethod
    def _real(cls, v: object) -> object:
        return _require_real(v, "nu")


class RegimeSwitchingDistribution(BaseModel):
    model_config = _STRICT
    kind: Literal["regime_switching_gmm"]
    prob_crisis: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, description="crisis probability in [0, 1]."
    )
    # Scalar (all factors) or per-factor list of positive multipliers. The
    # per-factor length is checked against K at construction (data-coupled).
    crisis_vol_multipliers: float | list[float] = Field(
        description="crisis vol multiplier(s), strictly positive."
    )

    @field_validator("prob_crisis", mode="before")
    @classmethod
    def _real_prob(cls, v: object) -> object:
        return _require_real(v, "prob_crisis")

    @field_validator("crisis_vol_multipliers", mode="before")
    @classmethod
    def _real_mults(cls, v: object) -> object:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError("crisis_vol_multipliers must be non-empty.")
            out = [_require_real(x, "crisis_vol_multipliers") for x in v]
            for x in out:
                if not (x == x and x not in (float("inf"), float("-inf")) and x > 0.0):
                    raise ValueError(
                        "crisis_vol_multipliers must be finite and strictly positive."
                    )
            return out
        _require_real(v, "crisis_vol_multipliers")
        if not (v == v and v not in (float("inf"), float("-inf")) and v > 0.0):
            raise ValueError(
                "crisis_vol_multipliers must be finite and strictly positive."
            )
        return v


ParametricDistributionConfig = Annotated[
    StandardNormalDistribution
    | LaplaceDistribution
    | StudentTDistribution
    | VarianceGammaDistribution
    | RegimeSwitchingDistribution,
    Field(discriminator="kind"),
]


class ParametricGeneratorConfig(BaseModel):
    """
    Selects the parametric Monte-Carlo model. Common config (``total_scenarios``,
    ``seed``) plus a nested distribution variant. The numerical construction data
    (factor ids, Cholesky, vols) is NOT here — it is supplied at build time as a
    ``ParametricRiskData`` object. The generator's ``__post_init__`` stays
    authoritative for model/runtime invariants.
    """

    model_config = _STRICT

    model: Literal["parametric"]
    total_scenarios: StrictInt = Field(gt=0, description="number of paths to draw.")
    seed: StrictInt | None = Field(
        default=None, description="RNG seed; None = OS entropy (not reproducible)."
    )
    distribution: ParametricDistributionConfig


# Five real models — the discriminated union grows by one tag, no other
# machinery. Callers import this alias, so each new model is a one-line change.
ScenarioModelConfig = Annotated[
    HistoricalGeneratorConfig
    | SbbGeneratorConfig
    | FilteredHistoricalGeneratorConfig
    | ParametricGeneratorConfig
    | FilteredBootstrapGeneratorConfig,
    Field(discriminator="model"),
]
