# src/riskquant/solutions/credit_var/catalog.py
"""
Credit-VaR model catalog (solution-tier).

The menu the front-end renders and the source of per-model parameter rules the
request schema validates against. This is deliberately solution-side: it carries
human display labels and the authoring contract, which the core engine has no
business knowing. It lists only models this solution can run; the underlying
construction lives in `riskquant.solutions.credit_var.recipe`.
"""

from __future__ import annotations

__all__ = [
    "ParamSpec",
    "ModelEntry",
    "MODEL_CATALOG",
    "supported_model_keys",
    "model_entry",
]

from dataclasses import dataclass, field

from riskquant.solutions.credit_var.recipes import SUPPORTED_RECIPES


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One scalar knob a model exposes to the author."""

    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    min_inclusive: bool = True
    max_inclusive: bool = False

    def contains(self, value: float) -> bool:
        lo = value >= self.min_value if self.min_inclusive else value > self.min_value
        hi = value <= self.max_value if self.max_inclusive else value < self.max_value
        return bool(lo and hi)

    @property
    def interval_repr(self) -> str:
        lb = "[" if self.min_inclusive else "("
        rb = "]" if self.max_inclusive else ")"
        return f"{lb}{self.min_value}, {self.max_value}{rb}"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    key: str
    display_name: str
    description: str
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)
    requires_loadings: bool = False  # multi-factor: needs a systemic-loadings sheet


_RHO = ParamSpec(
    key="rho",
    label="Asset correlation (rho)",
    default=0.15,
    min_value=0.0,
    max_value=1.0,
    min_inclusive=True,
    max_inclusive=False,
)
_NU = ParamSpec(
    key="nu",
    label="Degrees of freedom (nu)",
    default=8.0,
    min_value=2.0,
    max_value=250.0,
    min_inclusive=True,
    max_inclusive=True,
)
_BETA_ALPHA = ParamSpec(
    key="alpha",
    label="Beta shape alpha",
    default=1.0,
    min_value=0.0,
    max_value=1000.0,
    min_inclusive=False,
    max_inclusive=True,
)
_BETA_BETA = ParamSpec(
    key="beta",
    label="Beta shape beta",
    default=24.0,
    min_value=0.0,
    max_value=1000.0,
    min_inclusive=False,
    max_inclusive=True,
)
_STRESS_PROB = ParamSpec(
    key="stress_prob",
    label="Stress-regime probability",
    default=0.05,
    min_value=0.0,
    max_value=1.0,
    min_inclusive=True,
    max_inclusive=True,
)
_STRESS_MULT = ParamSpec(
    key="stress_pd_multiplier",
    label="Stressed PD multiplier",
    default=3.0,
    min_value=1.0,
    max_value=50.0,
    min_inclusive=True,
    max_inclusive=True,
)


MODEL_CATALOG: dict[str, ModelEntry] = {
    "forced_default": ModelEntry(
        key="forced_default",
        display_name="Forced Default",
        description="Every obligor defaults in every scenario; loss = EAD x LGD. "
        "A deterministic worst-case bound (PD is ignored).",
        params=(),
    ),
    "beta_binomial": ModelEntry(
        key="beta_binomial",
        display_name="Beta-Binomial",
        description="Conditionally-independent defaults driven by a shared per-"
        "scenario PD drawn from Beta(alpha, beta). Mean PD = alpha/(alpha+beta); "
        "the obligor PD column is not used.",
        params=(_BETA_ALPHA, _BETA_BETA),
    ),
    "stress_regime_bernoulli": ModelEntry(
        key="stress_regime_bernoulli",
        display_name="Stress-Regime Bernoulli",
        description="Independent-Bernoulli defaults with a shared stress regime "
        "that fires with probability stress_prob and multiplies every PD by "
        "stress_pd_multiplier (capped at 1).",
        params=(_STRESS_PROB, _STRESS_MULT),
    ),
    "sf_gaussian": ModelEntry(
        key="sf_gaussian",
        display_name="Single-Factor Gaussian Copula",
        description="One systemic factor with asset correlation rho (Vasicek / "
        "Basel ASRF). The standard desk credit-VaR model.",
        params=(_RHO,),
    ),
    "independent_bernoulli": ModelEntry(
        key="independent_bernoulli",
        display_name="Independent Bernoulli",
        description="Each obligor defaults independently with probability PD. "
        "No correlation; the diversified baseline.",
        params=(),
    ),
    "sf_students_t": ModelEntry(
        key="sf_students_t",
        display_name="Single-Factor Student-t Copula",
        description="Single systemic factor with asset correlation rho plus a "
        "shared chi-square mixing variable at nu degrees of freedom (tail "
        "dependence / fatter joint tails than Gaussian at the same rho).",
        params=(_RHO, _NU),
    ),
    "mf_gaussian": ModelEntry(
        key="mf_gaussian",
        display_name="Multi-Factor Gaussian Copula",
        description="Per-obligor systemic loadings across several factors (e.g. a "
        "global factor plus sector factors), supplied as a loadings sheet. "
        "Idiosyncratic weight is the residual sqrt(1 - sum beta^2).",
        params=(),
        requires_loadings=True,
    ),
    "mf_students_t": ModelEntry(
        key="mf_students_t",
        display_name="Multi-Factor Student-t Copula",
        description="Multi-factor Gaussian loadings plus a shared chi-square "
        "mixing variable at nu degrees of freedom (multi-factor structure with "
        "tail dependence). Needs a loadings sheet.",
        params=(_NU,),
        requires_loadings=True,
    ),
}

# No-drift guard: the catalog must not advertise a model the core can't assemble.
_missing = set(MODEL_CATALOG) - set(SUPPORTED_RECIPES)
if _missing:
    raise RuntimeError(
        f"credit_var.catalog advertises model(s) with no core recipe: {sorted(_missing)}."
    )


def supported_model_keys() -> tuple[str, ...]:
    return tuple(MODEL_CATALOG)


def model_entry(key: str) -> ModelEntry:
    entry = MODEL_CATALOG.get(key)
    if entry is None:
        raise KeyError(f"unknown model '{key}'. Supported: {sorted(MODEL_CATALOG)}.")
    return entry
