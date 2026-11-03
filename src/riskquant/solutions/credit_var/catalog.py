# src/riskquant/solutions/credit_var/catalog.py
"""
Credit-VaR model catalog (solution-tier).

The menu the front-end renders and the source of per-model parameter rules the
request schema validates against. This is deliberately solution-side: it carries
human display labels and the authoring contract, which the core engine has no
business knowing. It lists only models this solution can run; the underlying
construction lives in `riskquant.solutions.credit_var.recipe`.

Process state (WS_CREDIT_VAR_API C.22)
--------------------------------------
`MODEL_CATALOG` is a read-only mapping (``types.MappingProxyType``) over a dict
that no name in this module keeps. `api` re-exports it, so a plain dict would
let one consumer change the menu, the parameter rules and the request
validation of every other caller in the process (F-16). Item assignment and
deletion raise ``TypeError``. The values are frozen dataclasses with tuple
fields, so no level of the catalog is mutable.
"""

from __future__ import annotations

__all__ = [
    "MODEL_CATALOG",
    "ModelEntry",
    "ParamSpec",
    "model_entry",
    "supported_model_keys",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from riskquant.solutions.credit_var.constants import (
    SF_BETA_BINOMIAL_ALPHA,
    SF_BETA_BINOMIAL_BETA,
    SF_MO_SYSTEMIC_PD,
    SF_NU,
    SF_RHO,
    SF_STRESS_REGIME_MULT,
    SF_STRESS_REGIME_PROB,
)
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
    requires_groups: bool = False  # grouped: needs per-group params + obligor groups
    accepts_obligor_rho: bool = False  # SF copulas: optional per-obligor rho column


# Every ParamSpec default is the shared constant the recipe also resolves an
# omitted param to (see `constants.py`). The catalog is the published surface a
# caller reads (`MODEL_CATALOG` is exported from `api`), so an advertised
# default that differs from the effective one is a lie to the author.
_RHO = ParamSpec(
    key="rho",
    label="Asset Correlation (rho)",
    default=SF_RHO,
    min_value=0.0,
    max_value=1.0,
    min_inclusive=True,
    max_inclusive=False,
)
_NU = ParamSpec(
    key="nu",
    label="Degrees of Freedom (nu)",
    default=SF_NU,
    min_value=2.0,
    max_value=250.0,
    min_inclusive=True,
    max_inclusive=True,
)
_BETA_ALPHA = ParamSpec(
    key="alpha",
    label="Beta Shape - Alpha",
    default=SF_BETA_BINOMIAL_ALPHA,
    min_value=0.0,
    max_value=1000.0,
    min_inclusive=False,
    max_inclusive=True,
)
_BETA_BETA = ParamSpec(
    key="beta",
    label="Beta Shape - Beta",
    default=SF_BETA_BINOMIAL_BETA,
    min_value=0.0,
    max_value=1000.0,
    min_inclusive=False,
    max_inclusive=True,
)
_STRESS_PROB = ParamSpec(
    key="stress_prob",
    label="Stress-Regime Probability",
    default=SF_STRESS_REGIME_PROB,
    min_value=0.0,
    max_value=1.0,
    min_inclusive=True,
    max_inclusive=True,
)
_STRESS_MULT = ParamSpec(
    key="stress_pd_multiplier",
    label="Stress-Regime PD Multiplier",
    default=SF_STRESS_REGIME_MULT,
    min_value=1.0,
    max_value=50.0,
    min_inclusive=True,
    max_inclusive=True,
)
_SYSTEMIC_PD = ParamSpec(
    key="systemic_pd",
    label="Systemic Common-Shock Probability",
    # Single source of truth, shared with the recipe that builds the model.
    default=SF_MO_SYSTEMIC_PD,
    min_value=0.0,
    max_value=1.0,
    min_inclusive=True,
    max_inclusive=False,
)


# Read-only view (C.22). The literal is the only reference to the backing dict.
MODEL_CATALOG: Final[Mapping[str, ModelEntry]] = MappingProxyType({
    "forced_default": ModelEntry(
        key="forced_default",
        display_name="Forced Default",
        description="Every obligor defaults in every scenario; loss = EAD x LGD. "
        "A deterministic worst-case bound (PD is ignored).",
        params=(),
    ),
    "independent_bernoulli": ModelEntry(
        key="independent_bernoulli",
        display_name="Independent Bernoulli",
        description="Each obligor defaults independently with probability PD. "
        "No correlation; the diversified baseline.",
        params=(),
    ),
    "stress_regime_bernoulli": ModelEntry(
        key="stress_regime_bernoulli",
        display_name="Stress-Regime Bernoulli",
        description="Independent-Bernoulli defaults with a shared stress regime "
        "that fires with probability stress_prob and multiplies every PD by "
        "stress_pd_multiplier (capped at 1).",
        params=(_STRESS_PROB, _STRESS_MULT),
    ),
    "stress_regime_bernoulli_grouped": ModelEntry(
        key="stress_regime_bernoulli_grouped",
        display_name="Stress-Regime Bernoulli (Grouped Regimes)",
        description="Segmented stress regime: each obligor belongs to a regime that "
        "fires with its own stress_prob and multiplies that group's PDs by its own "
        "stress_pd_multiplier. Regimes are independent across groups (block-diagonal "
        "correlation); parameters are supplied per group via 'groups'.",
        params=(_STRESS_PROB, _STRESS_MULT),
        requires_groups=True,
    ),
    "beta_binomial": ModelEntry(
        key="beta_binomial",
        display_name="Beta-Binomial",
        description="Conditionally-independent defaults driven by a shared per-"
        "scenario PD drawn from Beta(alpha, beta). Mean PD = alpha/(alpha+beta); "
        "the obligor PD column is not used.",
        params=(_BETA_ALPHA, _BETA_BETA),
    ),
    "beta_binomial_grouped": ModelEntry(
        key="beta_binomial_grouped",
        display_name="Beta-Binomial (Grouped Pools)",
        description="Segmented beta-binomial: each obligor belongs to a pool, and "
        "each pool draws its own per-scenario PD from Beta(alpha, beta). Defaults "
        "correlate within a pool and are independent across pools. The obligor PD "
        "column is not used; alpha/beta are supplied per pool via 'groups'.",
        params=(_BETA_ALPHA, _BETA_BETA),
        requires_groups=True,
    ),
    "sf_gaussian": ModelEntry(
        key="sf_gaussian",
        display_name="Single-Factor Gaussian Copula",
        description="One systemic factor with asset correlation rho (Vasicek / "
        "Basel ASRF). The standard desk credit-VaR model. rho is the global fill; "
        "obligors may carry a bespoke per-name rho that overrides it.",
        params=(_RHO,),
        accepts_obligor_rho=True,
    ),
    "sf_gaussian_basel": ModelEntry(
        key="sf_gaussian_basel",
        display_name="Single-Factor Gaussian Copula (Basel rho)",
        description="As Single-Factor Gaussian, but each obligor's asset "
        "correlation is the Basel II/III IRB corporate weighting R(PD) (0.24 at "
        "PD->0 down to 0.12 at PD->1). A per-name rho still overrides where given.",
        params=(),
        accepts_obligor_rho=True,
    ),
    "sf_students_t": ModelEntry(
        key="sf_students_t",
        display_name="Single-Factor Student-t Copula",
        description="Single systemic factor with asset correlation rho plus a "
        "shared chi-square mixing variable at nu degrees of freedom (tail "
        "dependence / fatter joint tails than Gaussian at the same rho). rho is the "
        "global fill; a per-obligor rho overrides it.",
        params=(_RHO, _NU),
        accepts_obligor_rho=True,
    ),
    "sf_students_t_basel": ModelEntry(
        key="sf_students_t_basel",
        display_name="Single-Factor Student-t Copula (Basel rho)",
        description="Single-Factor Student-t with per-obligor asset correlation "
        "from the Basel II/III IRB corporate weighting R(PD), plus a shared "
        "chi-square mixing at nu. A per-name rho still overrides where given.",
        params=(_NU,),
        accepts_obligor_rho=True,
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
    "marshall_olkin": ModelEntry(
        key="marshall_olkin",
        display_name="Marshall-Olkin (Common Shock)",
        description="Each obligor defaults on its own idiosyncratic shock OR a "
        "single shared common shock that fires with probability systemic_pd. A "
        "firing shock defaults the whole cohort at once (deterministic contagion / "
        "the framework's most extreme tail). Requires PD >= systemic_pd per name.",
        params=(_SYSTEMIC_PD,),
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
})

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
