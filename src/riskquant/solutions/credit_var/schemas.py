# src/riskquant/solutions/credit_var/schemas.py
"""
Pydantic schemas for the credit-VaR solution.

The wire contract for Emissary (and any JSON client): a request payload in, a
result payload out, both validated and JSON-native. Pydantic does the field- and
cross-field validation; its `ValidationError.errors()` already carries `loc`
paths (e.g. ``("obligors", 2, "ead")``) that a front-end maps straight to a cell,
so this doubles as the structured, cell-addressable validation layer.

Dict-keyed-by-confidence fields (`var`, `es`, `loss_percentiles`) use STRING keys
because JSON object keys are strings — e.g. ``{"0.99": 1234.5}`` — so the result
round-trips through JSON without surprise float-key coercion.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = [
    "CreditVarRequest",
    "CreditVarResult",
    "GroupRow",
    "LoadingRow",
    "ModelTail",
    "ObligorRow",
    "RiskSummary",
    "RunProvenance",
]

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from riskquant.solutions.credit_var.catalog import MODEL_CATALOG, model_entry

_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


class ObligorRow(BaseModel):
    """One obligor: the flat desk row."""

    model_config = _STRICT

    obligor_uid: int
    ead: float = Field(ge=0.0, description="Exposure at default (>= 0).")
    pd: float = Field(ge=0.0, le=1.0, description="Probability of default in [0, 1].")
    lgd: float = Field(ge=0.0, le=1.0, description="Loss given default in [0, 1].")
    in_stress_test: bool = Field(
        default=False,
        description="Include in the standalone hard-default stress test (does not "
        "affect the simulation).",
    )
    force_default: bool = Field(
        default=False,
        description="Force this obligor to default inside the simulation; the "
        "selected model runs on the rest. Ignored when the model is forced_default.",
    )
    group: str | None = Field(
        default=None,
        min_length=1,
        description="Pool/regime label for grouped models (beta_binomial_grouped, "
        "stress_regime_bernoulli_grouped): obligors sharing a group share that "
        "group's factor. Omit for non-grouped models.",
    )
    rho: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Bespoke per-obligor asset correlation for single-factor "
        "copulas; overrides the global/regulatory fill. Omit to use the fill.",
    )


class LoadingRow(BaseModel):
    """One systemic-factor loading for a multi-factor model: obligor `obligor_uid` loads
    on systemic `factor` with `loading`. The set of distinct `factor` values
    defines the model's systemic factors; per-obligor idiosyncratic weight is the
    engine residual sqrt(1 - sum loading^2)."""

    model_config = _STRICT

    obligor_uid: int
    factor: str = Field(min_length=1)
    loading: float = Field(ge=-1.0, le=1.0)


class GroupRow(BaseModel):
    """One pool/regime for a grouped model: a named group with its own factor
    parameters. For `beta_binomial_grouped`, `params` carries `alpha`, `beta`;
    for `stress_regime_bernoulli_grouped`, `stress_prob`, `stress_pd_multiplier`.
    Obligors are assigned to a group via `ObligorRow.group`; each group becomes one
    BETA_PD / STRESS_INDICATOR factor, so defaults correlate within a group and are
    independent across groups."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)

_PREVIEW = 10  # bounded offender preview in validation errors

class CreditVarRequest(BaseModel):
    """A single credit-VaR run, as a JSON payload."""

    model_config = _STRICT

    model: str = Field(description="Model key, e.g. 'sf_gaussian'.")
    obligors: list[ObligorRow] = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)
    loadings: list[LoadingRow] | None = Field(
        default=None,
        description="Systemic loadings for multi-factor models; omit otherwise.",
    )
    groups: list[GroupRow] | None = Field(
        default=None,
        description="Per-group factor parameters for grouped models; omit otherwise.",
    )
    n_scenarios: int = Field(default=250_000, gt=0)
    alphas: list[float] = Field(
        default_factory=lambda: [0.9500, 0.9750, 0.9900, 0.9990]
    )
    seed: int = 12345

    @property
    def n_obligors(self) -> int:
        return len(self.obligors)

    @field_validator("alphas")
    @classmethod
    def _alphas_unit_interval(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("at least one confidence level is required.")
        for a in v:
            if not (0.0 < a < 1.0):
                raise ValueError(f"confidence level {a} must be in (0, 1).")
        return v

    @model_validator(mode="after")
    def _validate_model_and_params(self) -> CreditVarRequest:
        if self.model not in MODEL_CATALOG:
            raise ValueError(
                f"unknown model '{self.model}'. Supported: {sorted(MODEL_CATALOG)}."
            )
        entry = model_entry(self.model)
        known = {p.key for p in entry.params}
        unknown = set(self.params) - known
        if unknown:
            raise ValueError(
                f"model '{self.model}' does not accept param(s) {sorted(unknown)}. "
                f"Accepted: {sorted(known)}."
            )
        for p in entry.params:
            if p.key in self.params and not p.contains(self.params[p.key]):
                raise ValueError(
                    f"param '{p.key}' = {self.params[p.key]} out of range "
                    f"{p.interval_repr}."
                )
        return self

    @model_validator(mode="after")
    def _unique_uids(self) -> CreditVarRequest:
        uids = [o.obligor_uid for o in self.obligors]
        if len(uids) != len(set(uids)):
            raise ValueError("duplicate obligor uid(s).")
        return self

    @model_validator(mode="after")
    def _validate_loadings(self) -> CreditVarRequest:
        if self.model not in MODEL_CATALOG:
            return self  # unknown model already reported
        requires = model_entry(self.model).requires_loadings
        if not requires:
            if self.loadings:
                raise ValueError(f"model '{self.model}' does not use a loadings table.")
            return self
        if not self.loadings:
            raise ValueError(
                f"model '{self.model}' requires a non-empty 'loadings' table "
                "(one row per obligor x systemic factor)."
            )
        obligor_uids = {o.obligor_uid for o in self.obligors}
        stray = sorted({lo.obligor_uid for lo in self.loadings} - obligor_uids)
        if stray:
            raise ValueError(f"loadings reference uid(s) not in obligors: {stray}.")
        uncovered = sorted(obligor_uids - {lo.obligor_uid for lo in self.loadings})
        if uncovered:
            raise ValueError(f"obligor(s) have no systemic loadings: {uncovered}.")
        return self

    @model_validator(mode="after")
    def _validate_groups(self) -> CreditVarRequest:
        if self.model not in MODEL_CATALOG:
            return self  # unknown model already reported
        requires = model_entry(self.model).requires_groups
        has_obligor_groups = any(o.group is not None for o in self.obligors)
        if not requires:
            if self.groups:
                raise ValueError(f"model '{self.model}' does not use obligor groups.")
            if has_obligor_groups:
                raise ValueError(
                    f"model '{self.model}' does not use obligor groups; remove the "
                    "'group' column."
                )
            return self
        # Grouped model: per-group params live in 'groups', not request-level params.
        if self.params:
            raise ValueError(
                f"model '{self.model}' takes per-group params (in 'groups'), not "
                "request-level 'params'."
            )
        if not self.groups:
            raise ValueError(
                f"model '{self.model}' requires a non-empty 'groups' table "
                "(one row per pool/regime with its parameters)."
            )
        ungrouped = sorted(o.obligor_uid for o in self.obligors if o.group is None)
        if ungrouped:
            raise ValueError(f"obligor(s) have no group: {ungrouped}.")
        names = [g.name for g in self.groups]
        if len(names) != len(set(names)):
            raise ValueError("duplicate group name(s) in 'groups'.")
        defined = set(names)
        used = {o.group for o in self.obligors}
        undefined = used - defined
        if undefined:
            raise ValueError(f"obligor group(s) not defined in 'groups': {undefined}.")
        # Validate each group's params against the model's per-group spec.
        entry = model_entry(self.model)
        spec_keys = {p.key for p in entry.params}
        for g in self.groups:
            unknown = set(g.params) - spec_keys
            if unknown:
                raise ValueError(
                    f"group '{g.name}' has unknown param(s) {sorted(unknown)}. "
                    f"Accepted: {sorted(spec_keys)}."
                )
            for p in entry.params:
                if p.key not in g.params:
                    raise ValueError(
                        f"group '{g.name}' is missing required param '{p.key}'."
                    )
                if not p.contains(g.params[p.key]):
                    raise ValueError(
                        f"group '{g.name}' param '{p.key}' = {g.params[p.key]} out of "
                        f"range {p.interval_repr}."
                    )
        return self

    @model_validator(mode="after")
    def _validate_marshall_olkin_systemic_pd(self) -> CreditVarRequest:
        """Reject a common-shock probability no obligor can carry (P2-07).

        Marshall-Olkin decouples each obligor's total PD into an idiosyncratic
        and a systemic leg::

            p_idio_i = 1 - (1 - PD_i) / (1 - systemic_pd)

        which is a probability only when ``PD_i >= systemic_pd``. Equality is
        legal: it means the obligor defaults through the shared shock alone.
        Below it the decoupling is negative and the model is undefined.

        The builder enforces the same inequality and keeps doing so — this is
        the earliest boundary at which the request already holds BOTH the
        obligor PDs and the shock probability, so the failure lands on the
        request the author wrote rather than after factor-universe and master
        construction. Scoped to the Marshall-Olkin key: no other model carries
        this constraint.
        """
        if self.model != "marshall_olkin":
            return self

        spec = next(
            (p for p in model_entry(self.model).params if p.key == "systemic_pd"),
            None,
        )
        if spec is None:  # pragma: no cover - catalog would have to change
            return self
        systemic_pd = float(self.params.get("systemic_pd", spec.default))

        offenders = [o for o in self.obligors if o.pd < systemic_pd]
        if offenders:
            preview = ", ".join(
                f"uid {o.obligor_uid} (PD={o.pd})" for o in offenders[:_PREVIEW]
            )
            raise ValueError(
                f"model 'marshall_olkin': {len(offenders)} obligor(s) have a total "
                f"PD below systemic_pd={systemic_pd}: {preview}. The common shock "
                "cannot be more likely than an obligor's own default: the "
                "decoupling 1 - (1 - PD) / (1 - systemic_pd) would be negative. "
                "Lower systemic_pd or remove the obligor(s)."
            )
        return self

    @model_validator(mode="after")
    def _validate_obligor_rho(self) -> CreditVarRequest:
        if self.model not in MODEL_CATALOG:
            return self
        if not model_entry(self.model).accepts_obligor_rho and any(
            o.rho is not None for o in self.obligors
        ):
            raise ValueError(f"model '{self.model}' does not use a per-obligor 'rho'.")
        return self

    @classmethod
    def from_dataframe(
        cls,
        obligor_table: pd.DataFrame,
        *,
        model: str,
        loadings_table: pd.DataFrame | None = None,
        group_params: dict[str, dict[str, float]] | None = None,
        n_scenarios: int = 100_000,
        alphas: list[float] | None = None,
        seed: int = 12345,
        **params: float,
    ) -> CreditVarRequest:
        """Build a request from a flat obligor DataFrame (notebook ergonomics).

        The DataFrame needs `uid, ead, pd, lgd`; extra columns are ignored except
        an optional `group` column (used by grouped models). Model params are
        passed as keywords, e.g.::

            CreditVarRequest.from_dataframe(df, model="sf_gaussian", rho=0.20)

        For multi-factor models, pass `loadings_table` with `uid, factor, loading`::

            CreditVarRequest.from_dataframe(df, model="mf_gaussian", loadings_table=ld)

        For grouped models, put a `group` column on `df` and pass per-group params::

            CreditVarRequest.from_dataframe(
                df, model="beta_binomial_grouped",
                group_params={"IG": {"alpha": 1.0, "beta": 80.0},
                              "HY": {"alpha": 2.0, "beta": 18.0}})

        Validation is identical to the dict/JSON path — a bad cell still raises a
        `ValidationError` with a `loc` into `obligors` / `loadings` / `groups`.
        """
        required = ("obligor_uid", "ead", "pd", "lgd")
        missing = [c for c in required if c not in obligor_table.columns]
        if missing:
            raise ValueError(
                f"from_dataframe: obligor_table missing column(s) {missing}."
            )
        has_group = "group" in obligor_table.columns
        has_rho = "rho" in obligor_table.columns
        read_cols = (
            list(required)
            + (["group"] if has_group else [])
            + (["rho"] if has_rho else [])
        )
        obligors = [
            ObligorRow(
                obligor_uid=int(rec["obligor_uid"]),
                ead=float(rec["ead"]),
                pd=float(rec["pd"]),
                lgd=float(rec["lgd"]),
                group=(
                    str(rec["group"]) if has_group and pd.notna(rec["group"]) else None
                ),
                rho=(float(rec["rho"]) if has_rho and pd.notna(rec["rho"]) else None),
            )
            for rec in obligor_table.loc[:, read_cols].to_dict("records")
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "obligors": obligors,
            "params": dict(params),
            "n_scenarios": n_scenarios,
            "seed": seed,
        }
        if group_params is not None:
            kwargs["groups"] = [
                GroupRow(name=str(name), params={k: float(v) for k, v in gp.items()})
                for name, gp in group_params.items()
            ]
        if loadings_table is not None:
            lcols = ("obligor_uid", "factor", "loading")
            lmissing = [c for c in lcols if c not in loadings_table.columns]
            if lmissing:
                raise ValueError(
                    f"from_dataframe: loadings_table missing column(s) {lmissing}."
                )
            kwargs["loadings"] = [
                LoadingRow(
                    obligor_uid=int(rec["obligor_uid"]),
                    factor=str(rec["factor"]),
                    loading=float(rec["loading"]),
                )
                for rec in loadings_table.loc[:, list(lcols)].to_dict("records")
            ]
        if alphas is not None:
            kwargs["alphas"] = list(alphas)
        return cls(**kwargs)


class RiskSummary(BaseModel):
    """
    Summary statistics of the simulated portfolio loss distribution.

    All metrics are computed from the simulated loss distribution. Losses are
    assumed to be positive quantities, where larger values correspond to worse
    portfolio outcomes.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # Distribution moments
    # ------------------------------------------------------------------

    loss_mean: float  # mean of portfolio loss (Expected Loss)
    loss_std_dev: float  # standard deviation of portfolio loss
    loss_skewness: float  # right-skew of the loss distribution
    loss_excess_kurtosis: float  # excess kurtosis (fat-tailedness)

    # ------------------------------------------------------------------
    # Distribution characteristics
    # ------------------------------------------------------------------

    loss_probability: float  # fraction of scenarios with any loss
    loss_min: float  # best simulated scenario loss
    loss_max: float  # worst simulated scenario loss
    coefficient_of_variation: float  # loss_std_dev / loss_mean (0 if loss_mean=0)

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    var: dict[str, float]
    es: dict[str, float]

    # ------------------------------------------------------------------
    # Economic capital
    # ------------------------------------------------------------------

    economic_capital_var: dict[str, float]
    economic_capital_es: dict[str, float]

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    expected_loss_by_model: dict[str, float]

    @property
    def es_to_var_ratio(self) -> dict[str, float]:
        """The ratio of ES to VaR at each confidence level."""
        return {
            alpha: self.es.get(alpha, float("nan")) / self.var.get(alpha, float("nan"))
            for alpha in self.var
        }

    @property
    def max_alpha(self) -> float:
        """The highest confidence level requested in the run."""
        return max(float(a) for a in self.var)

    def get_var(self, alpha: float) -> float:
        """Get the Value-at-Risk at a given confidence level."""
        alpha_pct = round(alpha * 100, 2)
        alpha_pct = f"{alpha_pct:.2f}%"
        if alpha_pct not in self.var:
            raise ValueError(
                f"confidence level '{alpha_pct}' not in run; available: {sorted(self.var)}"
            )
        return self.var.get(alpha_pct, float("nan"))

    def get_es(self, alpha: float) -> float:
        """Get the Expected Shortfall at a given confidence level."""
        alpha_pct = round(alpha * 100, 2)
        alpha_pct = f"{alpha_pct:.2f}%"
        if alpha_pct not in self.es:
            raise ValueError(
                f"confidence level '{alpha_pct}' not in run; available: {sorted(self.es)}"
            )
        return self.es.get(alpha_pct, float("nan"))

    def get_es_to_var_ratio(self, alpha: float) -> float:
        """Get the ES/VaR ratio at a given confidence level."""
        alpha_pct = round(alpha * 100, 2)
        alpha_pct = f"{alpha_pct:.2f}%"
        es_to_var_map = self.es_to_var_ratio
        if alpha_pct not in es_to_var_map:
            raise ValueError(
                f"confidence level '{alpha_pct}' not in run; available: {sorted(es_to_var_map)}"
            )
        return es_to_var_map.get(alpha_pct, float("nan"))


class ModelTail(BaseModel):
    """Standalone tail risk for one model at the headline confidence level."""

    model_config = ConfigDict(extra="forbid")

    var: float
    es: float

    @property
    def es_to_var_ratio(self) -> float:
        return self.es / self.var if self.var != 0 else float("nan")


class RunProvenance(BaseModel):
    """Audit stamp: enough to reproduce a number."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str
    model: str
    params: dict[str, float]
    n_obligors: int
    n_scenarios: int
    seed: int
    alphas: list[float]
    input_hash: str
    generated_at: str


class ObligorContribution(BaseModel):
    """One obligor's share of portfolio risk, at the headline confidence level."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    obligor_uid: int
    ead: float
    expected_loss: float
    es_contribution: float  # Euler ES contribution; sums to portfolio ES
    es_pct: float  # share of portfolio ES (the contributions sum to 1.0)
    incremental_var: float  # VaR(L) - VaR(L - L_i): VaR drop if obligor removed
    incremental_es: float  # ES(L) - ES(L - L_i): ES drop if obligor removed
    ul_contribution: float  # Cov(L_i, L) / sigma_L; sums to portfolio UL
    ul_pct: float  # share of portfolio UL (the contributions sum to 1.0)


class LossHistogram(BaseModel):
    """Binned portfolio loss for a distribution plot."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    bin_edges: list[float]  # length nbins + 1
    counts: list[int]  # length nbins; last bin folds in the > p99.9 overflow


class KernelDistribution(BaseModel):
    """Fitted kernel density estimate for portfolio loss distribution."""

    x_values: list[float]
    density: list[float]
    bandwidth: float


class StressTestObligor(BaseModel):
    """One name in the standalone hard-default stress test."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    obligor_uid: int
    ead: float
    lgd: float
    loss: float  # EAD * LGD — the certain loss if this name defaults


class StressTestResult(BaseModel):
    """Deterministic 'what if these names all default' result.

    Independent of the chosen model and of the simulation — the answer to the
    flagged obligors (`in_stress_test`) defaulting together. Present only when at
    least one obligor is flagged.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    n_obligors: int
    total_loss: float  # sum of EAD * LGD over the flagged names
    obligors: list[StressTestObligor]


class CreditVarResult(BaseModel):
    """A completed credit-VaR run, JSON-native."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str
    headline_alpha: float
    summary: RiskSummary
    per_model_tail: dict[str, ModelTail]
    loss_percentiles: dict[str, float]  # percentile (as string) -> loss
    obligor_contributions: list[ObligorContribution]
    loss_histogram: LossHistogram
    stress_test: StressTestResult | None = None
    provenance: RunProvenance
