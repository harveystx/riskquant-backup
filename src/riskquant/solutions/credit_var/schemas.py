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

import pandas as pd

__all__ = [
    "ObligorRow",
    "LoadingRow",
    "CreditVarRequest",
    "RiskSummary",
    "ModelTail",
    "RunProvenance",
    "CreditVarResult",
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

    uid: int
    ead: float = Field(ge=0.0, description="Exposure at default (>= 0).")
    pd: float = Field(ge=0.0, le=1.0, description="Probability of default in [0, 1].")
    lgd: float = Field(ge=0.0, le=1.0, description="Loss given default in [0, 1].")


class LoadingRow(BaseModel):
    """One systemic-factor loading for a multi-factor model: obligor `uid` loads
    on systemic `factor` with `loading`. The set of distinct `factor` values
    defines the model's systemic factors; per-obligor idiosyncratic weight is the
    engine residual sqrt(1 - sum loading^2)."""

    model_config = _STRICT

    uid: int
    factor: str = Field(min_length=1)
    loading: float = Field(ge=-1.0, le=1.0)


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
    n_scenarios: int = Field(default=100_000, gt=0)
    alphas: list[float] = Field(default_factory=lambda: [0.95, 0.99, 0.999])
    seed: int = 12345

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
        uids = [o.uid for o in self.obligors]
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
        obligor_uids = {o.uid for o in self.obligors}
        stray = sorted({lo.uid for lo in self.loadings} - obligor_uids)
        if stray:
            raise ValueError(f"loadings reference uid(s) not in obligors: {stray}.")
        uncovered = sorted(obligor_uids - {lo.uid for lo in self.loadings})
        if uncovered:
            raise ValueError(f"obligor(s) have no systemic loadings: {uncovered}.")
        return self

    @classmethod
    def from_dataframe(
        cls,
        obligor_table: pd.DataFrame,
        *,
        model: str,
        loadings_table: pd.DataFrame | None = None,
        n_scenarios: int = 100_000,
        alphas: list[float] | None = None,
        seed: int = 12345,
        **params: float,
    ) -> CreditVarRequest:
        """Build a request from a flat obligor DataFrame (notebook ergonomics).

        The DataFrame needs `uid, ead, pd, lgd`; extra columns are ignored. Model
        params are passed as keywords, e.g.::

            CreditVarRequest.from_dataframe(df, model="sf_gaussian", rho=0.15)

        For multi-factor models, pass `loadings_table` with `uid, factor, loading`::

            CreditVarRequest.from_dataframe(df, model="mf_gaussian", loadings_table=ld)

        Validation is identical to the dict/JSON path — a bad cell still raises a
        `ValidationError` with a `loc` into `obligors` / `loadings`.
        """
        required = ("uid", "ead", "pd", "lgd")
        missing = [c for c in required if c not in obligor_table.columns]
        if missing:
            raise ValueError(
                f"from_dataframe: obligor_table missing column(s) {missing}."
            )
        obligors = [
            ObligorRow(
                uid=int(rec["uid"]),
                ead=float(rec["ead"]),
                pd=float(rec["pd"]),
                lgd=float(rec["lgd"]),
            )
            for rec in obligor_table.loc[:, list(required)].to_dict("records")
        ]
        kwargs: dict[str, object] = {
            "model": model,
            "obligors": obligors,
            "params": dict(params),
            "n_scenarios": n_scenarios,
            "seed": seed,
        }
        if loadings_table is not None:
            lcols = ("uid", "factor", "loading")
            lmissing = [c for c in lcols if c not in loadings_table.columns]
            if lmissing:
                raise ValueError(
                    f"from_dataframe: loadings_table missing column(s) {lmissing}."
                )
            kwargs["loadings"] = [
                LoadingRow(
                    uid=int(rec["uid"]),
                    factor=str(rec["factor"]),
                    loading=float(rec["loading"]),
                )
                for rec in loadings_table.loc[:, list(lcols)].to_dict("records")
            ]
        if alphas is not None:
            kwargs["alphas"] = list(alphas)
        # return cls(**kwargs)
        return cls.model_validate(kwargs)


class RiskSummary(BaseModel):
    """Headline portfolio risk metrics."""

    model_config = ConfigDict(extra="forbid")

    expected_loss: float
    var: dict[str, float]  # confidence (as string) -> VaR
    es: dict[str, float]  # confidence (as string) -> ES
    expected_loss_by_model: dict[str, float]


class ModelTail(BaseModel):
    """Standalone tail risk for one model at the headline confidence level."""

    model_config = ConfigDict(extra="forbid")

    var: float
    es: float


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

    uid: int
    ead: float
    expected_loss: float
    es_contribution: float  # Euler ES contribution; sums to portfolio ES
    es_pct: float  # share of portfolio ES (the contributions sum to 1.0)
    incremental_var: float  # VaR(L) - VaR(L - L_i): VaR drop if obligor removed
    incremental_es: float  # ES(L) - ES(L - L_i): ES drop if obligor removed


class LossHistogram(BaseModel):
    """Binned portfolio loss for a distribution plot."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    bin_edges: list[float]  # length nbins + 1
    counts: list[int]  # length nbins; last bin folds in the > p99.9 overflow


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
    provenance: RunProvenance
