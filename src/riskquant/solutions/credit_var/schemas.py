# src/riskquant/solutions/credit_var/schemas.py
"""
Pydantic schemas for the credit-VaR solution.

The wire contract for Emissary (and any JSON client): a request payload in, a
result payload out, both validated and JSON-native. Pydantic does the field- and
cross-field validation; its `ValidationError.errors()` already carries `loc`
paths (e.g. ``("obligors", 2, "ead")``) that a front-end maps straight to a cell,
so this doubles as the structured, cell-addressable validation layer.

Result contract (WS_CREDIT_VAR_API S1, result schema version 1.0)
-----------------------------------------------------------------
* No result field is a dict keyed by a confidence level. A level is data: each
  `AlphaRisk` carries the requested ``alpha`` float bit for bit, and each
  `PercentileLoss` carries its ``percentile`` float. A display label such as
  ``"99.90%"`` rounds, collides (0.9999 and 0.99991) and saturates ("100.00%"
  for 0.99995), so it is never a key (O-1).
* Every result model is frozen, forbids extra fields and rejects NaN and
  +/-inf in every float, including a float inside a dict or a tuple. A
  non-finite number cannot reach JSON as ``null``.
* Sequence fields are tuples: ``frozen=True`` blocks assignment only, not the
  in-place mutation of a list.
* Ordered sequences are strictly ascending by their key (alpha, percentile,
  obligor_uid) and the models reject any other order, so a lookup such as
  `RiskSummary.at` has exactly one answer.
* The result models enforce the bounds of A_CONTRACT sections 3.2 and 3.8
  (owner decision OD-1 (b)): ``loss_std_dev >= 0``, ``loss_probability`` in
  [0, 1], ``len(bin_edges) == len(counts) + 1``, strictly ascending
  ``bin_edges``, and ``sum(counts) == provenance.n_scenarios``. The engine
  cannot produce a violation; a violation is therefore an engine fault, and
  the bound makes it raise at construction instead of reaching a consumer.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import pandas as pd

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "AlphaRisk",
    "CreditVarRequest",
    "CreditVarResult",
    "GroupRow",
    "LoadingRow",
    "LossHistogram",
    "ModelTail",
    "ObligorContribution",
    "ObligorRow",
    "PercentileLoss",
    "ResultSchemaVersion",
    "RiskSummary",
    "RunProvenance",
    "StressTestObligor",
    "StressTestResult",
]

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from riskquant.solutions.credit_var.catalog import MODEL_CATALOG, model_entry
from riskquant.solutions.credit_var.provenance import ScenarioConstructionId

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


# ===========================================================================
# Result (WS_CREDIT_VAR_API S1, A_CONTRACT section 3)
# ===========================================================================

ResultSchemaVersion = Literal["1.0"]

#: MAJOR.MINOR of the serialized result (P-CREDIT_VAR_API-001 clause 4). MAJOR
#: increments on a breaking change of the serialized form, MINOR on an added
#: optional field. A reader validates only the version it declares.
RESULT_SCHEMA_VERSION: Final[ResultSchemaVersion] = "1.0"

_RESULT = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    protected_namespaces=(),
)


def _require_strictly_ascending(keys: tuple[float, ...], what: str) -> None:
    """Reject an order other than strictly ascending (which also rejects a
    duplicate key). Raised as ``ValueError`` inside a model validator, so
    pydantic reports it as a ``ValidationError``.

    Integer keys (obligor_uid) are passed as ``int``, never cast to ``float``:
    the int64 uid range exceeds 2**53, where two uids could compare equal as
    floats.
    """
    for position in range(1, len(keys)):
        if not keys[position - 1] < keys[position]:
            raise ValueError(
                f"{what} must be strictly ascending: position {position - 1} holds "
                f"{keys[position - 1]!r}, position {position} holds "
                f"{keys[position]!r}."
            )


class AlphaRisk(BaseModel):
    """Tail risk of the portfolio loss at one requested confidence level."""

    model_config = _RESULT

    alpha: float  # the request float, bit for bit (probability, not percent)
    var: float  # VaR at alpha
    es: float  # ES at alpha
    economic_capital_var: float  # var - expected loss
    economic_capital_es: float  # es - expected loss


class PercentileLoss(BaseModel):
    """One point of the portfolio-loss percentile grid.

    `percentile` is in PERCENT units (99.9, not 0.999) so that it stays bitwise
    equal to the grid of earlier releases; `AlphaRisk.alpha` is a probability.
    The two names mark the two units.
    """

    model_config = _RESULT

    percentile: float  # percent units
    loss: float  # np.percentile (linear) of the portfolio loss


class RiskSummary(BaseModel):
    """
    Summary statistics of the simulated portfolio loss distribution.

    All metrics are computed from the simulated loss distribution. Losses are
    assumed to be positive quantities, where larger values correspond to worse
    portfolio outcomes.

    There is no ES/VaR ratio member (O-5): the ratio is undefined at VaR = 0,
    and a consumer that needs it divides `AlphaRisk.es` by `AlphaRisk.var` with
    its own policy for a zero VaR.
    """

    model_config = _RESULT

    # ------------------------------------------------------------------
    # Distribution moments
    # ------------------------------------------------------------------

    loss_mean: float  # mean of portfolio loss (Expected Loss)
    # Standard deviation of portfolio loss. Exactly 0.0 when loss_min ==
    # loss_max (owner decision ZV, applied in run.py).
    loss_std_dev: float = Field(ge=0.0)
    loss_skewness: float  # right-skew of the loss distribution
    loss_excess_kurtosis: float  # excess kurtosis (fat-tailedness)

    # ------------------------------------------------------------------
    # Distribution characteristics
    # ------------------------------------------------------------------

    # Fraction of scenarios with any loss.
    loss_probability: float = Field(ge=0.0, le=1.0)
    loss_min: float  # best simulated scenario loss
    loss_max: float  # worst simulated scenario loss
    coefficient_of_variation: float  # loss_std_dev / loss_mean (0 if loss_mean=0)

    # ------------------------------------------------------------------
    # Tail risk and economic capital: one record per requested alpha,
    # strictly ascending by alpha.
    # ------------------------------------------------------------------

    by_alpha: tuple[AlphaRisk, ...]

    # ------------------------------------------------------------------
    # Attribution: keyed by JtdModel member name (e.g. "FORCED_DEFAULT").
    # ------------------------------------------------------------------

    expected_loss_by_model: dict[str, float]

    @model_validator(mode="after")
    def _by_alpha_strictly_ascending(self) -> RiskSummary:
        _require_strictly_ascending(
            tuple(record.alpha for record in self.by_alpha), "by_alpha alpha"
        )
        return self

    def at(self, alpha: float) -> AlphaRisk:
        """The record whose ``alpha`` equals `alpha` under ``==``.

        No rounding and no nearest match: a level the run did not compute is a
        caller error, not a value to approximate.

        Raises
        ------
        KeyError
            No record has ``record.alpha == alpha``.
        """
        for record in self.by_alpha:
            if record.alpha == alpha:
                return record
        raise KeyError(
            f"confidence level {alpha!r} was not computed in this run; computed "
            f"levels: {[record.alpha for record in self.by_alpha]}."
        )


class ModelTail(BaseModel):
    """Standalone tail risk for one model at the headline confidence level."""

    model_config = _RESULT

    var: float
    es: float


class RunProvenance(BaseModel):
    """Audit stamp: enough to reproduce a number (D-C15, D040).

    Two results are reproducibility-equal when their `model_dump_json` bytes are
    equal after removal of `generated_at`. The comparison is meaningful only
    when `engine_version`, `scenario_construction_id` and the three library
    versions are equal: reproducibility is environment-relative (D-C15).
    """

    model_config = _RESULT

    engine_version: str = Field(pattern=r"^[^+]+\+src\.[0-9a-f]{64}$")
    scenario_construction_id: ScenarioConstructionId
    numpy_version: str
    scipy_version: str
    numba_version: str
    model: str
    params: dict[str, float]
    n_obligors: int
    n_scenarios: int
    seed: int
    alphas: tuple[float, ...]  # request alphas, in request order
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str  # ISO 8601 UTC; outside reproducibility equality


class ObligorContribution(BaseModel):
    """One obligor's share of portfolio risk, at the headline confidence level."""

    model_config = _RESULT

    obligor_uid: int
    ead: float  # the request ead of this obligor
    expected_loss: float
    es_contribution: float  # Euler ES contribution; sums to portfolio ES
    es_pct: float  # share of portfolio ES (the contributions sum to 1.0)
    incremental_var: float  # VaR(L) - VaR(L - L_i): VaR drop if obligor removed
    incremental_es: float  # ES(L) - ES(L - L_i): ES drop if obligor removed
    ul_contribution: float  # Cov(L_i, L) / sigma_L; sums to portfolio UL
    ul_pct: float  # share of portfolio UL (the contributions sum to 1.0)


class LossHistogram(BaseModel):
    """Binned portfolio loss for a distribution plot.

    The model checks the rules that it can see (A_CONTRACT section 3.8):
    ``len(bin_edges) == len(counts) + 1`` and strictly ascending ``bin_edges``.
    The rule ``sum(counts) == n_scenarios`` needs the scenario count, so
    `CreditVarResult` checks it against ``provenance.n_scenarios``.
    """

    model_config = _RESULT

    bin_edges: tuple[float, ...]  # length nbins + 1
    counts: tuple[int, ...]  # length nbins; last bin folds in the > p99.9 overflow

    @model_validator(mode="after")
    def _edges_match_counts(self) -> LossHistogram:
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError(
                f"bin_edges must hold len(counts) + 1 values: len(bin_edges) = "
                f"{len(self.bin_edges)}, len(counts) = {len(self.counts)}."
            )
        _require_strictly_ascending(self.bin_edges, "bin_edges")
        return self


class KernelDistribution(BaseModel):
    """Fitted kernel density estimate for portfolio loss distribution."""

    x_values: list[float]
    density: list[float]
    bandwidth: float


class StressTestObligor(BaseModel):
    """One name in the standalone hard-default stress test."""

    model_config = _RESULT

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

    model_config = _RESULT

    n_obligors: int
    total_loss: float  # sum of EAD * LGD over the flagged names
    obligors: tuple[StressTestObligor, ...]  # strictly ascending obligor_uid

    @model_validator(mode="after")
    def _obligors_strictly_ascending(self) -> StressTestResult:
        _require_strictly_ascending(
            tuple(o.obligor_uid for o in self.obligors), "obligors obligor_uid"
        )
        return self


class CreditVarResult(BaseModel):
    """A completed credit-VaR run, JSON-native (result schema version 1.0)."""

    model_config = _RESULT

    result_schema_version: ResultSchemaVersion
    model: str
    headline_alpha: float
    summary: RiskSummary
    per_model_tail: dict[str, ModelTail]  # keyed by JtdModel member name
    loss_percentiles: tuple[PercentileLoss, ...]  # strictly ascending percentile
    obligor_contributions: tuple[ObligorContribution, ...]  # ascending obligor_uid
    loss_histogram: LossHistogram
    stress_test: StressTestResult | None = None
    provenance: RunProvenance

    @model_validator(mode="after")
    def _sequences_strictly_ascending(self) -> CreditVarResult:
        _require_strictly_ascending(
            tuple(p.percentile for p in self.loss_percentiles),
            "loss_percentiles percentile",
        )
        _require_strictly_ascending(
            tuple(c.obligor_uid for c in self.obligor_contributions),
            "obligor_contributions obligor_uid",
        )
        return self

    @model_validator(mode="after")
    def _histogram_counts_every_scenario(self) -> CreditVarResult:
        """Every scenario lands in exactly one bin (A_CONTRACT section 3.8).

        The last bin folds in the overflow above the display percentile, so a
        count total other than n_scenarios means a dropped or a double-counted
        scenario.
        """
        total = sum(self.loss_histogram.counts)
        if total != self.provenance.n_scenarios:
            raise ValueError(
                f"loss_histogram counts must sum to provenance.n_scenarios: sum = "
                f"{total}, n_scenarios = {self.provenance.n_scenarios}."
            )
        return self
