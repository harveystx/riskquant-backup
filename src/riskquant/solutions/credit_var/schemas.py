# src/riskquant/solutions/credit_var/schemas.py
"""
Pydantic schemas for the credit-VaR solution.

The wire contract for Emissary (and any JSON client): a request payload in, a
result payload out, both validated and JSON-native. Pydantic does the field- and
cross-field validation; its `ValidationError.errors()` carries `loc` paths
(e.g. ``("obligors", 2, "ead")``) that a front-end maps straight to a cell, so
this doubles as the structured, cell-addressable validation layer.

Request contract (WS_CREDIT_VAR_API S2, A_CONTRACT sections 1 and 2)
--------------------------------------------------------------------
* Every request model is frozen, forbids extra fields and rejects NaN and
  +/-inf in every float, including a float inside a dict or a tuple.
* Coercion is strict (O-8). An int field accepts an int only (not a bool, a
  float, a str or a numpy integer). A float field accepts a float or an int
  and stores a float. A bool field accepts a bool only. A str field accepts a
  str only. The rules apply to dict keys, dict values and sequence elements.
  Silent coercion would let ``"100"`` or ``7.0`` reach the engine as a value
  that the author did not write.
* Every float field (``ead``, ``pd``, ``lgd``, ``rho``, ``loading``, the
  values of ``GroupRow.params`` and ``CreditVarRequest.params``, and each
  ``alphas`` element) admits only a float (a subclass included, stored as
  float) or an int (not a bool) (WS_CREDIT_VAR_API S7, R-003). Strict mode of
  pydantic alone also admits every object with ``__float__``: Decimal,
  Fraction, numpy.float32 and numpy integers. Those types round or change
  type silently on the way in, so `_float_or_int` rejects them first.
* A sequence field (``obligors``, ``loadings``, ``groups``, ``alphas``)
  accepts a list or a tuple and stores a tuple. A set, a generator or a str
  raises. ``frozen=True`` blocks assignment only, so the tuple is what stops
  in-place mutation of a sequence. Dict fields stay dicts; the entry points
  validate the request state again at call time (C.15).
* Every request-content rule that the engine enforces is also a request rule,
  so a request that validates does not fail in the engine for its content
  (C.17): seed >= 0, obligor_uid in the int64 range, no duplicate
  (obligor_uid, factor) loading, a per-obligor sum of loading^2 <= 1 + 1e-9,
  and no loadings factor name in the reserved set of the multi-factor recipes.
* ``LoadingRow.factor``, ``ObligorRow.group`` and ``GroupRow.name`` contain
  no NUL character (OD-S2-2 (a)). The fault loc is the field path (rule 1).
* There is no size limit other than the field constraints (O-12).
* Each rule raises ``pydantic.ValidationError`` with exactly one entry at the
  loc of A_CONTRACT section 2.5. A cross-row rule addresses the offending row
  and field, not the request root: `_request_fault` builds the error with its
  own loc, which pydantic passes through from a model validator (F-35).

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
* Ordered sequences are strictly ascending by their key and the models reject
  any other order, so a lookup such as `RiskSummary.at` has exactly one
  answer: ``summary.by_alpha`` by alpha, ``loss_percentiles`` by percentile,
  ``obligor_contributions`` and ``stress_test.obligors`` by obligor_uid
  (owner decision OD-2).

Enforced rules (A_CONTRACT sections 3.2 and 3.8, owner decision OD-1 (b))
-------------------------------------------------------------------------
Validation raises ``pydantic.ValidationError`` for each of these rules:

* `RiskSummary`: ``loss_std_dev >= 0``; ``0 <= loss_probability <= 1``.
* `RiskSummary`, zero variance (owner decision ZV): when ``loss_min ==
  loss_max``, ``loss_std_dev``, ``loss_skewness`` and
  ``loss_excess_kurtosis`` each equal 0.0 under ``==``.
* `RiskSummary`: each ``expected_loss_by_model`` key is a `JtdModel` member
  name (C.7).
* `CreditVarResult`: each ``per_model_tail`` key is a `JtdModel` member name
  (C.7).
* `LossHistogram`: ``len(bin_edges) == len(counts) + 1``; ``bin_edges``
  strictly ascending.
* `CreditVarResult`: ``sum(loss_histogram.counts) == provenance.n_scenarios``.
* `CreditVarResult`: the set of ``summary.by_alpha`` alpha values equals the
  set of ``provenance.alphas``.

The engine cannot produce a violation. A violation is therefore an engine
fault, and the rule makes it raise at construction instead of reaching a
consumer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Final, Literal

import pandas as pd

__all__ = [
    "DEFAULT_N_SCENARIOS",
    "LOADING_SUM_SQ_TOLERANCE",
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
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError, PydanticKnownError

from riskquant.credit.jtd.models import VALID_JTD_MODEL_NAMES
from riskquant.solutions.credit_var.catalog import MODEL_CATALOG, model_entry
from riskquant.solutions.credit_var.provenance import ScenarioConstructionId
from riskquant.solutions.credit_var.recipes import (
    CHI_FACTOR_NAME,
    idiosyncratic_factor_name,
)

# ===========================================================================
# Request (WS_CREDIT_VAR_API S2, A_CONTRACT section 2)
# ===========================================================================

_REQUEST = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    protected_namespaces=(),
    strict=True,
)

_PREVIEW = 10  # bounded offender preview in validation errors

#: int64 range of `obligor_uid`. The engine holds uids in int64 arrays; a uid
#: outside the range would wrap silently or raise OverflowError there (F-12).
_INT64_MIN: Final[int] = -(2**63)
_INT64_MAX: Final[int] = 2**63 - 1

#: Tolerance of the per-obligor rule sum(loading^2) <= 1 + tolerance. It is the
#: tolerance of the engine check `credit.jtd.ingest.validate._semantic_mf_gaussian`.
#: `CreditVarRequest._check_loading_sum_sq` also uses the arithmetic of that check,
#: so the request rule and the engine rule accept the same loadings.
LOADING_SUM_SQ_TOLERANCE: Final[float] = 1e-9

#: Default scenario count of a request. `CreditVarRequest.from_dataframe` uses
#: the same name, so the two defaults cannot differ (C.21).
DEFAULT_N_SCENARIOS: Final[int] = 250_000

#: Multi-factor recipes and whether each adds the shared chi-square factor. The
#: recipe adds one idiosyncratic factor per obligor to the universe beside the
#: loadings factors, so a loadings factor with one of these names collides in
#: the factor universe (F-36, A_CONTRACT section 2.5 rule 15).
_MF_RECIPE_HAS_CHI: Final[Mapping[str, bool]] = {
    "mf_gaussian": False,
    "mf_students_t": True,
}


def _list_or_tuple(value: object) -> object:
    """Admit a list or a tuple as a tuple (O-8, C.18).

    ``None`` passes through: the field type then decides (an optional field
    accepts it, a required field rejects it). Every other type raises, also a
    set (no defined order), a generator (consumed once) and a str (a sequence
    of characters, never an intended sequence of rows).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise PydanticCustomError(
        "list_or_tuple_type",
        "Input should be a list or a tuple, not {input_type}",
        {"input_type": type(value).__name__},
    )


def _non_empty(value: tuple[Any, ...]) -> tuple[Any, ...]:
    """Reject an empty sequence.

    An after-validator and not ``Field(min_length=1)``: pydantic adds a second
    ``too_short`` entry to an element fault when the only element fails, which
    breaks the one-entry-per-fault rule of A_CONTRACT section 2.5. An
    after-validator runs only when every element is valid.
    """
    if not value:
        raise PydanticCustomError(
            "too_short", "Input should have at least 1 item, not 0"
        )
    return value


_AsTuple = BeforeValidator(_list_or_tuple)
_NonEmpty = AfterValidator(_non_empty)
#: JSON-schema statement of the non-empty rule that `_NonEmpty` enforces.
_NON_EMPTY_SCHEMA: Final[dict[str, Any]] = {"minItems": 1}


def _float_or_int(value: object) -> object:
    """Admit a float or an int (not a bool) for a float field (C.18, R-003).

    A float subclass (for example numpy.float64) is stored as a plain float,
    which keeps its value and the sign of zero. An int and ``None`` pass
    through: the field type then stores the int as a float, and an optional
    field accepts ``None``. Every other type raises one ``float_type`` entry
    at the field path, the entry that strict mode gives for a bool. The check
    is a before-validator, so ``model_json_schema()`` does not change.
    """
    if isinstance(value, float):
        return value if type(value) is float else float(value)
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise PydanticKnownError("float_type")


#: A float field of the request: `_float_or_int` runs before the strict float
#: validator. An optional field puts the validator outside the union
#: (``Annotated[float | None, _FloatOrInt]``), so a rejected input gives one
#: entry and not one entry per union member.
_FloatOrInt = BeforeValidator(_float_or_int)
_Float = Annotated[float, _FloatOrInt]

#: The validator comes after the Field: a BeforeValidator wraps the schema to its
#: left, so it still runs first. In the other order the constraints leave the
#: float schema and ``model_json_schema()`` writes "gt"/"lt" in place of
#: "exclusiveMinimum"/"exclusiveMaximum" (C.29 golden file).
_Alpha = Annotated[float, Field(gt=0.0, lt=1.0), _FloatOrInt]

#: A str field that contains no NUL character (OD-S2-2 (a), A_CONTRACT sections
#: 2.1 to 2.3). A NUL in a label is not data that an author writes; it reaches
#: the engine's string arrays and frames as an invisible character. A pattern,
#: not a validator function: the JSON schema states the rule, and pydantic
#: reports one ``string_pattern_mismatch`` entry at the field path (rule 1).
_NO_NUL_PATTERN: Final[str] = r"^[^\x00]*$"
_ObligorUid = Annotated[int, Field(ge=_INT64_MIN, le=_INT64_MAX)]

_Loc = tuple[str | int, ...]


def _request_fault(loc: _Loc, reason: str, value: object) -> ValidationError:
    """A request error with one entry at `loc` (A_CONTRACT section 2.5).

    Raised inside a model validator, a ``ValidationError`` reaches the caller
    with its own loc; a ``ValueError`` would get the loc ``()`` (F-35). The
    reason goes through the ``{reason}`` template slot, so braces in it are
    text, not template fields.
    """
    return ValidationError.from_exception_data(
        title="CreditVarRequest",
        line_errors=[
            InitErrorDetails(
                type=PydanticCustomError(
                    "credit_var_request", "{reason}", {"reason": reason}
                ),
                loc=loc,
                input=value,
            )
        ],
    )


def _second_occurrence(values: Sequence[object]) -> int | None:
    """Index of the first value that occurs at a lower index, else ``None``."""
    seen: set[object] = set()
    for index, value in enumerate(values):
        if value in seen:
            return index
        seen.add(value)
    return None


def _dataframe_uid(value: Any) -> Any:
    """An ``obligor_uid`` cell of a `from_dataframe` table, cast to int.

    A float column holds an integral uid as, for example, ``7.0``; the cast
    keeps it. A float that is not integral (``7.9``, NaN, +/-inf) is returned
    unchanged, so the strict int field raises R. ``int(7.9)`` would give 7: a
    uid that the author did not write. The types are ``Any`` because a cell
    has no static type and the non-integral case returns the cell unchanged.
    """
    if isinstance(value, float) and not value.is_integer():
        return value
    return int(value)


class ObligorRow(BaseModel):
    """One obligor: the flat desk row."""

    model_config = _REQUEST

    obligor_uid: _ObligorUid
    ead: _Float = Field(ge=0.0, description="Exposure at default (finite, >= 0).")
    pd: _Float = Field(ge=0.0, le=1.0, description="Probability of default in [0, 1].")
    lgd: _Float = Field(ge=0.0, le=1.0, description="Loss given default in [0, 1].")
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
        pattern=_NO_NUL_PATTERN,
        description="Pool/regime label for grouped models (beta_binomial_grouped, "
        "stress_regime_bernoulli_grouped): obligors sharing a group share that "
        "group's factor. Omit for non-grouped models.",
    )
    rho: Annotated[float | None, _FloatOrInt] = Field(
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

    model_config = _REQUEST

    obligor_uid: _ObligorUid
    factor: str = Field(min_length=1, pattern=_NO_NUL_PATTERN)
    loading: _Float = Field(ge=-1.0, le=1.0)


class GroupRow(BaseModel):
    """One pool/regime for a grouped model: a named group with its own factor
    parameters. For `beta_binomial_grouped`, `params` carries `alpha`, `beta`;
    for `stress_regime_bernoulli_grouped`, `stress_prob`, `stress_pd_multiplier`.
    Obligors are assigned to a group via `ObligorRow.group`; each group becomes one
    BETA_PD / STRESS_INDICATOR factor, so defaults correlate within a group and are
    independent across groups."""

    model_config = _REQUEST

    name: str = Field(min_length=1, pattern=_NO_NUL_PATTERN)
    params: dict[str, _Float] = Field(default_factory=dict)


class CreditVarRequest(BaseModel):
    """A single credit-VaR run, as a JSON payload.

    The request keeps the caller's alpha order; the result sorts (C.3). The
    cross-field rules run in the order of A_CONTRACT section 2.5, and the first
    fault raises. A request with one fault therefore gives one error entry.
    """

    model_config = _REQUEST

    model: str = Field(description="Model key, e.g. 'sf_gaussian'.")
    obligors: Annotated[tuple[ObligorRow, ...], _AsTuple, _NonEmpty] = Field(
        json_schema_extra=_NON_EMPTY_SCHEMA
    )
    params: dict[str, _Float] = Field(default_factory=dict)
    loadings: Annotated[tuple[LoadingRow, ...] | None, _AsTuple] = Field(
        default=None,
        description="Systemic loadings for multi-factor models; omit otherwise.",
    )
    groups: Annotated[tuple[GroupRow, ...] | None, _AsTuple] = Field(
        default=None,
        description="Per-group factor parameters for grouped models; omit otherwise.",
    )
    n_scenarios: int = Field(default=DEFAULT_N_SCENARIOS, gt=0)
    alphas: Annotated[tuple[_Alpha, ...], _AsTuple, _NonEmpty] = Field(
        default=(0.9500, 0.9750, 0.9900, 0.9990),
        json_schema_extra=_NON_EMPTY_SCHEMA,
    )
    seed: int = Field(default=12345, ge=0)

    @property
    def n_obligors(self) -> int:
        return len(self.obligors)

    @model_validator(mode="after")
    def _admission_rules(self) -> CreditVarRequest:
        """Cross-field rules of A_CONTRACT section 2.5, rules 4 to 25, in order.

        Field rules (1 to 3) ran before this validator; it runs only when every
        field is valid. One validator, not several: pydantic stops at the first
        validator that raises, and a fixed rule order makes the reported fault
        deterministic.
        """
        self._check_alphas_distinct()
        self._check_model_and_params()
        self._check_unique_uids()
        self._check_loadings()
        self._check_groups()
        self._check_marshall_olkin_systemic_pd()
        self._check_obligor_rho()
        return self

    def _check_alphas_distinct(self) -> None:
        """Rule 4. The result holds one record per alpha, and `RiskSummary.at`
        looks a level up under ``==``; two equal alphas would give two records
        for one level (O-1). Distinct floats, even 1 ulp apart, are distinct
        levels."""
        index = _second_occurrence(self.alphas)
        if index is not None:
            raise _request_fault(
                ("alphas", index),
                f"duplicate confidence level {self.alphas[index]!r}; each alpha "
                "must be distinct under ==.",
                self.alphas[index],
            )

    def _check_model_and_params(self) -> None:
        """Rules 5, 6 and 7."""
        if self.model not in MODEL_CATALOG:
            raise _request_fault(
                ("model",),
                f"unknown model '{self.model}'. Supported: {sorted(MODEL_CATALOG)}.",
                self.model,
            )
        entry = model_entry(self.model)
        known = {p.key for p in entry.params}
        for key in self.params:
            if key not in known:
                raise _request_fault(
                    ("params", key),
                    f"model '{self.model}' does not accept param '{key}'. "
                    f"Accepted: {sorted(known)}.",
                    self.params[key],
                )
        for spec in entry.params:
            if spec.key in self.params and not spec.contains(self.params[spec.key]):
                raise _request_fault(
                    ("params", spec.key),
                    f"param '{spec.key}' = {self.params[spec.key]} out of range "
                    f"{spec.interval_repr}.",
                    self.params[spec.key],
                )

    def _check_unique_uids(self) -> None:
        """Rule 8."""
        index = _second_occurrence([o.obligor_uid for o in self.obligors])
        if index is not None:
            uid = self.obligors[index].obligor_uid
            raise _request_fault(
                ("obligors", index, "obligor_uid"),
                f"duplicate obligor uid {uid}.",
                uid,
            )

    def _check_loadings(self) -> None:
        """Rules 9 to 15."""
        if not model_entry(self.model).requires_loadings:
            if self.loadings:
                raise _request_fault(
                    ("loadings",),
                    f"model '{self.model}' does not use a loadings table.",
                    len(self.loadings),
                )
            return
        if not self.loadings:
            raise _request_fault(
                ("loadings",),
                f"model '{self.model}' requires a non-empty 'loadings' table "
                "(one row per obligor x systemic factor).",
                self.loadings,
            )
        loadings = self.loadings
        obligor_uids = {o.obligor_uid for o in self.obligors}
        for j, row in enumerate(loadings):
            if row.obligor_uid not in obligor_uids:
                raise _request_fault(
                    ("loadings", j, "obligor_uid"),
                    f"loadings reference uid {row.obligor_uid}, which is not in "
                    "obligors.",
                    row.obligor_uid,
                )
        loaded_uids = {row.obligor_uid for row in loadings}
        for i, obligor in enumerate(self.obligors):
            if obligor.obligor_uid not in loaded_uids:
                raise _request_fault(
                    ("obligors", i, "obligor_uid"),
                    f"obligor {obligor.obligor_uid} has no systemic loadings.",
                    obligor.obligor_uid,
                )
        index = _second_occurrence([(row.obligor_uid, row.factor) for row in loadings])
        if index is not None:
            row = loadings[index]
            raise _request_fault(
                ("loadings", index, "factor"),
                f"obligor {row.obligor_uid} loads on systemic factor "
                f"'{row.factor}' twice.",
                row.factor,
            )
        self._check_loading_sum_sq(loadings)
        self._check_reserved_factor_names(loadings)

    @staticmethod
    def _check_loading_sum_sq(loadings: tuple[LoadingRow, ...]) -> None:
        """Rule 14: per obligor, sum(loading^2) <= 1 + LOADING_SUM_SQ_TOLERANCE.

        The idiosyncratic weight is sqrt(1 - sum loading^2); above 1 there is
        no real residual.

        The decision uses the arithmetic of the engine check
        `credit.jtd.ingest.validate._semantic_mf_gaussian`, not a more accurate
        sum. Near the bound, two summations of the same squares can round to
        adjacent floats on the two sides of 1 + 1e-9 (``math.fsum`` and the
        engine sum differ in both directions), so only the same arithmetic
        gives the same decision. The engine arithmetic is:

        * each square is the float64 ``x ** 2``, equal to ``x * x``;
        * the sum is ``pandas.Series.groupby(uid).sum()``, which pandas
          evaluates as a Kahan compensated sum over the rows of each group,
          in row order, from ``sum = compensation = 0.0``;
        * the rows arrive in request order: `run._loadings_frame` and
          `recipes._mf_universe_and_bundle` keep the order and only drop the
          rows of other obligors, which does not change a group's sum.

        One pass over the rows, linear in the row count: per obligor, the
        Kahan state and the first row at which the running sum exceeds the
        bound. The decision uses the final sum of each obligor. The loc is,
        over the rejected obligors, the lowest such row. A rejected obligor
        always has such a row, because its last row holds the final sum.
        """
        bound = 1.0 + LOADING_SUM_SQ_TOLERANCE
        # uid -> [running sum, compensation]: the state of pandas group_sum.
        state: dict[int, list[float]] = {}
        first_over: dict[int, int] = {}
        for j, row in enumerate(loadings):
            uid = row.obligor_uid
            kahan = state.get(uid)
            if kahan is None:
                kahan = state[uid] = [0.0, 0.0]
            total, compensation = kahan
            y = row.loading * row.loading - compensation
            t = total + y
            kahan[1] = (t - total) - y
            kahan[0] = t
            if t > bound and uid not in first_over:
                first_over[uid] = j
        rejected = [j for uid, j in first_over.items() if state[uid][0] > bound]
        if rejected:
            j = min(rejected)
            row = loadings[j]
            raise _request_fault(
                ("loadings", j, "loading"),
                f"obligor {row.obligor_uid}: sum of squared loadings "
                f"{state[row.obligor_uid][0]!r} exceeds 1 + "
                f"{LOADING_SUM_SQ_TOLERANCE!r}; the running sum first exceeds "
                "the bound at this row. The idiosyncratic weight sqrt(1 - sum) "
                "is undefined.",
                row.loading,
            )

    def _check_reserved_factor_names(self, loadings: tuple[LoadingRow, ...]) -> None:
        """Rule 15: no loadings factor takes a name that the recipe adds.

        The reserved set is the recipe's own names: one idiosyncratic factor
        per request obligor_uid (`recipes.idiosyncratic_factor_name`) and, for
        mf_students_t, the shared chi-square factor (`recipes.CHI_FACTOR_NAME`).
        """
        has_chi = _MF_RECIPE_HAS_CHI.get(self.model)
        if has_chi is None:
            return
        reserved = {idiosyncratic_factor_name(o.obligor_uid) for o in self.obligors}
        if has_chi:
            reserved.add(CHI_FACTOR_NAME)
        for j, row in enumerate(loadings):
            if row.factor in reserved:
                raise _request_fault(
                    ("loadings", j, "factor"),
                    f"factor name '{row.factor}' is reserved for model "
                    f"'{self.model}': the recipe adds a factor of that name to "
                    "the factor universe.",
                    row.factor,
                )

    def _check_groups(self) -> None:
        """Rules 16 to 23."""
        entry = model_entry(self.model)
        if not entry.requires_groups:
            if self.groups:
                raise _request_fault(
                    ("groups",),
                    f"model '{self.model}' does not use obligor groups.",
                    len(self.groups),
                )
            for i, obligor in enumerate(self.obligors):
                if obligor.group is not None:
                    raise _request_fault(
                        ("obligors", i, "group"),
                        f"model '{self.model}' does not use obligor groups; "
                        "remove the 'group' column.",
                        obligor.group,
                    )
            return
        # Grouped model: per-group params live in 'groups', not request-level params.
        if self.params:
            raise _request_fault(
                ("params",),
                f"model '{self.model}' takes per-group params (in 'groups'), not "
                "request-level 'params'.",
                dict(self.params),
            )
        if not self.groups:
            raise _request_fault(
                ("groups",),
                f"model '{self.model}' requires a non-empty 'groups' table "
                "(one row per pool/regime with its parameters).",
                self.groups,
            )
        groups = self.groups
        for i, obligor in enumerate(self.obligors):
            if obligor.group is None:
                raise _request_fault(
                    ("obligors", i, "group"),
                    f"obligor {obligor.obligor_uid} has no group.",
                    obligor.group,
                )
        index = _second_occurrence([g.name for g in groups])
        if index is not None:
            raise _request_fault(
                ("groups", index, "name"),
                f"duplicate group name '{groups[index].name}' in 'groups'.",
                groups[index].name,
            )
        defined = {g.name for g in groups}
        for i, obligor in enumerate(self.obligors):
            if obligor.group not in defined:
                raise _request_fault(
                    ("obligors", i, "group"),
                    f"obligor group '{obligor.group}' is not defined in 'groups'.",
                    obligor.group,
                )
        # Each group's params against the model's per-group spec.
        spec_keys = {p.key for p in entry.params}
        for g, group in enumerate(groups):
            for key in group.params:
                if key not in spec_keys:
                    raise _request_fault(
                        ("groups", g, "params", key),
                        f"group '{group.name}' has unknown param '{key}'. "
                        f"Accepted: {sorted(spec_keys)}.",
                        group.params[key],
                    )
            for spec in entry.params:
                if spec.key not in group.params:
                    raise _request_fault(
                        ("groups", g, "params", spec.key),
                        f"group '{group.name}' is missing required param '{spec.key}'.",
                        dict(group.params),
                    )
                if not spec.contains(group.params[spec.key]):
                    raise _request_fault(
                        ("groups", g, "params", spec.key),
                        f"group '{group.name}' param '{spec.key}' = "
                        f"{group.params[spec.key]} out of range {spec.interval_repr}.",
                        group.params[spec.key],
                    )

    def _check_marshall_olkin_systemic_pd(self) -> None:
        """Rule 24: reject a common-shock probability no obligor can carry (P2-07).

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
        this constraint. The loc is the first offender; the message counts all
        offenders and previews a bounded number of them.
        """
        if self.model != "marshall_olkin":
            return

        spec = next(
            (p for p in model_entry(self.model).params if p.key == "systemic_pd"),
            None,
        )
        if spec is None:  # pragma: no cover - catalog would have to change
            return
        systemic_pd = float(self.params.get("systemic_pd", spec.default))

        offenders = [(i, o) for i, o in enumerate(self.obligors) if o.pd < systemic_pd]
        if offenders:
            preview = ", ".join(
                f"uid {o.obligor_uid} (PD={o.pd})" for _, o in offenders[:_PREVIEW]
            )
            first_index, first = offenders[0]
            raise _request_fault(
                ("obligors", first_index, "pd"),
                f"model 'marshall_olkin': {len(offenders)} obligor(s) have a total "
                f"PD below systemic_pd={systemic_pd}: {preview}. The common shock "
                "cannot be more likely than an obligor's own default: the "
                "decoupling 1 - (1 - PD) / (1 - systemic_pd) would be negative. "
                "Lower systemic_pd or remove the obligor(s).",
                first.pd,
            )

    def _check_obligor_rho(self) -> None:
        """Rule 25."""
        if model_entry(self.model).accepts_obligor_rho:
            return
        for i, obligor in enumerate(self.obligors):
            if obligor.rho is not None:
                raise _request_fault(
                    ("obligors", i, "rho"),
                    f"model '{self.model}' does not use a per-obligor 'rho'.",
                    obligor.rho,
                )

    @classmethod
    def from_dataframe(
        cls,
        obligor_table: pd.DataFrame,
        *,
        model: str,
        loadings_table: pd.DataFrame | None = None,
        group_params: dict[str, dict[str, float]] | None = None,
        n_scenarios: int = DEFAULT_N_SCENARIOS,
        alphas: list[float] | None = None,
        seed: int = 12345,
        **params: float,
    ) -> CreditVarRequest:
        """Build a request from a flat obligor DataFrame (notebook ergonomics).

        Outside the stable surface (P-CREDIT_VAR_API-001 clause 1, O-7).

        `obligor_table` needs the columns ``obligor_uid, ead, pd, lgd``. Extra
        columns are ignored, except an optional ``group`` column (grouped
        models) and an optional ``rho`` column (single-factor copulas). Model
        params are passed as keywords, e.g.::

            CreditVarRequest.from_dataframe(df, model="sf_gaussian", rho=0.20)

        For multi-factor models, pass `loadings_table` with the columns
        ``obligor_uid, factor, loading``::

            CreditVarRequest.from_dataframe(df, model="mf_gaussian", loadings_table=ld)

        For grouped models, put a ``group`` column on `df` and pass per-group
        params::

            CreditVarRequest.from_dataframe(
                df, model="beta_binomial_grouped",
                group_params={"IG": {"alpha": 1.0, "beta": 80.0},
                              "HY": {"alpha": 2.0, "beta": 18.0}})

        The default `n_scenarios` is the request default, `DEFAULT_N_SCENARIOS`.
        Each cell is cast with ``int``, ``float`` or ``str`` before validation;
        the request rules then apply as on the dict/JSON path. An
        ``obligor_uid`` cell that is a float with a fractional part (or NaN or
        +/-inf) is not cast: it reaches the strict int field and raises
        ``pydantic.ValidationError``.
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
                obligor_uid=_dataframe_uid(rec["obligor_uid"]),
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
                    obligor_uid=_dataframe_uid(rec["obligor_uid"]),
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


def _require_jtd_model_names(keys: Iterable[str], what: str) -> None:
    """Reject a key that is not a `JtdModel` member name (C.7).

    One concept has one key. The retired overlay key ``"forced_default"`` and
    any display label fail here, so a consumer never merges two keys for one
    model. Raised as ``ValueError`` inside a field validator, so pydantic
    reports it as a ``ValidationError`` at the field.
    """
    unknown = sorted(set(keys) - VALID_JTD_MODEL_NAMES)
    if unknown:
        raise ValueError(
            f"{what} keys must be JtdModel member names: unknown key(s) "
            f"{unknown[:_PREVIEW]}; valid names {sorted(VALID_JTD_MODEL_NAMES)}."
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
    # loss_max (owner decision ZV: run.py sets it, _zero_variance_moments
    # rejects any other value).
    loss_std_dev: float = Field(ge=0.0)
    loss_skewness: float  # right-skew of the loss distribution; 0.0 under ZV
    loss_excess_kurtosis: float  # excess kurtosis (fat-tailedness); 0.0 under ZV

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

    @field_validator("expected_loss_by_model")
    @classmethod
    def _keys_are_jtd_model_names(cls, v: dict[str, float]) -> dict[str, float]:
        _require_jtd_model_names(v, "expected_loss_by_model")
        return v

    @model_validator(mode="after")
    def _by_alpha_strictly_ascending(self) -> RiskSummary:
        _require_strictly_ascending(
            tuple(record.alpha for record in self.by_alpha), "by_alpha alpha"
        )
        return self

    @model_validator(mode="after")
    def _zero_variance_moments(self) -> RiskSummary:
        """Owner decision ZV: a constant loss has zero central moments.

        When ``loss_min == loss_max``, every scenario has the same loss. The
        floating-point mean of equal values can differ from that value by
        rounding, which gives a nonzero std and a skewness near +/-1. The
        moments are therefore exact 0.0 by rule, not by computation. The
        comparison is ``==``, so -0.0 also satisfies the rule.
        """
        if self.loss_min != self.loss_max:
            return self
        moments = {
            "loss_std_dev": self.loss_std_dev,
            "loss_skewness": self.loss_skewness,
            "loss_excess_kurtosis": self.loss_excess_kurtosis,
        }
        nonzero = {name: value for name, value in moments.items() if value != 0.0}
        if nonzero:
            raise ValueError(
                f"zero-variance rule (ZV): loss_min == loss_max == "
                f"{self.loss_min!r}, so loss_std_dev, loss_skewness and "
                f"loss_excess_kurtosis must equal 0.0; nonzero: {nonzero}."
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
    counts: tuple[int, ...]  # length nbins; every scenario in exactly one bin

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

    @field_validator("per_model_tail")
    @classmethod
    def _tail_keys_are_jtd_model_names(
        cls, v: dict[str, ModelTail]
    ) -> dict[str, ModelTail]:
        _require_jtd_model_names(v, "per_model_tail")
        return v

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

        No overflow above the display percentile occurs:
        `_attribution.loss_histogram` sets the last edge to
        ``max(p99.9, max loss, 1.0)``, which is at least the maximum loss, and
        ``np.histogram`` includes the last edge in the last bin. The first
        edge is 0.0 and a credit loss is >= 0. A count total other than
        n_scenarios therefore means a dropped or a double-counted scenario.
        """
        total = sum(self.loss_histogram.counts)
        if total != self.provenance.n_scenarios:
            raise ValueError(
                f"loss_histogram counts must sum to provenance.n_scenarios: sum = "
                f"{total}, n_scenarios = {self.provenance.n_scenarios}."
            )
        return self

    @model_validator(mode="after")
    def _by_alpha_matches_request_alphas(self) -> CreditVarResult:
        """The run reports exactly the requested levels (C.3).

        ``summary.by_alpha`` holds one record per request alpha and
        ``provenance.alphas`` holds the request alphas, so the two alpha sets
        are equal. Each request alpha is finite and in (0, 1), and ``==``
        equality of two such floats is bit equality. A missing level or a
        level that the request did not ask for is an engine fault.
        """
        computed = {record.alpha for record in self.summary.by_alpha}
        requested = set(self.provenance.alphas)
        if computed != requested:
            raise ValueError(
                f"summary.by_alpha alpha set must equal the provenance.alphas "
                f"set: in by_alpha only {sorted(computed - requested)}; in "
                f"provenance.alphas only {sorted(requested - computed)}."
            )
        return self
