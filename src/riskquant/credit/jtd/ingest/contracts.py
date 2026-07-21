# src/riskquant/credit/jtd/ingest/contracts.py
"""
JTD ingest contracts: the single source of truth for what each model's master
frame must contain.

Every JTD model is authored as its own tightly-typed per-model master (one row
per obligor, no nullable union columns). This module declares, per model, the
exact column contract — which columns are required, which are factor-NAME
references (resolved to UIX downstream), which are float-valued, and whether the
model consumes the shared multi-factor loadings relation. Both the structural
validators (`validate.py`) and any generated documentation read from here, so
there is exactly one place that answers "what does it take to run model X".

The obligor key on every per-model frame is `OBLIGOR_KEY` ("uid"); PD and LGD are
never carried here — they are joined from the obligor master. The `instrument`
column is intentionally NOT part of any contract: a per-model frame's identity
(its slot in `JtdIngestBundle`) already declares the model, so an inline
`instrument` column would be redundant state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.enums.credit_factor_kind import CreditFactorKind

# Obligor key column on every per-model master frame.
OBLIGOR_KEY: Final[str] = "uid"


@dataclass(frozen=True, slots=True, kw_only=True)
class JtdModelContract:
    """Column contract for one JTD model's per-obligor master frame.

    `required_columns` always includes `OBLIGOR_KEY`. `factor_name_columns` and
    `float_columns` are subsets of `required_columns`, partitioning the
    model-specific columns by how they are validated and resolved downstream.
    `uses_mf_loadings` flags models whose systemic loadings live in the separate
    long-format relation rather than on this frame.
    """

    model: JtdModel
    required_columns: frozenset[str]
    factor_name_columns: frozenset[str] = field(default_factory=frozenset)
    float_columns: frozenset[str] = field(default_factory=frozenset)
    distinct_columns: frozenset[str] = field(default_factory=frozenset)
    uses_mf_loadings: bool = False

    def __post_init__(self) -> None:
        if OBLIGOR_KEY not in self.required_columns:
            raise ValueError(
                f"{self.model.name} contract must require the obligor key "
                f"'{OBLIGOR_KEY}'."
            )
        extra = (self.factor_name_columns | self.float_columns) - self.required_columns
        if extra:
            raise ValueError(
                f"{self.model.name} contract: factor_name/float columns "
                f"{sorted(extra)} are not in required_columns."
            )
        overlap = self.factor_name_columns & self.float_columns
        if overlap:
            raise ValueError(
                f"{self.model.name} contract: column(s) {sorted(overlap)} cannot "
                "be both factor-name and float."
            )
        bad_distinct = self.distinct_columns - self.factor_name_columns
        if bad_distinct:
            raise ValueError(
                f"{self.model.name} contract: distinct_columns {sorted(bad_distinct)} "
                "must be factor-name columns."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MfLoadingsContract:
    """Column contract for the shared long-format multi-factor loadings relation
    (one row per obligor x systemic factor)."""

    required_columns: frozenset[str]
    obligor_key: str = "obligor_uid"
    factor_name_column: str = "factor_name"
    loading_column: str = "factor_loading"


# -------------------------------------------------------------------------
# Per-model contracts
# -------------------------------------------------------------------------

JTD_MODEL_CONTRACTS: Final[dict[JtdModel, JtdModelContract]] = {
    JtdModel.FORCED_DEFAULT: JtdModelContract(
        model=JtdModel.FORCED_DEFAULT,
        required_columns=frozenset({OBLIGOR_KEY}),
    ),
    JtdModel.INDEPENDENT_BERNOULLI: JtdModelContract(
        model=JtdModel.INDEPENDENT_BERNOULLI,
        required_columns=frozenset({OBLIGOR_KEY, "uniform_factor"}),
        factor_name_columns=frozenset({"uniform_factor"}),
        distinct_columns=frozenset({"uniform_factor"}),
    ),
    JtdModel.BETA_BINOMIAL: JtdModelContract(
        model=JtdModel.BETA_BINOMIAL,
        required_columns=frozenset({OBLIGOR_KEY, "uniform_factor", "beta_factor"}),
        factor_name_columns=frozenset({"uniform_factor", "beta_factor"}),
        distinct_columns=frozenset({"uniform_factor"}),
    ),
    JtdModel.STRESS_REGIME_BERNOULLI: JtdModelContract(
        model=JtdModel.STRESS_REGIME_BERNOULLI,
        required_columns=frozenset(
            {
                OBLIGOR_KEY,
                "uniform_factor",
                "stress_regime_factor",
                "stress_pd_multiplier",
            }
        ),
        factor_name_columns=frozenset({"uniform_factor", "stress_regime_factor"}),
        float_columns=frozenset({"stress_pd_multiplier"}),
        distinct_columns=frozenset({"uniform_factor"}),
    ),
    JtdModel.COPULA_SF_GAUSSIAN: JtdModelContract(
        model=JtdModel.COPULA_SF_GAUSSIAN,
        required_columns=frozenset(
            {OBLIGOR_KEY, "factor_loading", "systemic_factor", "idiosyncratic_factor"}
        ),
        factor_name_columns=frozenset({"systemic_factor", "idiosyncratic_factor"}),
        float_columns=frozenset({"factor_loading"}),
    ),
    JtdModel.COPULA_SF_STUDENTS_T: JtdModelContract(
        model=JtdModel.COPULA_SF_STUDENTS_T,
        required_columns=frozenset(
            {
                OBLIGOR_KEY,
                "factor_loading",
                "systemic_factor",
                "idiosyncratic_factor",
                "chi_squared_factor",
                "nu",
            }
        ),
        factor_name_columns=frozenset(
            {"systemic_factor", "idiosyncratic_factor", "chi_squared_factor"}
        ),
        float_columns=frozenset({"factor_loading", "nu"}),
    ),
    # MF Copula models
    JtdModel.COPULA_MF_GAUSSIAN: JtdModelContract(
        model=JtdModel.COPULA_MF_GAUSSIAN,
        required_columns=frozenset({OBLIGOR_KEY, "idiosyncratic_factor"}),
        factor_name_columns=frozenset({"idiosyncratic_factor"}),
        uses_mf_loadings=True,
    ),
    JtdModel.COPULA_MF_STUDENTS_T: JtdModelContract(
        model=JtdModel.COPULA_MF_STUDENTS_T,
        required_columns=frozenset(
            {OBLIGOR_KEY, "idiosyncratic_factor", "chi_squared_factor", "nu"}
        ),
        factor_name_columns=frozenset({"idiosyncratic_factor", "chi_squared_factor"}),
        float_columns=frozenset({"nu"}),
        uses_mf_loadings=True,
    ),
    # Marshall-Olkin
    JtdModel.MARSHALL_OLKIN: JtdModelContract(
        model=JtdModel.MARSHALL_OLKIN,
        required_columns=frozenset(
            {
                OBLIGOR_KEY,
                "uniform_factor",
                "systemic_uniform_factor",
                "systemic_pd",
            }
        ),
        factor_name_columns=frozenset({"uniform_factor", "systemic_uniform_factor"}),
        float_columns=frozenset({"systemic_pd"}),
        distinct_columns=frozenset({"uniform_factor"}),
        uses_mf_loadings=False,
    ),
}

MF_LOADINGS_CONTRACT: Final[MfLoadingsContract] = MfLoadingsContract(
    required_columns=frozenset({"obligor_uid", "factor_name", "factor_loading"}),
)


# Role (factor-name column) -> required CreditFactorKind. Model-independent: a
# `uniform_factor` is always UNIFORM, a `systemic_factor` always SYSTEMIC, etc.
# Drives the semantic kind-validation loop so each role's kind is asserted in
# exactly one place.
FACTOR_COLUMN_KIND: Final[dict[str, CreditFactorKind]] = {
    "systemic_factor": CreditFactorKind.SYSTEMIC,
    "idiosyncratic_factor": CreditFactorKind.IDIOSYNCRATIC,
    "chi_squared_factor": CreditFactorKind.CHI,
    "uniform_factor": CreditFactorKind.UNIFORM,
    "systemic_uniform_factor": CreditFactorKind.UNIFORM,
    "beta_factor": CreditFactorKind.BETA_PD,
    "stress_regime_factor": CreditFactorKind.STRESS_INDICATOR,
}


def contract_for(model: JtdModel) -> JtdModelContract:
    return JTD_MODEL_CONTRACTS[model]


# Fail loud at import if a model ships without a contract (mirrors the engine's
# evaluator-registration guard).
_missing = [m.name for m in JtdModel if m not in JTD_MODEL_CONTRACTS]
if _missing:
    raise RuntimeError(
        f"No ingest contract registered for JTD model(s): {_missing}. "
        "Add them to JTD_MODEL_CONTRACTS in credit/jtd/ingest/contracts.py."
    )


_uncovered = sorted(
    {c for ct in JTD_MODEL_CONTRACTS.values() for c in ct.factor_name_columns}
    - set(FACTOR_COLUMN_KIND)
)
if _uncovered:
    raise RuntimeError(
        f"factor-name column(s) {_uncovered} have no FACTOR_COLUMN_KIND entry."
    )
