# src/riskquant/credit/jtd/registry.py
"""
JTD model registry — the active "vending machine" for model wiring.

A single `JtdModelSpec` per model captures everything needed to wire that model:
its contract, builder, compiled block type, evaluator (by lazy import
coordinates), and the attribute names of its compiled-portfolio slot and ingest
bundle field. `JTD_MODEL_SPECS` is the ONE ordered list; every dispatch table is
derived from it:

    * `evaluator_table()`   -> engine's model-id -> evaluator dispatch
    * `model_field_order()` -> bundle's per-model frame slots
    * `model_contracts()`   -> ingest contracts by model

Layering note (why evaluators are referenced by import coordinates, not imported
at module top): the ingest edge (compile / bundle / contracts) is Numba-free by
design — strings die at ingest, kernels live in the core. Evaluators import
kernels, which import Numba. If the registry imported evaluators eagerly, then
routing the ingest path through the registry would drag Numba into the ingest
edge. So evaluators are resolved lazily in `evaluator_table()` (called at engine
import time, where Numba is required anyway), keeping the registry itself — and
therefore `compile.py` and `bundle.py` — Numba-free.

Adding a model: write its block/kernel/evaluator/builder, append one
`JtdModelSpec`, append one `JtdModel` enum value. The import-time guards reject
any drift between the enum, the specs, and the compiled/bundle slot names.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.blocks.beta_binomial import BetaBinomialJtdBlock
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.marshall_olkin import MarshallOlkinJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_gaussian import MfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_students_t import (
    MfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.sf_copula_students_t import SfStudentsTCopulaJtdBlock
from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.type_alias import JtdBlockTypeUnion
from riskquant.credit.jtd.evaluators.base import JtdEvaluator
from riskquant.credit.jtd.ingest.builders.beta_binomial import build_beta_binomial_block
from riskquant.credit.jtd.ingest.builders.forced_default import (
    build_forced_default_block,
)
from riskquant.credit.jtd.ingest.builders.independent_bernoulli import (
    build_independent_bernoulli_block,
)
from riskquant.credit.jtd.ingest.builders.marshall_olkin import (
    build_marshall_olkin_block,
)
from riskquant.credit.jtd.ingest.builders.mf_gaussian import build_mf_gaussian_block
from riskquant.credit.jtd.ingest.builders.mf_students_t import (
    build_mf_students_t_block,
)
from riskquant.credit.jtd.ingest.builders.sf_gaussian import build_sf_gaussian_block
from riskquant.credit.jtd.ingest.builders.sf_students_t import build_sf_students_t_block
from riskquant.credit.jtd.ingest.builders.stress_regime_bernoulli import (
    build_stress_regime_bernoulli_block,
)
from riskquant.credit.jtd.ingest.contracts import JTD_MODEL_CONTRACTS, JtdModelContract
from riskquant.credit.jtd.models import NUM_JTD_MODELS, JtdModel

_EVALUATOR_PKG: Final[str] = "riskquant.credit.jtd.evaluators"


@dataclass(frozen=True, slots=True, kw_only=True)
class JtdModelSpec:
    """Everything needed to wire one JTD model into the pipeline.

    The evaluator is referenced by ``(evaluator_module, evaluator_attr)`` and
    resolved lazily (see module docstring); the builder is held directly because
    builders are Numba-free. `needs_factor_uix` / `needs_mf_loadings` let the
    registry dispatch the non-uniform builder signatures positionally.
    """

    model: JtdModel
    contract: JtdModelContract
    block_type: type[_JtdBlock]
    builder: Callable[..., JtdBlockTypeUnion]
    evaluator_module: str  # module under riskquant.credit.jtd.evaluators
    evaluator_attr: str  # evaluator callable name within that module
    compiled_slot: str  # attribute on CompiledJtdPortfolio
    bundle_field: str  # attribute on JtdIngestBundle
    needs_factor_uix: bool  # builder takes factor_name_to_uix
    needs_mf_loadings: bool  # builder takes the shared mf_loadings frame
    # Name of this model's per-model semantic validator in ingest/validate.py,
    # resolved there (validators co-locate with the predicates they call); None
    # if the model needs no semantic checks beyond the contract-driven defaults.
    semantic_validator_attr: str | None = None

    def resolve_evaluator(self) -> JtdEvaluator[Any]:
        """Import and return the evaluator (pulls Numba via the kernel)."""
        module = importlib.import_module(f"{_EVALUATOR_PKG}.{self.evaluator_module}")
        evaluator = getattr(module, self.evaluator_attr)
        if not isinstance(evaluator, JtdEvaluator):
            raise TypeError(
                f"{self.model.name}: {self.evaluator_module}.{self.evaluator_attr} "
                f"does not satisfy the JtdEvaluator protocol: {evaluator!r}"
            )
        return evaluator


# The ONE ordered registration list. Order is JtdModel band order.
JTD_MODEL_SPECS: Final[tuple[JtdModelSpec, ...]] = (
    JtdModelSpec(
        model=JtdModel.FORCED_DEFAULT,
        contract=JTD_MODEL_CONTRACTS[JtdModel.FORCED_DEFAULT],
        block_type=ForcedDefaultJtdBlock,
        builder=build_forced_default_block,
        evaluator_module="forced_default",
        evaluator_attr="evaluate_forced_default_jtd",
        compiled_slot="forced_default_jtd_block",
        bundle_field="forced_default_df",
        needs_factor_uix=False,
        needs_mf_loadings=False,
    ),
    JtdModelSpec(
        model=JtdModel.INDEPENDENT_BERNOULLI,
        contract=JTD_MODEL_CONTRACTS[JtdModel.INDEPENDENT_BERNOULLI],
        block_type=IndependentBernoulliJtdBlock,
        builder=build_independent_bernoulli_block,
        evaluator_module="independent_bernoulli",
        evaluator_attr="evaluate_independent_bernoulli_jtd",
        compiled_slot="independent_bernoulli_jtd_block",
        bundle_field="independent_bernoulli_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
    ),
    JtdModelSpec(
        model=JtdModel.BETA_BINOMIAL,
        contract=JTD_MODEL_CONTRACTS[JtdModel.BETA_BINOMIAL],
        block_type=BetaBinomialJtdBlock,
        builder=build_beta_binomial_block,
        evaluator_module="beta_binomial",
        evaluator_attr="evaluate_beta_binomial_jtd",
        compiled_slot="beta_binomial_jtd_block",
        bundle_field="beta_binomial_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
    ),
    JtdModelSpec(
        model=JtdModel.STRESS_REGIME_BERNOULLI,
        contract=JTD_MODEL_CONTRACTS[JtdModel.STRESS_REGIME_BERNOULLI],
        block_type=StressRegimeBernoulliJtdBlock,
        builder=build_stress_regime_bernoulli_block,
        evaluator_module="stress_regime_bernoulli",
        evaluator_attr="evaluate_stress_regime_bernoulli_jtd",
        compiled_slot="stress_regime_bernoulli_jtd_block",
        bundle_field="stress_regime_bernoulli_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
    ),
    JtdModelSpec(
        model=JtdModel.COPULA_SF_GAUSSIAN,
        contract=JTD_MODEL_CONTRACTS[JtdModel.COPULA_SF_GAUSSIAN],
        block_type=SfGaussianCopulaJtdBlock,
        builder=build_sf_gaussian_block,
        evaluator_module="sf_copula_gaussian",
        evaluator_attr="evaluate_sf_gaussian_copula_jtd",
        compiled_slot="sf_gaussian_copula_jtd_block",
        bundle_field="sf_gaussian_copula_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
        semantic_validator_attr="_semantic_sf_gaussian",
    ),
    JtdModelSpec(
        model=JtdModel.COPULA_SF_STUDENTS_T,
        contract=JTD_MODEL_CONTRACTS[JtdModel.COPULA_SF_STUDENTS_T],
        block_type=SfStudentsTCopulaJtdBlock,
        builder=build_sf_students_t_block,
        evaluator_module="sf_copula_students_t",
        evaluator_attr="evaluate_sf_students_t_copula_jtd",
        compiled_slot="sf_students_t_copula_jtd_block",
        bundle_field="sf_students_t_copula_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
        semantic_validator_attr="_semantic_sf_students_t",
    ),
    JtdModelSpec(
        model=JtdModel.COPULA_MF_GAUSSIAN,
        contract=JTD_MODEL_CONTRACTS[JtdModel.COPULA_MF_GAUSSIAN],
        block_type=MfGaussianCopulaJtdBlock,
        builder=build_mf_gaussian_block,
        evaluator_module="mf_copula_gaussian",
        evaluator_attr="evaluate_mf_gaussian_copula_jtd",
        compiled_slot="mf_gaussian_copula_jtd_block",
        bundle_field="mf_gaussian_copula_df",
        needs_factor_uix=True,
        needs_mf_loadings=True,
        semantic_validator_attr="_semantic_mf_gaussian",
    ),
    JtdModelSpec(
        model=JtdModel.COPULA_MF_STUDENTS_T,
        contract=JTD_MODEL_CONTRACTS[JtdModel.COPULA_MF_STUDENTS_T],
        block_type=MfStudentsTCopulaJtdBlock,
        builder=build_mf_students_t_block,
        evaluator_module="mf_copula_students_t",
        evaluator_attr="evaluate_mf_students_t_copula_jtd",
        compiled_slot="mf_students_t_copula_jtd_block",
        bundle_field="mf_students_t_copula_df",
        needs_factor_uix=True,
        needs_mf_loadings=True,
        semantic_validator_attr="_semantic_mf_students_t",
    ),
    JtdModelSpec(
        model=JtdModel.MARSHALL_OLKIN,
        contract=JTD_MODEL_CONTRACTS[JtdModel.MARSHALL_OLKIN],
        block_type=MarshallOlkinJtdBlock,
        builder=build_marshall_olkin_block,
        evaluator_module="marshall_olkin",
        evaluator_attr="evaluate_marshall_olkin_jtd",
        compiled_slot="marshall_olkin_jtd_block",
        bundle_field="marshall_olkin_df",
        needs_factor_uix=True,
        needs_mf_loadings=False,
        semantic_validator_attr="_semantic_marshall_olkin",
    ),
)

JTD_MODEL_REGISTRY: Final[MappingProxyType[JtdModel, JtdModelSpec]] = MappingProxyType(
    {spec.model: spec for spec in JTD_MODEL_SPECS}
)


# -------------------------------------------------------------------------
# Import-time validation (all Numba-free — evaluators are NOT resolved here).
# -------------------------------------------------------------------------
if len(JTD_MODEL_REGISTRY) != len(JTD_MODEL_SPECS):
    raise RuntimeError("JTD_MODEL_SPECS contains duplicate models.")

_missing = [m.name for m in JtdModel if m not in JTD_MODEL_REGISTRY]
if _missing:
    raise RuntimeError(f"JTD models with no registry spec: {_missing}.")

if tuple(spec.model for spec in JTD_MODEL_SPECS) != tuple(JtdModel):
    raise RuntimeError("JTD_MODEL_SPECS order must match JtdModel band order exactly.")

for _spec in JTD_MODEL_SPECS:
    if _spec.contract.model is not _spec.model:
        raise RuntimeError(
            f"{_spec.model.name}: spec.contract is for {_spec.contract.model.name}."
        )
# Slot/field name existence is validated by the consumers that own those
# dataclasses (compiled.py at construction, bundle.py at import), avoiding a
# registry -> compiled/bundle import cycle.


def evaluator_table() -> tuple[JtdEvaluator[Any], ...]:
    """Model-id-indexed evaluator dispatch — the engine's `JTD_EVALUATORS`.

    Resolves (and Numba-imports) every evaluator. Completeness + contiguity are
    guaranteed at import, so entry ``i`` is the evaluator for ``JtdModel(i)``.
    """
    table: list[JtdEvaluator[Any] | None] = [None] * NUM_JTD_MODELS
    for spec in JTD_MODEL_SPECS:
        table[int(spec.model)] = spec.resolve_evaluator()
    return tuple(table)  # type: ignore[arg-type]  # completeness validated above


def model_field_order() -> tuple[tuple[JtdModel, str], ...]:
    """Per-model bundle frame slots (single source for bundle._MODEL_FIELD_ORDER)."""
    return tuple((spec.model, spec.bundle_field) for spec in JTD_MODEL_SPECS)


def model_contracts() -> MappingProxyType[JtdModel, JtdModelContract]:
    """Ingest contracts by model (single source for JTD_MODEL_CONTRACTS)."""
    return MappingProxyType({spec.model: spec.contract for spec in JTD_MODEL_SPECS})
