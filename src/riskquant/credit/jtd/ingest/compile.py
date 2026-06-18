# src/riskquant/credit/jtd/ingest/compile.py
"""
JTD ingest orchestration: a `JtdIngestBundle` (per-model masters + the shared
multi-factor loadings relation) + a validated `CompiledCreditMasters`
aggregate -> `CompiledJtdPortfolio`.

The compile is a thin pipeline:

    structural validate  (validate.py: shape, dtypes, uid invariants)
        -> semantic validate (validate.py: factor kinds, distinctness, ranges,
                              chi/nu reconciliation, sum-beta^2)
        -> per-model builders (builders/: resolve + construct; input assumed valid)
        -> assemble CompiledJtdPortfolio

PD and LGD are never on the ingest frames — they are joined from the obligor
master (the single source of truth) inside the builders. Factor names die here,
resolved to int32 UIX against the factor universe.
"""

from __future__ import annotations

__all__ = [
    "compile_jtd_portfolio",
]

from typing import Any

from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.validate import (
    validate_ingest_bundle,
    validate_ingest_bundle_semantics,
)
from riskquant.credit.jtd.registry import JTD_MODEL_SPECS
from riskquant.credit.masters.cross_master.compiled import CompiledCreditMasters


def compile_jtd_portfolio(
    *,
    bundle: JtdIngestBundle,
    masters: CompiledCreditMasters,
) -> CompiledJtdPortfolio:
    """Compile a validated JTD ingest bundle into a ``CompiledJtdPortfolio``.

    Parameters
    ----------
    bundle
        The mixed JTD ingest master: one optional per-model frame per
        ``JtdModel`` plus the shared long-format ``mf_loadings`` relation. Each
        per-model frame is one row per obligor and carries only that model's
        contract columns (``instrument`` is implicit in the frame's slot). PD/LGD
        are NOT carried — they are joined from the obligor master.
    masters
        Validated credit master aggregate. Holding it is proof its constituent
        masters cross-validate, so the cross-master guarantee is a precondition
        of compilation rather than the caller's discipline. The obligor master
        is the single source of truth for PD/LGD; factor names resolve to int32
        UIX against the factor universe, whose contract also supplies the chi
        degrees of freedom that reconcile obligor ``nu``.
    """
    obligor_master = masters.obligor_master
    factor_universe = masters.factor_universe
    # 1. Structural validation (shape / dtypes / uid invariants / loadings shape).
    validate_ingest_bundle(bundle, obligor_master)

    # 2. Semantic / wiring validation (factor kinds, distinctness, ranges, chi
    #    reconciliation, sum-beta^2). After this, builders assume valid input.
    validate_ingest_bundle_semantics(bundle, factor_universe)

    factor_name_to_uix = factor_universe.factor_name_to_uix
    frames = bundle.frames_by_model()
    mf_loadings_df = bundle.mf_loadings_df

    # Registry-driven dispatch: one pass over the specs, build each populated
    # model's block via its (signature-tagged) builder, key it by its compiled
    # slot. No per-model branch here — adding a model is one registry entry.
    blocks_by_slot: dict[str, Any] = {}
    for spec in JTD_MODEL_SPECS:
        frame = frames.get(spec.model)
        if frame is None:
            continue
        args: list[object] = [frame, obligor_master]
        if spec.needs_factor_uix:
            args.append(factor_name_to_uix)
        if spec.needs_mf_loadings:
            if mf_loadings_df is None:
                raise ValueError(
                    f"compile_jtd_portfolio: {spec.model.name} requires a "
                    f"long-format loadings frame, but bundle.mf_loadings_df is "
                    f"None."
                )
            args.append(mf_loadings_df)
        blocks_by_slot[spec.compiled_slot] = spec.builder(*args)

    return CompiledJtdPortfolio(**blocks_by_slot)
