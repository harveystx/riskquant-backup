# src/riskquant/solutions/risk_run/api.py
"""
The production scenario-VaR entry point.

    typed inputs -> validate -> build run identity -> [verify environment /
    inputs against a prior manifest] -> execute -> RiskRun (result + identity)

Composes ONLY existing production functions — the D036 scenario seam,
`apply_shocks_to_base_levels`, the compiled `SensitivityPnlCalculator`, the FX
conversion plan and `compute_var_es` — and adds no numerical logic. The manifest
is built from the very objects handed to the computation, before any numerical
work, so identity and computation cannot drift apart; it is never passed into a
generator, a kernel or a Numba signature (D001).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.reporting.fx_conversion import (
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.risk_run.schemas import (
    NonReproducibleRunError,
    ReportingSpec,
    ReproducibilityMode,
    RiskRun,
)
from riskquant.solutions.run_identity.api import (
    build_run_manifest,
    verify_environment,
    verify_run_inputs,
)
from riskquant.solutions.run_identity.hashing import (
    digest_reporting_plan,
    hash_compiled_book,
)
from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.scenario_models.api import build_scenario_generator
from riskquant.solutions.scenario_models.schemas import ScenarioModelConfig
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityPnlCalculator,
)

__all__ = ["run_scenario_var_es"]


_NO_SEED = object()


def _is_reproducible(config: ScenarioModelConfig) -> bool:
    """Deterministic models carry no `seed` field and are always reproducible;
    stochastic ones are reproducible only with an explicit seed (D039/D041)."""
    seed = getattr(config, "seed", _NO_SEED)
    return seed is _NO_SEED or seed is not None


def run_scenario_var_es(
    *,
    scenario_config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    required_factor_ids: npt.NDArray[np.int32],
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
    base_levels: npt.NDArray[np.float64],
    calculator: SensitivityPnlCalculator,
    reporting: ReportingSpec,
    confidence_level: float,
    batch_size: int,
    mode: ReproducibilityMode = "reproducible",
    reproduce: RunManifest | None = None,
) -> RiskRun:
    """
    Run one scenario-based VaR/ES and return the result with its identity.

    Parameters
    ----------
    required_factor_ids, required_shift_type_id, source_factor_ids, shock_scale
        The shock universe on the required-factor axis, in axis order. These are
        result-defining and are part of the identity (D040) — they are NOT part
        of the scenario config.
    base_levels
        Base levels on the same axis, `(K,)` float64.
    calculator
        The compiled production sensitivity book. Its CSR arrays are what is
        hashed, so the identity names exactly what priced the run.
    batch_size
        Operational only. The result is bitwise invariant to it (D041); it is
        deliberately absent from the identity.
    mode
        `"reproducible"` refuses a stochastic model with `seed=None` before any
        numerical work; `"exploratory"` permits it and marks the result.
    reproduce
        A prior manifest to reproduce against. When given, the environment and
        every supplied input are verified BEFORE execution and any mismatch
        raises; a stale or substituted snapshot never reaches the numerics.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    reproducible = _is_reproducible(scenario_config)
    if mode == "reproducible" and not reproducible:
        raise NonReproducibleRunError(
            "A reproducible run requires an explicit seed for a stochastic model; "
            "pass mode='exploratory' to run with seed=None (result marked "
            "non-reproducible)."
        )

    # ---- identity, from the exact objects the computation will consume -------
    manifest = build_run_manifest(
        scenario_config=scenario_config,
        data=data,
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=source_factor_ids,
        shock_scale=shock_scale,
        base_levels=base_levels,
        sensitivities_digest=hash_compiled_book(calculator.book),
        reporting_currency=reporting.reporting_ccy,
        fx_digest=digest_reporting_plan(
            reporting_ccy=reporting.reporting_ccy,
            settlement_ccys=reporting.settlement_ccys,
            fx_factor_names=reporting.fx_factor_names,
            fx_factor_rows=reporting.fx_factor_rows,
        ),
        confidence_level=confidence_level,
    )
    identity = manifest.identity()

    # ---- fail fast BEFORE numerics when reproducing a prior run --------------
    if reproduce is not None:
        verify_environment(reproduce)
        verify_run_inputs(
            reproduce,
            scenario_config=scenario_config,
            data=data,
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            shock_scale=shock_scale,
            base_levels=base_levels,
            book=calculator.book,
            confidence_level=confidence_level,
            reporting_currency=reporting.reporting_ccy,
        )
        if identity != reproduce.identity():
            # every verifiable input matched, so the residue is the reporting
            # plan or the environment: name it rather than compute a different run
            raise RuntimeError(
                "run identity differs from the manifest being reproduced although "
                "all verifiable inputs match; check the reporting plan."
            )

    # ---- execute: scenarios -> shocked levels ---------------------------------
    generator = build_scenario_generator(scenario_config, data=data)
    k = base_levels.shape[0]
    chunks: list[npt.NDArray[np.float64]] = []
    for batch in generator.iter_batches(
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=source_factor_ids,
        shock_scale=shock_scale,
        batch_size=batch_size,
    ):
        out = np.empty((k, batch.shocks.shape[0]), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=out)
        chunks.append(out)
    shocked_levels = np.ascontiguousarray(np.hstack(chunks), dtype=np.float64)

    # ---- P&L on the official calculator ---------------------------------------
    axis = FactorAxis.from_keys(
        pl.Series("factor_id", required_factor_ids.tolist(), dtype=pl.Int32)
    )
    settlement = calculator.run(
        axis.levels(base_levels=base_levels, shocked_levels=shocked_levels)
    )
    settlement_pnl = np.ascontiguousarray(settlement.pnl, dtype=np.float64)

    # ---- reporting currency ----------------------------------------------------
    plan = build_fx_conversion_plan(
        settlement_ccys=list(reporting.settlement_ccys),
        reporting_ccy=reporting.reporting_ccy,
        fx_factor_names=list(reporting.fx_factor_names),
    )
    fx_levels = np.ascontiguousarray(
        shocked_levels[reporting.fx_factor_rows], dtype=np.float64
    )
    converted = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl, fx_levels=fx_levels, plan=plan
    )

    # ---- VaR / ES: positions then portfolio, one canonical call ---------------
    groups = np.ascontiguousarray(
        np.vstack([converted.converted_pnl, converted.portfolio_pnl[None, :]]),
        dtype=np.float64,
    )
    metrics = compute_var_es(groups, confidence_level)

    return RiskRun(
        position_uids=np.asarray(settlement.position_uids, dtype=np.int64),
        var=np.asarray(metrics.var, dtype=np.float64),
        es=np.asarray(metrics.es, dtype=np.float64),
        reporting_ccy=reporting.reporting_ccy,
        manifest=manifest,
        identity=identity,
        reproducible=reproducible,
        pnl_provenance=settlement.provenance,
    )
