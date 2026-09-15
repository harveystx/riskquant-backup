# src/riskquant/solutions/run_identity/api.py
"""
Public surface for run identity: build a manifest from the same inputs the risk
path consumes, and verify a persisted manifest against the inputs in hand.

The manifest is built beside the computation, never inside it: no function here
is called by a generator or a kernel, and nothing here is passed into one.
"""

from __future__ import annotations

import platform
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.run_identity.hashing import (
    digest_array,
    hash_parametric_risk_data,
    hash_risk_factor_matrix,
)
from riskquant.solutions.run_identity.schemas import (
    DataIdentity,
    EngineIdentity,
    PnlIdentity,
    ReportingIdentity,
    RiskMeasureIdentity,
    RunManifest,
    ShockIdentity,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.scenario_models.schemas import ScenarioModelConfig

__all__ = [
    "InputMismatchError",
    "build_run_manifest",
    "engine_identity",
    "scenario_config_from_manifest",
    "verify_run_inputs",
]


class InputMismatchError(RuntimeError):
    """A supplied input does not match the one the manifest identifies.

    Raised instead of silently computing a different, plausible risk number from
    a stale or substituted snapshot.
    """


def engine_identity() -> EngineIdentity:
    """Numerical environment of the current process."""
    try:
        from riskquant import __version__ as rq_version
    except Exception:  # pragma: no cover - version module is optional
        rq_version = "unknown"
    return EngineIdentity(
        riskquant_version=str(rq_version),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        numba_version=numba.__version__,
    )


def _data_identity(data: RiskFactorMatrix | ParametricRiskData) -> DataIdentity:
    if isinstance(data, RiskFactorMatrix):
        return DataIdentity(
            kind="risk_factor_matrix", digest=hash_risk_factor_matrix(data)
        )
    if isinstance(data, ParametricRiskData):
        return DataIdentity(
            kind="parametric_risk_data", digest=hash_parametric_risk_data(data)
        )
    raise TypeError(f"Unsupported construction data: {type(data).__name__}.")


def build_run_manifest(
    *,
    scenario_config: ScenarioModelConfig,
    data: RiskFactorMatrix | ParametricRiskData,
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
    base_levels: npt.NDArray[np.float64],
    sensitivities: npt.NDArray[np.float64],
    reporting_currency: str,
    fx: npt.NDArray[np.float64],
    confidence_level: float,
) -> RunManifest:
    """
    Assemble the immutable identity of one risk run.

    Every argument is a result-defining input established by mutation testing;
    nothing operational is included, so the identity is stable across reruns of
    the same computation. `batch_size` is deliberately absent: the official risk
    path is bitwise invariant to batch partitioning (D041), so it is an
    execution knob, not a mathematical input.
    """
    return RunManifest(
        engine=engine_identity(),
        scenario_config=scenario_config.model_dump(mode="json"),
        data=_data_identity(data),
        shocks=ShockIdentity(
            required_shift_type_ids_digest=digest_array(required_shift_type_id),
            source_factor_ids_digest=digest_array(source_factor_ids),
            shock_scale_digest=digest_array(shock_scale),
        ),
        pnl=PnlIdentity(
            base_levels_digest=digest_array(base_levels),
            sensitivities_digest=digest_array(sensitivities),
        ),
        reporting=ReportingIdentity(
            reporting_currency=reporting_currency, fx_digest=digest_array(fx)
        ),
        risk_measure=RiskMeasureIdentity(confidence_level=confidence_level),
    )


def scenario_config_from_manifest(manifest: RunManifest) -> ScenarioModelConfig:
    """
    Rebuild the typed scenario config from a persisted manifest.

    Reuses the frozen D036 seam validation rather than duplicating it, so a
    manifest cannot reconstruct a configuration the seam would reject.
    """
    return parse_scenario_model_config(manifest.scenario_config)


def verify_run_inputs(manifest: RunManifest, **inputs: Any) -> None:
    """
    Fail fast when the inputs in hand are not the ones the manifest identifies.

    Pass any subset of the `build_run_manifest` inputs by the same keyword. Each
    supplied input is re-digested and compared; the first mismatch raises,
    naming the input. This is what makes a stale or substituted snapshot
    detectable instead of silently producing a different risk number.
    """
    expected: dict[str, str] = {
        "required_shift_type_id": manifest.shocks.required_shift_type_ids_digest,
        "source_factor_ids": manifest.shocks.source_factor_ids_digest,
        "shock_scale": manifest.shocks.shock_scale_digest,
        "base_levels": manifest.pnl.base_levels_digest,
        "sensitivities": manifest.pnl.sensitivities_digest,
        "fx": manifest.reporting.fx_digest,
    }
    for name, value in inputs.items():
        if name == "data":
            actual = _data_identity(value)
            if actual.kind != manifest.data.kind:
                raise InputMismatchError(
                    f"data kind mismatch: manifest identifies "
                    f"{manifest.data.kind}, got {actual.kind}."
                )
            if actual.digest != manifest.data.digest:
                raise InputMismatchError(
                    f"data content mismatch: manifest identifies "
                    f"{manifest.data.digest[:16]}..., got {actual.digest[:16]}..."
                )
            continue
        if name == "scenario_config":
            if value.model_dump(mode="json") != manifest.scenario_config:
                raise InputMismatchError("scenario_config does not match manifest.")
            continue
        if name == "confidence_level":
            if float(value) != manifest.risk_measure.confidence_level:
                raise InputMismatchError("confidence_level does not match manifest.")
            continue
        if name == "reporting_currency":
            if value != manifest.reporting.reporting_currency:
                raise InputMismatchError("reporting_currency does not match manifest.")
            continue
        if name not in expected:
            raise TypeError(f"Unknown run input: {name!r}.")
        actual_digest = digest_array(value)
        if actual_digest != expected[name]:
            raise InputMismatchError(
                f"{name} content mismatch: manifest identifies "
                f"{expected[name][:16]}..., got {actual_digest[:16]}..."
            )
