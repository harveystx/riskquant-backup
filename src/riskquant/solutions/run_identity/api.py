# src/riskquant/solutions/run_identity/api.py
"""
Public surface for run identity: build a manifest from the same inputs the risk
path consumes, and verify a persisted manifest against the inputs in hand.

The manifest is built beside the computation, never inside it: no function here
is called by a generator or a kernel, and nothing here is passed into one.
"""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.run_identity.hashing import (
    digest_array,
    hash_compiled_book,
    hash_parametric_risk_data,
    hash_proxy_model,
    hash_risk_factor_matrix,
)
from riskquant.solutions.run_identity.schemas import (
    CompositionIdentity,
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
    "EnvironmentMismatchError",
    "InputMismatchError",
    "build_run_manifest",
    "engine_identity",
    "scenario_config_from_manifest",
    "verify_environment",
    "verify_run_inputs",
]


class EnvironmentMismatchError(RuntimeError):
    """The current numerical environment differs from the one the manifest
    declares. Bitwise reproduction is only claimed under a matching
    environment (D041), so a rerun must not proceed as if it were reproducing.
    """


class InputMismatchError(RuntimeError):
    """A supplied input does not match the one the manifest identifies.

    Raised instead of silently computing a different, plausible risk number from
    a stale or substituted snapshot.
    """


# def engine_identity() -> EngineIdentity:
#     """Numerical environment of the current process."""
#     try:
#         from riskquant import __version__ as rq_version
#     except Exception:  # pragma: no cover - version module is optional
#         rq_version = "unknown"
#     return EngineIdentity(
#         riskquant_version=str(rq_version),
#         python_version=platform.python_version(),
#         numpy_version=np.__version__,
#         numba_version=numba.__version__,
#     )


def engine_identity() -> EngineIdentity:
    """Numerical environment of the current process."""
    try:
        # Fetches version from package metadata rather than the module namespace
        rq_version = importlib.metadata.version("riskquant")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
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
    proxy_model: CompiledProxyModel,
    base_levels: npt.NDArray[np.float64],
    sensitivities_digest: str,
    reporting_currency: str,
    fx_digest: str,
    confidence_level: float,
    composition: CompositionIdentity | None = None,
) -> RunManifest:
    """
    Assemble the immutable identity of one risk run.

    `sensitivities_digest` and `fx_digest` are supplied as digests so that a plain
    sensitivity vector (`digest_array`) and a compiled book (`hash_compiled_book`)
    share one manifest path without a nullable alternative.

    Every argument is a result-defining input established by mutation testing;
    nothing operational is included, so the identity is stable across reruns of
    the same computation. `batch_size` is deliberately absent: the official risk
    path is bitwise invariant to batch partitioning (D041), so it is an
    execution knob, not a mathematical input.

    `proxy_model` is the run-bound Proxy V1 operator compiled over the run's
    full axis; its identity is `hash_proxy_model(proxy_model)` (D088 §7). A
    manifest carrying it requires `composition` (D088 §13.3).
    """
    return RunManifest(
        engine=engine_identity(),
        scenario_config=scenario_config.model_dump(mode="json"),
        data=_data_identity(data),
        shocks=ShockIdentity(proxy_model_digest=hash_proxy_model(proxy_model)),
        pnl=PnlIdentity(
            base_levels_digest=digest_array(base_levels),
            sensitivities_digest=sensitivities_digest,
            composition=composition,
        ),
        reporting=ReportingIdentity(
            reporting_currency=reporting_currency, fx_digest=fx_digest
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


def verify_environment(manifest: RunManifest) -> None:
    """
    Fail fast when the running environment is not the manifest's environment.

    Strict equality on every `engine` field. No compatibility policy is applied:
    the reproducibility claim is BITWISE, NumPy publishes no cross-version
    guarantee for `Generator` distributions and Numba/LLVM codegen governs FMA
    contraction, so no evidence exists that any two versions are bit-compatible.
    """
    current = engine_identity()
    expected = manifest.engine
    for field in (
        "riskquant_version",
        "python_version",
        "numpy_version",
        "numba_version",
    ):
        got, want = getattr(current, field), getattr(expected, field)
        if got != want:
            raise EnvironmentMismatchError(
                f"{field} mismatch: manifest declares {want!r}, runtime is {got!r}; "
                f"bitwise reproduction is not claimed across environments."
            )


def verify_run_inputs(manifest: RunManifest, **inputs: Any) -> None:
    """
    Fail fast when the inputs in hand are not the ones the manifest identifies.

    Pass any subset of the `build_run_manifest` inputs by the same keyword. Each
    supplied input is re-digested and compared; the first mismatch raises,
    naming the input. This is what makes a stale or substituted snapshot
    detectable instead of silently producing a different risk number.
    """
    expected: dict[str, str] = {
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
        if name == "proxy_model":
            actual_digest = hash_proxy_model(value)
            if actual_digest != manifest.shocks.proxy_model_digest:
                raise InputMismatchError(
                    f"proxy model mismatch: manifest identifies "
                    f"{manifest.shocks.proxy_model_digest[:16]}..., got "
                    f"{actual_digest[:16]}..."
                )
            continue
        if name == "book":
            actual_digest = hash_compiled_book(value)
            if actual_digest != manifest.pnl.sensitivities_digest:
                raise InputMismatchError(
                    f"book content mismatch: manifest identifies "
                    f"{manifest.pnl.sensitivities_digest[:16]}..., got "
                    f"{actual_digest[:16]}..."
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
