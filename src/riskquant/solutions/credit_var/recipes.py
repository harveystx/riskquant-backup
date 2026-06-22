# src/riskquant/solutions/credit_var/recipes.py
"""
Model recipes: flat obligor table + scalar params -> compiled engine inputs.

The authoring layer of the credit-VaR solution: it turns a desk's flat sheet plus
a handful of scalar knobs into the factor universe + per-model JTD frame the engine
needs, which the author never sees. Single-factor Gaussian: rho becomes a systemic
factor, per-obligor idiosyncratic factors, and a sqrt(rho) loading. Beta-binomial:
(alpha, beta) become a shared BETA_PD factor. Student-t: adds a shared CHI factor
at nu. Stress-regime: a shared STRESS_INDICATOR plus a per-run stress probability.

This is solution-tier, not core: the core engine takes a fully-authored universe +
bundle; "construct them from a flat table" is presentation/authoring sugar that
belongs with the solution. Plain DataFrame + params in, engine objects out,
fail-fast. Adding a flat-sheet model is one assembler here plus a catalog entry.

Multi-factor copulas are deliberately absent: their systemic loadings are a
relational input (a loadings sheet), not a scalar, and are handled by a separate
relational recipe path.
"""

from __future__ import annotations

__all__ = [
    "AssembledInputs",
    "SUPPORTED_RECIPES",
    "assemble_inputs",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad
from riskquant.credit.exposure.obligor_ead.ingest import compile_obligor_ead
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.cross_master.compiled import CompiledCreditMasters
from riskquant.credit.masters.factor_universe.compiled import (
    CompiledCreditFactorUniverse,
)
from riskquant.credit.masters.factor_universe.ingest import (
    compile_credit_factor_universe,
)
from riskquant.credit.masters.obligors.ingest import compile_obligor_master

_SYSTEMIC_GLOBAL = "SYSTEMIC_GLOBAL"
_BETA_PD_GLOBAL = "BETA_PD_GLOBAL"
_STRESS_GLOBAL = "STRESS_GLOBAL"
_CHI_GLOBAL = "CHI_GLOBAL"
_FORCED_DUMMY = (
    "FORCED_DUMMY"  # forced-default reads no factor; the generator needs >=1
)

_DEFAULT_RHO = 0.15
_DEFAULT_NU = 8.0
_DEFAULT_BETA_ALPHA = 1.0
_DEFAULT_BETA_BETA = 24.0  # mean PD = alpha/(alpha+beta) = 0.04
_DEFAULT_STRESS_PROB = 0.05
_DEFAULT_STRESS_MULT = 3.0


@dataclass(frozen=True, slots=True, kw_only=True)
class AssembledInputs:
    """Compiled engine inputs for one run.

    `stress_p_by_uix` carries the per-run STRESS_INDICATOR probabilities the
    generator needs (the one knob kept off the static universe); empty for models
    with no stress factor.
    """

    masters: CompiledCreditMasters
    bundle: JtdIngestBundle
    obligor_ead: CompiledObligorEad
    stress_p_by_uix: dict[int, float] = field(default_factory=dict)


# A universe row: factor name + kind, with optional marginal params the kind needs
# (nu for CHI, alpha/beta for BETA_PD).
@dataclass(frozen=True, slots=True, kw_only=True)
class _FactorRow:
    name: str
    kind: str
    nu: float | None = None
    alpha: float | None = None
    beta: float | None = None


def _uids(obligor_table: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(obligor_table["uid"].to_numpy(dtype=np.int64))


def _universe(rows: list[_FactorRow]) -> CompiledCreditFactorUniverse:
    """Compile factor rows into a universe, emitting the nu / alpha / beta columns
    only when a kind that needs them is present (the ingest maps NaN elsewhere to
    its sentinels)."""
    data: dict[str, object] = {
        "factor_id": np.arange(len(rows), dtype=np.int64),
        "factor_name": [r.name for r in rows],
        "factor_kind": [r.kind for r in rows],
    }
    if any(r.nu is not None for r in rows):
        data["nu"] = [r.nu if r.nu is not None else np.nan for r in rows]
    if any(r.alpha is not None for r in rows):
        data["alpha"] = [r.alpha if r.alpha is not None else np.nan for r in rows]
    if any(r.beta is not None for r in rows):
        data["beta"] = [r.beta if r.beta is not None else np.nan for r in rows]
    return compile_credit_factor_universe(pd.DataFrame(data))


def _masters(
    obligor_table: pd.DataFrame, universe: CompiledCreditFactorUniverse
) -> CompiledCreditMasters:
    # compile_obligor_master reads uid/pd/lgd and ignores the rest (e.g. ead).
    return CompiledCreditMasters(
        obligor_master=compile_obligor_master(obligor_table),
        factor_universe=universe,
    )


def _ead(obligor_table: pd.DataFrame) -> CompiledObligorEad:
    return compile_obligor_ead(obligor_table)


def _require(params: Mapping[str, float], key: str, default: float) -> float:
    return float(params.get(key, default))


# --------------------------------------------------------------------------
# Per-model assemblers
# --------------------------------------------------------------------------


def _assemble_forced_default(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    # Forced default reads no factor, but the generator needs a non-empty universe.
    universe = _universe([_FactorRow(name=_FORCED_DUMMY, kind="UNIFORM")])
    frame = pd.DataFrame({"uid": uids, "instrument": JtdModel.FORCED_DEFAULT.name})
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_independent_bernoulli(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    uniform_names = [f"U_{int(u)}" for u in uids]
    universe = _universe([_FactorRow(name=n, kind="UNIFORM") for n in uniform_names])
    frame = pd.DataFrame(
        {
            "uid": uids,
            "instrument": JtdModel.INDEPENDENT_BERNOULLI.name,
            "uniform_factor": uniform_names,
        }
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_beta_binomial(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    alpha = _require(params, "alpha", _DEFAULT_BETA_ALPHA)
    beta = _require(params, "beta", _DEFAULT_BETA_BETA)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"beta_binomial: alpha={alpha}, beta={beta} must be > 0.")

    uniform_names = [f"U_{int(u)}" for u in uids]
    universe = _universe(
        [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
        + [_FactorRow(name=_BETA_PD_GLOBAL, kind="BETA_PD", alpha=alpha, beta=beta)]
    )
    frame = pd.DataFrame(
        {
            "uid": uids,
            "instrument": JtdModel.BETA_BINOMIAL.name,
            "uniform_factor": uniform_names,
            "beta_factor": _BETA_PD_GLOBAL,
        }
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_stress_regime_bernoulli(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    stress_prob = _require(params, "stress_prob", _DEFAULT_STRESS_PROB)
    stress_mult = _require(params, "stress_pd_multiplier", _DEFAULT_STRESS_MULT)
    if not (0.0 <= stress_prob <= 1.0):
        raise ValueError(f"stress_regime: stress_prob={stress_prob} must be in [0, 1].")
    if stress_mult < 1.0:
        raise ValueError(
            f"stress_regime: stress_pd_multiplier={stress_mult} must be >= 1."
        )

    uniform_names = [f"U_{int(u)}" for u in uids]
    universe = _universe(
        [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
        + [_FactorRow(name=_STRESS_GLOBAL, kind="STRESS_INDICATOR")]
    )
    frame = pd.DataFrame(
        {
            "uid": uids,
            "instrument": JtdModel.STRESS_REGIME_BERNOULLI.name,
            "uniform_factor": uniform_names,
            "stress_regime_factor": _STRESS_GLOBAL,
            "stress_pd_multiplier": np.full(uids.shape[0], stress_mult, np.float64),
        }
    )
    stress_p_by_uix = {int(u): stress_prob for u in universe.stress_indicator_uix}
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
        stress_p_by_uix=stress_p_by_uix,
    )


def _assemble_sf_gaussian(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    rho = _require(params, "rho", _DEFAULT_RHO)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"sf_gaussian: rho={rho} must be in [0, 1).")
    beta = math.sqrt(rho)  # systemic loading; idiosyncratic weight = sqrt(1 - rho)

    idio_names = [f"IDIO_{int(u)}" for u in uids]
    universe = _universe(
        [_FactorRow(name=_SYSTEMIC_GLOBAL, kind="SYSTEMIC")]
        + [_FactorRow(name=n, kind="IDIOSYNCRATIC") for n in idio_names]
    )
    frame = pd.DataFrame(
        {
            "uid": uids,
            "instrument": JtdModel.COPULA_SF_GAUSSIAN.name,
            "factor_loading": np.full(uids.shape[0], beta, dtype=np.float64),
            "systemic_factor": _SYSTEMIC_GLOBAL,
            "idiosyncratic_factor": idio_names,
        }
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_sf_students_t(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    rho = _require(params, "rho", _DEFAULT_RHO)
    nu = _require(params, "nu", _DEFAULT_NU)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"sf_students_t: rho={rho} must be in [0, 1).")
    if nu <= 0.0:
        raise ValueError(f"sf_students_t: nu={nu} must be > 0.")
    beta = math.sqrt(rho)

    idio_names = [f"IDIO_{int(u)}" for u in uids]
    universe = _universe(
        [_FactorRow(name=_SYSTEMIC_GLOBAL, kind="SYSTEMIC")]
        + [_FactorRow(name=n, kind="IDIOSYNCRATIC") for n in idio_names]
        + [_FactorRow(name=_CHI_GLOBAL, kind="CHI", nu=nu)]
    )
    frame = pd.DataFrame(
        {
            "uid": uids,
            "instrument": JtdModel.COPULA_SF_STUDENTS_T.name,
            "factor_loading": np.full(uids.shape[0], beta, dtype=np.float64),
            "systemic_factor": _SYSTEMIC_GLOBAL,
            "idiosyncratic_factor": idio_names,
            "chi_squared_factor": _CHI_GLOBAL,
            # nu on the frame must match the chi factor's nu (engine invariant).
            "nu": np.full(uids.shape[0], nu, dtype=np.float64),
        }
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _mf_universe_and_bundle(
    obligor_table: pd.DataFrame,
    loadings_table: pd.DataFrame | None,
    *,
    instrument: str,
    with_chi: bool,
    nu: float | None = None,
) -> tuple[CompiledCreditFactorUniverse, JtdIngestBundle]:
    """Shared MF build: distinct systemic factors from the loadings relation,
    per-obligor idiosyncratic factors, an optional shared CHI factor, the engine
    long-format loadings frame, and the per-obligor dispatch frame.

    Numeric validation (coverage, no-duplicate, sum-beta^2 <= 1, loading range,
    chi/nu reconciliation) is the engine's at compile; this only shapes inputs.
    """
    if loadings_table is None or len(loadings_table) == 0:
        raise ValueError(
            f"{instrument}: a multi-factor model requires a non-empty loadings table."
        )
    uids = _uids(obligor_table)
    idio_names = [f"IDIO_{int(u)}" for u in uids]
    # Distinct systemic factors, authoring order preserved.
    systemic_names = list(dict.fromkeys(loadings_table["factor"].astype(str)))

    rows = [_FactorRow(name=s, kind="SYSTEMIC") for s in systemic_names]
    rows += [_FactorRow(name=n, kind="IDIOSYNCRATIC") for n in idio_names]
    if with_chi:
        rows.append(_FactorRow(name=_CHI_GLOBAL, kind="CHI", nu=nu))
    universe = _universe(rows)

    mf_loadings_df = pd.DataFrame(
        {
            "obligor_uid": loadings_table["uid"].to_numpy(dtype=np.int64),
            "factor_name": loadings_table["factor"].astype(str).to_numpy(),
            "factor_loading": loadings_table["loading"].to_numpy(dtype=np.float64),
        }
    )

    frame_data: dict[str, object] = {
        "uid": uids,
        "instrument": instrument,
        "idiosyncratic_factor": idio_names,
    }
    if with_chi:
        frame_data["chi_squared_factor"] = _CHI_GLOBAL
        # frame nu must match the chi factor's nu (engine reconciliation).
        frame_data["nu"] = np.full(uids.shape[0], nu, dtype=np.float64)
    bundle = JtdIngestBundle.from_instrument_frame(
        pd.DataFrame(frame_data), mf_loadings_df=mf_loadings_df
    )
    return universe, bundle


def _assemble_mf_gaussian(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    universe, bundle = _mf_universe_and_bundle(
        obligor_table,
        loadings_table,
        instrument=JtdModel.COPULA_MF_GAUSSIAN.name,
        with_chi=False,
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=bundle,
        obligor_ead=_ead(obligor_table),
    )


def _assemble_mf_students_t(
    obligor_table: pd.DataFrame,
    params: Mapping[str, float],
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    nu = _require(params, "nu", _DEFAULT_NU)
    if nu <= 0.0:
        raise ValueError(f"mf_students_t: nu={nu} must be > 0.")
    universe, bundle = _mf_universe_and_bundle(
        obligor_table,
        loadings_table,
        instrument=JtdModel.COPULA_MF_STUDENTS_T.name,
        with_chi=True,
        nu=nu,
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=bundle,
        obligor_ead=_ead(obligor_table),
    )


# Recipe key -> assembler. The key is the stable model identifier the solution
# dispatches on; the catalog mirrors it (no-drift guard).
SUPPORTED_RECIPES: dict[str, object] = {
    "forced_default": _assemble_forced_default,
    "independent_bernoulli": _assemble_independent_bernoulli,
    "beta_binomial": _assemble_beta_binomial,
    "stress_regime_bernoulli": _assemble_stress_regime_bernoulli,
    "sf_gaussian": _assemble_sf_gaussian,
    "sf_students_t": _assemble_sf_students_t,
    "mf_gaussian": _assemble_mf_gaussian,
    "mf_students_t": _assemble_mf_students_t,
}


def assemble_inputs(
    *,
    model_key: str,
    obligor_table: pd.DataFrame,
    params: Mapping[str, float] | None = None,
    loadings_table: pd.DataFrame | None = None,
) -> AssembledInputs:
    """Build compiled engine inputs for `model_key` from the obligor table + params.

    Raises ValueError on an unknown/unsupported recipe; each assembler fail-fast
    validates its own params, and the obligor columns flow through the master / EAD
    compilers' validation.
    """
    assembler = SUPPORTED_RECIPES.get(model_key)
    if assembler is None:
        raise ValueError(
            f"assemble_inputs: no recipe for model '{model_key}'. "
            f"Supported: {sorted(SUPPORTED_RECIPES)}."
        )
    return assembler(obligor_table, params or {}, loadings_table)  # type: ignore[operator]
