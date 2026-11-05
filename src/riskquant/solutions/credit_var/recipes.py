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
belongs with the solution. ``pl.DataFrame`` + params in, engine objects out,
fail-fast. Adding a flat-sheet model is one assembler here plus a catalog entry.

Frames (WS_CREDIT_POLARS C.10): ``obligor_table`` and ``loadings_table`` are
``pl.DataFrame``; any other type raises TypeError before a column is read. Every
frame built here (the factor universe, the per-model dispatch frame, the engine
loadings relation) is built column by column with a stated dtype, never by
schema inference from the first rows (F-23).

Multi-factor copulas are deliberately absent: their systemic loadings are a
relational input (a loadings sheet), not a scalar, and are handled by a separate
relational recipe path.
"""

from __future__ import annotations

from riskquant.basel.irb.correlation import basel_corporate_correlation

__all__ = [
    "CHI_FACTOR_NAME",
    "SUPPORTED_RECIPES",
    "AssembledInputs",
    "assemble_inputs",
    "idiosyncratic_factor_name",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.credit.exposure.add_on.ingest import (
    _float_column,
    _int_column,
    _require_polars_frame,
)
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
from riskquant.solutions.credit_var.constants import (
    SF_BETA_BINOMIAL_ALPHA,
    SF_BETA_BINOMIAL_BETA,
    SF_MO_SYSTEMIC_PD,
    SF_NU,
    SF_RHO,
    SF_STRESS_REGIME_MULT,
    SF_STRESS_REGIME_PROB,
)

_SYSTEMIC_GLOBAL = "SYSTEMIC_GLOBAL"
_BETA_PD_GLOBAL = "BETA_PD_GLOBAL"
_STRESS_GLOBAL = "STRESS_GLOBAL"
_CHI_GLOBAL = "CHI_GLOBAL"
#: Name of the shared chi-square factor that the Student-t recipes add to the
#: factor universe. Public so that the request rule on reserved loadings factor
#: names (WS_CREDIT_VAR_API C.17, A_CONTRACT section 2.5 rule 15) reads the
#: recipe's own name, not a copy.
CHI_FACTOR_NAME: Final[str] = _CHI_GLOBAL
# forced-default reads no factor; the generator needs >=1
_FORCED_DUMMY = "FORCED_DUMMY"
_SYSTEMIC_SHOCK = "SYSTEMIC_SHOCK"  # Marshall-Olkin shared common-shock (UNIFORM)

_DEFAULT_MO_SYSTEMIC_PD = SF_MO_SYSTEMIC_PD

_DEFAULT_RHO = SF_RHO
_DEFAULT_NU = SF_NU
_DEFAULT_BETA_ALPHA = SF_BETA_BINOMIAL_ALPHA
_DEFAULT_BETA_BETA = SF_BETA_BINOMIAL_BETA
_DEFAULT_STRESS_PROB = SF_STRESS_REGIME_PROB
_DEFAULT_STRESS_MULT = SF_STRESS_REGIME_MULT

_ENTRY = "assemble_inputs"


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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


def idiosyncratic_factor_name(obligor_uid: int) -> str:
    """Name of the idiosyncratic factor that the multi-factor recipes add for
    one obligor.

    The one owner of the format: `_mf_universe_and_bundle` builds its names
    with it, and the request rule on reserved loadings factor names
    (WS_CREDIT_VAR_API C.17) reads the same format, so the two cannot drift.
    """
    return f"IDIO_{int(obligor_uid):03d}"


def _uids(obligor_table: pl.DataFrame) -> npt.NDArray[np.int64]:
    return _int_column(obligor_table, "uid", np.int64, entry=_ENTRY)


def _str_col(values: list[str]) -> pl.Series:
    return pl.Series(values=values, dtype=pl.String)


def _dispatch_frame(
    uids: npt.NDArray[np.int64],
    instrument: str,
    *,
    strings: Mapping[str, str | list[str]] | None = None,
    floats: Mapping[str, npt.NDArray[np.float64]] | None = None,
) -> pl.DataFrame:
    """The per-model dispatch frame, column by column with stated dtypes.

    ``uid`` is Int64; ``instrument`` and every factor-name column are String (a
    str value is repeated for every row); every value column is Float64.
    """
    n = uids.shape[0]
    columns: dict[str, pl.Series] = {
        "uid": pl.Series(values=uids, dtype=pl.Int64),
        "instrument": _str_col([instrument] * n),
    }
    for name, value in (strings or {}).items():
        columns[name] = _str_col([value] * n if isinstance(value, str) else value)
    for name, arr in (floats or {}).items():
        columns[name] = pl.Series(values=arr, dtype=pl.Float64)
    return pl.DataFrame(columns)


def _universe(rows: list[_FactorRow]) -> CompiledCreditFactorUniverse:
    """Compile factor rows into a universe, emitting the nu / alpha / beta columns
    only when a kind that needs them is present (the ingest maps null elsewhere to
    its sentinels)."""
    columns: dict[str, pl.Series] = {
        "factor_id": pl.Series(values=np.arange(len(rows), dtype=np.int64)),
        "factor_name": _str_col([r.name for r in rows]),
        "factor_kind": _str_col([r.kind for r in rows]),
    }
    if any(r.nu is not None for r in rows):
        columns["nu"] = pl.Series(values=[r.nu for r in rows], dtype=pl.Float64)
    if any(r.alpha is not None for r in rows):
        columns["alpha"] = pl.Series(values=[r.alpha for r in rows], dtype=pl.Float64)
    if any(r.beta is not None for r in rows):
        columns["beta"] = pl.Series(values=[r.beta for r in rows], dtype=pl.Float64)
    return compile_credit_factor_universe(pl.DataFrame(columns))


def _masters(
    obligor_table: pl.DataFrame, universe: CompiledCreditFactorUniverse
) -> CompiledCreditMasters:
    # compile_obligor_master reads uid/pd/lgd and ignores the rest (e.g. ead).
    return CompiledCreditMasters(
        obligor_master=compile_obligor_master(obligor_table),
        factor_universe=universe,
    )


def _ead(obligor_table: pl.DataFrame) -> CompiledObligorEad:
    return compile_obligor_ead(obligor_table)


def _require(params: Mapping[str, float], key: str, default: float) -> float:
    return float(params.get(key, default))


# --------------------------------------------------------------------------
# Per-model assemblers
# --------------------------------------------------------------------------


def _assemble_forced_default(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    # Forced default reads no factor, but the generator needs a non-empty universe.
    universe = _universe([_FactorRow(name=_FORCED_DUMMY, kind="UNIFORM")])
    frame = _dispatch_frame(uids, JtdModel.FORCED_DEFAULT.name)
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_independent_bernoulli(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    uniform_names = [f"U_{int(u):03d}" for u in uids]
    universe = _universe([_FactorRow(name=n, kind="UNIFORM") for n in uniform_names])
    frame = _dispatch_frame(
        uids,
        JtdModel.INDEPENDENT_BERNOULLI.name,
        strings={"uniform_factor": uniform_names},
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_beta_binomial(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    uids = _uids(obligor_table)
    alpha = _require(params, "alpha", _DEFAULT_BETA_ALPHA)
    beta = _require(params, "beta", _DEFAULT_BETA_BETA)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"beta_binomial: alpha={alpha}, beta={beta} must be > 0.")

    uniform_names = [f"U_{int(u):03d}" for u in uids]
    universe = _universe(
        [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
        + [_FactorRow(name=_BETA_PD_GLOBAL, kind="BETA_PD", alpha=alpha, beta=beta)]
    )
    frame = _dispatch_frame(
        uids,
        JtdModel.BETA_BINOMIAL.name,
        strings={"uniform_factor": uniform_names, "beta_factor": _BETA_PD_GLOBAL},
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_stress_regime_bernoulli(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
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

    uniform_names = [f"U_{int(u):03d}" for u in uids]
    universe = _universe(
        [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
        + [_FactorRow(name=_STRESS_GLOBAL, kind="STRESS_INDICATOR")]
    )
    frame = _dispatch_frame(
        uids,
        JtdModel.STRESS_REGIME_BERNOULLI.name,
        strings={
            "uniform_factor": uniform_names,
            "stress_regime_factor": _STRESS_GLOBAL,
        },
        floats={
            "stress_pd_multiplier": np.full(uids.shape[0], stress_mult, np.float64)
        },
    )
    stress_p_by_uix = {int(u): stress_prob for u in universe.stress_indicator_uix}
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
        stress_p_by_uix=stress_p_by_uix,
    )


def _per_obligor_rho(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    *,
    regulatory: bool,
    ctx: str,
) -> np.ndarray:
    """Per-obligor asset correlation rho_i in [0, 1).

    An explicit obligor `rho` (the `rho` column, where present) always wins; the
    fill for the rest is the Basel R(PD_i) when `regulatory`, else the flat global
    `rho` param. This is the ASRF-style per-name correlation with a global default.
    """
    n = len(obligor_table)
    if regulatory:
        pd_vals = np.asarray(
            obligor_table.get_column("pd").cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        rho = basel_corporate_correlation(pd_vals)
    else:
        g = _require(params, "rho", _DEFAULT_RHO)
        if not (0.0 <= g < 1.0):
            raise ValueError(f"{ctx}: global rho={g} must be in [0, 1).")
        rho = np.full(n, g, dtype=np.float64)

    if "rho" in obligor_table.columns:
        col = obligor_table.get_column("rho")
        # A pl.Null column means no explicit rho for any row; a null cell means
        # no explicit rho for that obligor (B-5, C.10).
        if col.dtype != pl.Null:
            if not (col.dtype.is_integer() or col.dtype.is_float()):
                raise TypeError(
                    f"{ctx}: column 'rho' has dtype {col.dtype!s}; it admits an "
                    "integer or a float dtype, or pl.Null."
                )
            explicit_mask = ~col.is_null().to_numpy()
            if explicit_mask.any():
                explicit = np.asarray(col.cast(pl.Float64).to_numpy(), dtype=np.float64)
                rho = np.where(explicit_mask, explicit, rho)

    if not np.all((rho >= 0.0) & (rho < 1.0)):
        bad = sorted({float(x) for x in rho if not (0.0 <= x < 1.0)})
        raise ValueError(f"{ctx}: per-obligor rho out of [0, 1): {bad[:5]}.")
    return rho


def _assemble_sf(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    *,
    instrument: str,
    with_chi: bool,
    regulatory: bool,
    ctx: str,
) -> AssembledInputs:
    """Single-factor copula (Gaussian or Student-t), per-obligor systemic loading
    sqrt(rho_i). One global systemic factor; idiosyncratic weight = sqrt(1 - rho_i)
    is the engine residual."""
    uids = _uids(obligor_table)
    rho = _per_obligor_rho(obligor_table, params, regulatory=regulatory, ctx=ctx)
    loading = np.sqrt(rho)

    nu: float | None = None
    if with_chi:
        nu = _require(params, "nu", _DEFAULT_NU)
        if nu <= 0.0:
            raise ValueError(f"{ctx}: nu={nu} must be > 0.")

    idio_names = [f"IDIO_{int(u):03d}" for u in uids]
    rows = [_FactorRow(name=_SYSTEMIC_GLOBAL, kind="SYSTEMIC")]
    rows += [_FactorRow(name=n, kind="IDIOSYNCRATIC") for n in idio_names]
    if with_chi:
        rows.append(_FactorRow(name=_CHI_GLOBAL, kind="CHI", nu=nu))
    universe = _universe(rows)

    strings: dict[str, str | list[str]] = {
        "systemic_factor": _SYSTEMIC_GLOBAL,
        "idiosyncratic_factor": idio_names,
    }
    floats: dict[str, npt.NDArray[np.float64]] = {"factor_loading": loading}
    if with_chi:
        strings["chi_squared_factor"] = _CHI_GLOBAL
        # nu on the frame must match the chi factor's nu (engine invariant).
        floats["nu"] = np.full(uids.shape[0], nu, dtype=np.float64)
    frame = _dispatch_frame(uids, instrument, strings=strings, floats=floats)
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_sf_gaussian(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    return _assemble_sf(
        obligor_table,
        params,
        instrument=JtdModel.COPULA_SF_GAUSSIAN.name,
        with_chi=False,
        regulatory=False,
        ctx="sf_gaussian",
    )


def _assemble_sf_gaussian_basel(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    return _assemble_sf(
        obligor_table,
        params,
        instrument=JtdModel.COPULA_SF_GAUSSIAN.name,
        with_chi=False,
        regulatory=True,
        ctx="sf_gaussian_basel",
    )


def _assemble_sf_students_t(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    return _assemble_sf(
        obligor_table,
        params,
        instrument=JtdModel.COPULA_SF_STUDENTS_T.name,
        with_chi=True,
        regulatory=False,
        ctx="sf_students_t",
    )


def _assemble_sf_students_t_basel(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    return _assemble_sf(
        obligor_table,
        params,
        instrument=JtdModel.COPULA_SF_STUDENTS_T.name,
        with_chi=True,
        regulatory=True,
        ctx="sf_students_t_basel",
    )


def _mf_universe_and_bundle(
    obligor_table: pl.DataFrame,
    loadings_table: pl.DataFrame | None,
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
    if loadings_table is None or loadings_table.height == 0:
        raise ValueError(
            f"{instrument}: a multi-factor model requires a non-empty loadings table."
        )
    uids = _uids(obligor_table)
    idio_names = [idiosyncratic_factor_name(int(u)) for u in uids]
    factor_col = loadings_table.get_column("factor")
    if factor_col.dtype != pl.String:
        raise TypeError(
            f"{instrument}: loadings column 'factor' has dtype {factor_col.dtype!s}; "
            "it admits pl.Utf8."
        )
    if factor_col.null_count():
        raise ValueError(f"{instrument}: null in loadings column 'factor'.")
    factor_names = factor_col.to_list()
    # Distinct systemic factors, authoring order preserved (first seen).
    systemic_names = list(dict.fromkeys(factor_names))

    rows = [_FactorRow(name=s, kind="SYSTEMIC") for s in systemic_names]
    rows += [_FactorRow(name=n, kind="IDIOSYNCRATIC") for n in idio_names]
    if with_chi:
        rows.append(_FactorRow(name=_CHI_GLOBAL, kind="CHI", nu=nu))
    universe = _universe(rows)

    # The engine loadings relation keeps the row order of the table (F-6).
    mf_loadings_df = pl.DataFrame({
        "obligor_uid": pl.Series(
            values=_int_column(loadings_table, "uid", np.int64, entry=_ENTRY),
            dtype=pl.Int64,
        ),
        "factor_name": _str_col(factor_names),
        "factor_loading": pl.Series(
            values=_float_column(loadings_table, "loading", entry=_ENTRY),
            dtype=pl.Float64,
        ),
    })

    strings: dict[str, str | list[str]] = {"idiosyncratic_factor": idio_names}
    floats: dict[str, npt.NDArray[np.float64]] = {}
    if with_chi:
        strings["chi_squared_factor"] = _CHI_GLOBAL
        # frame nu must match the chi factor's nu (engine reconciliation).
        floats["nu"] = np.full(uids.shape[0], nu, dtype=np.float64)
    bundle = JtdIngestBundle.from_instrument_frame(
        _dispatch_frame(uids, instrument, strings=strings, floats=floats),
        mf_loadings_df=mf_loadings_df,
    )
    return universe, bundle


def _assemble_mf_gaussian(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
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
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
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


def _groups(obligor_table: pl.DataFrame, *, ctx: str) -> list[str]:
    """The ``group`` column of a grouped model as a list of str (C.10).

    Admits ``pl.Utf8`` or ``pl.Null``. A null (or a ``pl.Null`` column) means
    no group; a grouped model needs a group for every obligor, so it raises
    ValueError (the request rules reject it earlier, at
    ``("obligors", i, "group")``). The pandas ``astype(str)`` path read a
    missing group as the text "None".
    """
    col = obligor_table.get_column("group")
    if col.dtype not in (pl.String, pl.Null):
        raise TypeError(
            f"{ctx}: column 'group' has dtype {col.dtype!s}; it admits pl.Utf8 "
            "or pl.Null."
        )
    if col.null_count():
        uids = _uids(obligor_table)[col.is_null().to_numpy()]
        raise ValueError(
            f"{ctx}: obligor(s) {uids[:5].tolist()} have no group; a grouped "
            "model needs a group for every obligor."
        )
    return [str(g) for g in col.to_list()]


def _assemble_beta_binomial_grouped(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    """Segmented beta-binomial: one BETA_PD factor per obligor group, each with its
    own (alpha, beta). Obligors point at their group's factor (correlated within a
    group, independent across groups)."""
    if "group" not in obligor_table.columns:
        raise ValueError("beta_binomial_grouped: obligor_table needs a 'group' column.")
    if not group_params:
        raise ValueError("beta_binomial_grouped: per-group params are required.")
    uids = _uids(obligor_table)
    groups = _groups(obligor_table, ctx="beta_binomial_grouped")
    distinct = list(dict.fromkeys(groups))  # authoring order

    uniform_names = [f"U_{int(u):03d}" for u in uids]
    beta_factor = {g: f"BETA_PD::{g}" for g in distinct}
    rows = [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
    for g in distinct:
        gp = group_params.get(g)
        if gp is None:
            raise ValueError(f"beta_binomial_grouped: no params for group '{g}'.")
        alpha = float(gp["alpha"])
        beta = float(gp["beta"])
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError(
                f"beta_binomial_grouped: group '{g}' alpha={alpha}, beta={beta} "
                "must be > 0."
            )
        rows.append(
            _FactorRow(name=beta_factor[g], kind="BETA_PD", alpha=alpha, beta=beta)
        )
    universe = _universe(rows)
    frame = _dispatch_frame(
        uids,
        JtdModel.BETA_BINOMIAL.name,
        strings={
            "uniform_factor": uniform_names,
            "beta_factor": [beta_factor[g] for g in groups],
        },
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


def _assemble_stress_regime_bernoulli_grouped(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    """Segmented stress regime: one STRESS_INDICATOR factor per obligor group, each
    with its own stress_prob and stress_pd_multiplier. Regimes are independent
    across groups (block-diagonal); the multiplier is per group, applied to every
    obligor in it."""
    if "group" not in obligor_table.columns:
        raise ValueError(
            "stress_regime_bernoulli_grouped: obligor_table needs a 'group' column."
        )
    if not group_params:
        raise ValueError(
            "stress_regime_bernoulli_grouped: per-group params are required."
        )
    uids = _uids(obligor_table)
    groups = _groups(obligor_table, ctx="stress_regime_bernoulli_grouped")
    distinct = list(dict.fromkeys(groups))

    uniform_names = [f"U_{int(u):03d}" for u in uids]
    stress_factor = {g: f"STRESS::{g}" for g in distinct}
    prob_by_group: dict[str, float] = {}
    mult_by_group: dict[str, float] = {}
    rows = [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
    for g in distinct:
        gp = group_params.get(g)
        if gp is None:
            raise ValueError(
                f"stress_regime_bernoulli_grouped: no params for group '{g}'."
            )
        prob = float(gp["stress_prob"])
        mult = float(gp["stress_pd_multiplier"])
        if not (0.0 <= prob <= 1.0):
            raise ValueError(
                f"stress_regime_bernoulli_grouped: group '{g}' stress_prob={prob} "
                "must be in [0, 1]."
            )
        if mult < 1.0:
            raise ValueError(
                f"stress_regime_bernoulli_grouped: group '{g}' "
                f"stress_pd_multiplier={mult} must be >= 1."
            )
        prob_by_group[g] = prob
        mult_by_group[g] = mult
        rows.append(_FactorRow(name=stress_factor[g], kind="STRESS_INDICATOR"))
    universe = _universe(rows)
    frame = _dispatch_frame(
        uids,
        JtdModel.STRESS_REGIME_BERNOULLI.name,
        strings={
            "uniform_factor": uniform_names,
            "stress_regime_factor": [stress_factor[g] for g in groups],
        },
        floats={
            "stress_pd_multiplier": np.array(
                [mult_by_group[g] for g in groups], dtype=np.float64
            )
        },
    )
    name_to_uix = universe.factor_name_to_uix
    stress_p_by_uix = {
        int(name_to_uix[stress_factor[g]]): prob_by_group[g] for g in distinct
    }
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
        stress_p_by_uix=stress_p_by_uix,
    )


# Recipe key -> assembler. The key is the stable model identifier the solution
# dispatches on; the catalog mirrors it (no-drift guard).
def _assemble_marshall_olkin(
    obligor_table: pl.DataFrame,
    params: Mapping[str, float],
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    """Marshall-Olkin common-shock model.

    Every obligor carries its own idiosyncratic UNIFORM draw plus a single shared
    UNIFORM common shock that fires with probability `systemic_pd`; a firing shock
    defaults the whole cohort at once (deterministic contagion). The builder
    decouples each obligor's marginal PD, and requires PD >= systemic_pd per name.
    """
    uids = _uids(obligor_table)
    systemic_pd = _require(params, "systemic_pd", _DEFAULT_MO_SYSTEMIC_PD)
    if not (0.0 <= systemic_pd < 1.0):
        raise ValueError(
            f"marshall_olkin: systemic_pd={systemic_pd} must be in [0, 1)."
        )

    uniform_names = [f"U_{int(u):03d}" for u in uids]
    universe = _universe(
        [_FactorRow(name=n, kind="UNIFORM") for n in uniform_names]
        + [_FactorRow(name=_SYSTEMIC_SHOCK, kind="UNIFORM")]
    )
    frame = _dispatch_frame(
        uids,
        JtdModel.MARSHALL_OLKIN.name,
        strings={
            "uniform_factor": uniform_names,
            "systemic_uniform_factor": _SYSTEMIC_SHOCK,
        },
        floats={"systemic_pd": np.full(uids.shape[0], systemic_pd, np.float64)},
    )
    return AssembledInputs(
        masters=_masters(obligor_table, universe),
        bundle=JtdIngestBundle.from_instrument_frame(frame),
        obligor_ead=_ead(obligor_table),
    )


# Read-only view (WS_CREDIT_VAR_API C.22, F-16): the recipe table is process
# state that every caller dispatches on, so no caller may add, replace or remove
# an assembler. The literal is the only reference to the backing dict.
SUPPORTED_RECIPES: Final[Mapping[str, object]] = MappingProxyType({
    "forced_default": _assemble_forced_default,
    "independent_bernoulli": _assemble_independent_bernoulli,
    "stress_regime_bernoulli": _assemble_stress_regime_bernoulli,
    "stress_regime_bernoulli_grouped": _assemble_stress_regime_bernoulli_grouped,
    "beta_binomial": _assemble_beta_binomial,
    "beta_binomial_grouped": _assemble_beta_binomial_grouped,
    "sf_gaussian": _assemble_sf_gaussian,
    "sf_gaussian_basel": _assemble_sf_gaussian_basel,
    "sf_students_t": _assemble_sf_students_t,
    "sf_students_t_basel": _assemble_sf_students_t_basel,
    "mf_gaussian": _assemble_mf_gaussian,
    "mf_students_t": _assemble_mf_students_t,
    "marshall_olkin": _assemble_marshall_olkin,
})


def assemble_inputs(
    *,
    model_key: str,
    obligor_table: pl.DataFrame,
    params: Mapping[str, float] | None = None,
    loadings_table: pl.DataFrame | None = None,
    group_params: Mapping[str, Mapping[str, float]] | None = None,
) -> AssembledInputs:
    """Build compiled engine inputs for `model_key` from the obligor table + params.

    ``obligor_table`` is a ``pl.DataFrame`` (columns ``uid``, ``ead``, ``pd``,
    ``lgd``, and ``group`` / ``rho`` where the model reads them);
    ``loadings_table`` is None or a ``pl.DataFrame`` (columns ``uid``,
    ``factor``, ``loading``) for the multi-factor models.

    Raises TypeError for a table of another type, before any column is read;
    ValueError on an unknown/unsupported recipe; each assembler fail-fast
    validates its own params, and the obligor columns flow through the master / EAD
    compilers' validation.
    """
    _require_polars_frame(obligor_table, entry=_ENTRY, arg="obligor_table")
    if loadings_table is not None:
        _require_polars_frame(loadings_table, entry=_ENTRY, arg="loadings_table")
    assembler = SUPPORTED_RECIPES.get(model_key)
    if assembler is None:
        raise ValueError(
            f"assemble_inputs: no recipe for model '{model_key}'. "
            f"Supported: {sorted(SUPPORTED_RECIPES)}."
        )
    return assembler(  # type: ignore[operator, no-any-return]
        obligor_table, params or {}, loadings_table, group_params
    )
