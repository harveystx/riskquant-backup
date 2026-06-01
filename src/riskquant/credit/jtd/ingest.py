# src/riskquant/credit/jtd/ingest.py
"""
Credit-positions DataFrame + obligor master -> CompiledJtdPortfolio.

The credit edge: a human-facing per-obligor JTD frame is grouped by
`JtdModel` and collapsed into obligor-keyed SoA blocks. PD and LGD live
on the obligor master (`CompiledObligorCredit`) and are joined onto the
JTD rows by uid here — they are NEVER carried inline on the JTD frame.
This is what makes the obligor master the single source of truth for
obligor-level credit attributes.

Strings (factor names) die here, mapped to int32 indices into the
`CreditFactorMatrix` UIX space via vectorized ``pd.Series.map``. EAD is
intentionally absent — it is realized per scenario by the exposure
engine and bridged in at `run_jtd` time.

Single-factor models compile from one row per obligor. Multi-factor
models would carry multiple rows per obligor (one per loading) once
implemented; the compiler currently expects exactly one row per obligor.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtri

from riskquant.constants import PD_DUST_TOLERANCE
from riskquant.credit._uid_join import searchsorted_or_raise
from riskquant.credit.jtd.blocks.copula import (
    GaussianCopulaJtdBlock,
    TCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.deterministic import DeterministicJtdBlock
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.obligors.compiled import CompiledObligorCredit

_PREVIEW = 10


# ---------------------------------------------------------------------
# Vectorized primitives
# ---------------------------------------------------------------------


def _resolve_factor_uix(
    names: npt.NDArray[np.object_],
    factor_name_to_uix: dict[str, int],
    *,
    ctx: str,
) -> npt.NDArray[np.int32]:

    out = np.empty(names.shape, dtype=np.int32)
    mapping_get = factor_name_to_uix.get

    missing = set()
    sentinel = -999  # must not collide with valid UIX

    for i, name in enumerate(names):
        val = mapping_get(name, sentinel)
        if val == sentinel:
            missing.add(name)
        else:
            out[i] = val

    if missing:
        raise ValueError(f"Unknown credit factor(s) for '{ctx}': {sorted(missing)}")

    return out


def _gaussian_thresholds(pd_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """t_i = Phi^{-1}(PD_i) with dust-clipping to keep ndtri finite."""
    out = np.empty_like(pd_arr)
    np.clip(pd_arr, PD_DUST_TOLERANCE, 1.0 - PD_DUST_TOLERANCE, out=out)
    return ndtri(out, out=out)


def _i64(df: pd.DataFrame, col: str) -> npt.NDArray[np.int64]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int64))


def _f64(df: pd.DataFrame, col: str) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.float64))


def _join_pd_lgd_from_master(
    obligor_uids: npt.NDArray[np.int64],
    obligor_master: CompiledObligorCredit,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Look up PD and LGD on the obligor master in one ``searchsorted``.

    Delegates the actual lookup to the centralized
    ``_uid_join.searchsorted_or_raise`` — every credit-package join goes
    through the same primitive, never inline searchsorted/clip/match.
    """
    pos = searchsorted_or_raise(
        target=obligor_uids,
        superset_sorted=obligor_master.uid,
        ctx="compile_jtd_portfolio: JTD obligor uid(s)",
    )
    pd_arr = np.ascontiguousarray(obligor_master.pd[pos], dtype=np.float64)
    lgd_arr = np.ascontiguousarray(obligor_master.lgd[pos], dtype=np.float64)
    return pd_arr, lgd_arr


# ---------------------------------------------------------------------
# Per-model field validators
# ---------------------------------------------------------------------


def _validate_beta_range(
    obligor_uids: npt.NDArray[np.int64], beta: npt.NDArray[np.float64]
) -> None:
    bad = ~((beta >= -1.0) & (beta <= 1.0))
    if bad.any():
        raise ValueError(
            f"compile_jtd_portfolio: beta outside [-1, 1] for obligor(s) "
            f"{obligor_uids[bad][:_PREVIEW].tolist()}."
        )


def _validate_nu_positive(
    obligor_uids: npt.NDArray[np.int64], nu: npt.NDArray[np.float64]
) -> None:
    bad = ~(nu > 0.0)  # also catches NaN
    if bad.any():
        raise ValueError(
            f"compile_jtd_portfolio: nu must be > 0 (finite) for obligor(s) "
            f"{obligor_uids[bad][:_PREVIEW].tolist()}."
        )


# ---------------------------------------------------------------------
# Per-model block builders
# ---------------------------------------------------------------------


def _build_gaussian_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> GaussianCopulaJtdBlock:
    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    beta = _f64(df, "beta")
    _validate_beta_range(obligor_uids, beta)

    return GaussianCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_gaussian_thresholds(pd_arr),
        beta=beta,
        systemic_uix=_resolve_factor_uix(
            df["systemic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="systemic_factor",
        ),
        idiosyncratic_uix=_resolve_factor_uix(
            df["idiosyncratic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="idiosyncratic_factor",
        ),
    )


def _build_t_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> TCopulaJtdBlock:
    if "nu" not in df.columns:
        raise ValueError("COPULA_SF_T: missing required column 'nu'.")
    if df["nu"].isna().any():
        bad = df.loc[df["nu"].isna(), "uid"].tolist()[:_PREVIEW]
        raise ValueError(f"COPULA_SF_T: nu missing for obligor(s) {bad}")

    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)

    beta = _f64(df, "beta")
    _validate_beta_range(obligor_uids, beta)

    nu = _f64(df, "nu")
    _validate_nu_positive(obligor_uids, nu)

    return TCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_gaussian_thresholds(pd_arr),
        beta=beta,
        systemic_uix=_resolve_factor_uix(
            df["systemic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="systemic_factor",
        ),
        idiosyncratic_uix=_resolve_factor_uix(
            df["idiosyncratic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="idiosyncratic_factor",
        ),
        chi_squared_uix=_resolve_factor_uix(
            df["chi_squared_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="chi_squared_factor",
        ),
        nu=nu,
    )


def _build_deterministic_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
) -> DeterministicJtdBlock:
    """
    Deterministic JTD: loss = ead * lgd, default forced to 1. The kernel
    never reads PD, so the master join is for LGD only.
    """
    obligor_uids = _i64(df, "uid")
    _, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    return DeterministicJtdBlock(obligor_uids=obligor_uids, lgd=lgd_arr)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def compile_jtd_portfolio_from_df(
    *,
    credit_df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> CompiledJtdPortfolio:
    """
    Compile a validated credit-positions frame into a ``CompiledJtdPortfolio``.

    Parameters
    ----------
    credit_df
        One row per obligor. Carries ``uid``, ``instrument`` (a ``JtdModel``
        member name), plus model-specific fields. Single-factor copula
        rows carry ``beta``, ``systemic_factor``, ``idiosyncratic_factor``;
        t-copula adds ``chi_squared_factor`` and ``nu``. **PD and LGD are
        NOT on this frame** — they are joined from ``obligor_master``.
    obligor_master
        Compiled obligor master; the single source of truth for PD and
        LGD. ``uid`` ascending sorted. The JTD ``uid`` column must be a
        subset of the master.
    factor_name_to_uix
        Credit-factor name -> int32 UIX into the ``CreditFactorMatrix``
        universe.
    """
    if credit_df.empty:
        raise ValueError("compile_jtd_portfolio: cannot compile an empty frame.")

    if "instrument" not in credit_df.columns:
        raise ValueError("compile_jtd_portfolio: missing required column 'instrument'.")

    if "uid" not in credit_df.columns:
        raise ValueError("compile_jtd_portfolio: missing required column 'uid'.")

    if credit_df["uid"].duplicated().any():
        dups = credit_df.loc[credit_df["uid"].duplicated(), "uid"].tolist()[:_PREVIEW]
        raise ValueError(f"compile_jtd_portfolio: duplicate obligor uid(s): {dups}.")

    # Fail-fast: an unknown instrument name is a typo, NOT a silent drop.
    known_models = {m.name for m in JtdModel}
    present = set(credit_df["instrument"].astype(str).unique().tolist())
    unknown = present - known_models
    if unknown:
        raise ValueError(
            f"compile_jtd_portfolio: unknown JTD model(s) in 'instrument' column: "
            f"{sorted(unknown)}. Known: {sorted(known_models)}."
        )

    # Typed locals preserve each block's static type into the constructor;
    # a `dict[str, object]` + `**fields` unpack collapses everything to
    # `object` and breaks `CompiledJtdPortfolio.__init__`'s overload resolution.
    gaussian_block: GaussianCopulaJtdBlock | None = None
    t_block: TCopulaJtdBlock | None = None
    det_block: DeterministicJtdBlock | None = None

    for model in JtdModel:
        group = credit_df[credit_df["instrument"] == model.name]
        if group.empty:
            continue
        group = group.reset_index(drop=True)

        if model is JtdModel.COPULA_SF_GAUSSIAN:
            gaussian_block = _build_gaussian_block(
                group,
                obligor_master=obligor_master,
                factor_name_to_uix=factor_name_to_uix,
            )
        elif model is JtdModel.COPULA_SF_T:
            t_block = _build_t_block(
                group,
                obligor_master=obligor_master,
                factor_name_to_uix=factor_name_to_uix,
            )
        elif model is JtdModel.DETERMINISTIC:
            det_block = _build_deterministic_block(
                group,
                obligor_master=obligor_master,
            )
        else:
            # Defensive: the unknown-instrument guard above already filters
            # this, but a new enum member added without a builder here is a
            # programmer error that should fail loudly, not silently drop.
            raise NotImplementedError(
                f"compile_jtd_portfolio: no builder wired for JtdModel.{model.name}."
            )

    if gaussian_block is None and t_block is None and det_block is None:
        raise ValueError(
            "compile_jtd_portfolio: no rows matched a known JtdModel "
            f"({[m.name for m in JtdModel]})."
        )

    return CompiledJtdPortfolio(
        gaussian_copula_jtd=gaussian_block,
        t_copula_jtd=t_block,
        deterministic_jtd=det_block,
    )
