# src/riskquant/credit/masters/factor_universe/ingest.py
"""
Credit factor master DataFrame -> `CompiledCreditFactorUniverse`.

The factor edge: a human-facing authoring table is cast to strict dtypes,
sorted by `factor_id` ascending, validated as one self-consistent universe,
and collapsed into a single immutable SoA container.

`factor_id` ascending sorted fixes the identity axis; UIX is the row position
after the sort and is owned by the engine. The same permutation is applied to
every column so the SoA stays aligned row-wise.

Authoring table
---------------
One row per factor. Columns:
    factor_id    : int   — stable identity, unique. Sorted ascending -> UIX.
    factor_name  : str   — human identity obligor ingest references, unique.
    factor_kind  : str   — a `CreditFactorKind` name (SYSTEMIC, CHI, ...).
    nu           : float — OPTIONAL column. Required and > 0 for CHI rows;
                           blank / NA elsewhere. Mapped to the NO_CHI_NU
                           sentinel for non-CHI factors.
    alpha, beta  : float — OPTIONAL columns. Required and > 0 for BETA_PD rows;
                           blank / NA elsewhere. Mapped to the NO_BETA_PARAM
                           sentinel for non-BETA_PD factors.
"""

from __future__ import annotations

import warnings

__all__ = [
    "compile_credit_factor_universe",
]

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.constants import NO_BETA_PARAM, NO_CHI_NU
from riskquant.credit.masters.enums.credit_factor_kind import (
    VALID_CREDIT_FACTOR_KIND_NAMES,
    CreditFactorKind,
)
from riskquant.credit.masters.factor_universe.compiled import (
    CompiledCreditFactorUniverse,
)
from riskquant.credit.masters.factor_universe.validate import (
    validate_credit_factor_universe_arrays,
)

_REQUIRED_COLS: tuple[str, ...] = (
    "factor_id",
    "factor_name",
    "factor_kind",
)
_PREVIEW = 10

# Authored kind string -> int8 id (single source of truth = the enum).
_KIND_NAME_TO_ID: dict[str, np.int8] = {
    k.name: np.int8(k.value) for k in CreditFactorKind
}


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"compile_credit_factor_universe: missing required column(s) {missing}."
        )


def _i32(df: pd.DataFrame, col: str) -> npt.NDArray[np.int32]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int32))


def _names(df: pd.DataFrame) -> npt.NDArray[np.str_]:
    # Force a fixed-width unicode SoA column; .astype(str) first so any
    # non-string authoring (e.g. ints) fails visibly rather than boxing.
    return np.ascontiguousarray(
        np.asarray(df["factor_name"].astype(str).to_numpy(), dtype=np.str_)
    )


def _kind_ids(df: pd.DataFrame) -> npt.NDArray[np.int8]:
    """Map authored kind strings to int8 ids; fail-fast on unknown names."""
    raw = df["factor_kind"].astype(str).to_numpy()
    unknown = sorted(set(raw.tolist()) - VALID_CREDIT_FACTOR_KIND_NAMES)
    if unknown:
        raise ValueError(
            f"compile_credit_factor_universe: unknown factor_kind(s) "
            f"{unknown[:_PREVIEW]}. Valid: {sorted(VALID_CREDIT_FACTOR_KIND_NAMES)}."
        )
    return np.ascontiguousarray(
        np.array([_KIND_NAME_TO_ID[s] for s in raw], dtype=np.int8)
    )


def _chi_nu(
    df: pd.DataFrame, kind_ids: npt.NDArray[np.int8]
) -> npt.NDArray[np.float64]:
    """
    Map genuinely-missing nu to the NO_CHI_NU sentinel, preserving every
    authored value.

    Only NaN (truly absent) becomes the sentinel. A value authored on a non-CHI
    row is kept, NOT silently discarded, so `validate_chi_nu` can reject it as a
    likely kind-mislabel. A missing nu on a CHI row becomes the sentinel and is
    likewise rejected downstream (the sentinel is not > 0), with the offending
    factor id reported.
    """
    n = kind_ids.shape[0]
    if "nu" in df.columns:
        nu_raw = df["nu"].to_numpy(dtype=np.float64)  # NA -> NaN
    else:
        nu_raw = np.full(n, np.nan, dtype=np.float64)

    return np.ascontiguousarray(np.where(np.isnan(nu_raw), NO_CHI_NU, nu_raw))


def _require_param_columns_for_kinds(
    df: pd.DataFrame, factor_kind: npt.NDArray[np.int8]
) -> None:
    """Fail fast if a kind that needs a param column is present but the column
    is absent. Per-row value validity is still enforced downstream by
    `validate_chi_nu` / `validate_beta_params`; this just turns a "forgot the
    column entirely" schema mistake into a crisp, specific error."""
    if np.any(factor_kind == np.int8(CreditFactorKind.CHI)) and "nu" not in df.columns:
        raise ValueError(
            "compile_credit_factor_universe: universe has CHI factor(s) but no "
            "'nu' column. CHI factors require their degrees of freedom."
        )
    if np.any(factor_kind == np.int8(CreditFactorKind.BETA_PD)):
        missing = [c for c in ("alpha", "beta") if c not in df.columns]
        if missing:
            raise ValueError(
                "compile_credit_factor_universe: universe has BETA_PD factor(s) "
                f"but no {missing} column(s). BETA_PD factors require their Beta "
                "shape parameters."
            )


def _beta_param(df: pd.DataFrame, col: str, n: int) -> npt.NDArray[np.float64]:
    """
    Map a genuinely-missing Beta shape parameter to the NO_BETA_PARAM sentinel,
    preserving every authored value.

    Only NaN (truly absent) becomes the sentinel. A value authored on a
    non-BETA_PD row is kept, NOT silently discarded, so `validate_beta_params`
    can reject it as a likely kind-mislabel; a missing value on a BETA_PD row
    becomes the sentinel and is likewise rejected downstream.
    """
    if col in df.columns:
        raw = df[col].to_numpy(dtype=np.float64)  # NA -> NaN
    else:
        raw = np.full(n, np.nan, dtype=np.float64)

    return np.ascontiguousarray(np.where(np.isnan(raw), NO_BETA_PARAM, raw))


def compile_credit_factor_universe(
    factor_df: pd.DataFrame,
) -> CompiledCreditFactorUniverse:
    """
    Compile a validated credit factor-universe frame into
    `CompiledCreditFactorUniverse`.

    Parameters
    ----------
    factor_df
        One row per factor. Required columns: factor_id, factor_name,
        factor_kind. Optional `nu` column carries the chi degrees of freedom
        (required for CHI rows, blank otherwise).

    Returns
    -------
    CompiledCreditFactorUniverse
        `factor_ids` sorted ascending; all other columns reordered by the same
        permutation. UIX == row position.
    """
    if factor_df.empty:
        raise ValueError(
            "compile_credit_factor_universe: cannot compile an empty frame."
        )

    _check_required_columns(factor_df)

    # One stable permutation by the identity axis applied to every column.
    df = factor_df.sort_values("factor_id", kind="stable").reset_index(drop=True)

    factor_ids = _i32(df, "factor_id")
    factor_names = _names(df)
    factor_kind = _kind_ids(df)

    # Param columns stay OPTIONAL, but a column is REQUIRED the moment a factor
    # of its kind appears: a CHI factor without a 'nu' column, or a BETA_PD
    # factor without 'alpha'/'beta', is a schema mistake worth a crisp error
    # here rather than the value-level "missing/non-positive" downstream. The
    # structural requirement follows from the content, so simple universes
    # (no chi / no beta) need carry no param columns at all.
    _require_param_columns_for_kinds(df, factor_kind)

    factor_chi_nu = _chi_nu(df, factor_kind)
    n = factor_kind.shape[0]
    factor_beta_alpha = _beta_param(df, "alpha", n)
    factor_beta_beta = _beta_param(df, "beta", n)

    validate_credit_factor_universe_arrays(
        factor_ids=factor_ids,
        factor_names=factor_names,
        factor_kind=factor_kind,
        factor_chi_nu=factor_chi_nu,
        factor_beta_alpha=factor_beta_alpha,
        factor_beta_beta=factor_beta_beta,
    )

    return CompiledCreditFactorUniverse(
        factor_ids=factor_ids,
        factor_names=factor_names,
        factor_kind=factor_kind,
        factor_chi_nu=factor_chi_nu,
        factor_beta_alpha=factor_beta_alpha,
        factor_beta_beta=factor_beta_beta,
    )

