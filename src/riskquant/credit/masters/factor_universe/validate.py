# src/riskquant/credit/masters/factor_universe/validate.py
"""
Credit-factor master validators.

Pure vectorized predicates on the post-ingest SoA arrays. Each function
either passes silently or raises `ValueError` with the (bounded) list of
offending factor ids. No row loops, no pandas, no string lookups beyond the
identity axis — these run once at ingest time on arrays already cast to their
compiled dtypes.

Split per invariant (id uniqueness, name uniqueness, kind domain, chi/nu
consistency) so any concrete failure points at exactly one rule.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.constants import NO_BETA_PARAM, NO_CHI_NU
from riskquant.credit.masters.enums.credit_factor_kind import (
    VALID_CREDIT_FACTOR_KIND_IDS,
    CreditFactorKind,
)

_PREVIEW = 10


def _bad_ids(
    factor_ids: npt.NDArray[np.int32], mask: npt.NDArray[np.bool_]
) -> list[int]:
    return factor_ids[mask][:_PREVIEW].tolist()


def validate_lengths_aligned(
    *,
    factor_ids: npt.NDArray[np.int32],
    factor_names: npt.NDArray[np.str_],
    factor_kind: npt.NDArray[np.int8],
    factor_chi_nu: npt.NDArray[np.float64],
    factor_beta_alpha: npt.NDArray[np.float64],
    factor_beta_beta: npt.NDArray[np.float64],
) -> None:
    """All SoA columns share one length F > 0."""
    n = factor_ids.shape[0]
    if n == 0:
        raise ValueError("credit factor universe: empty factor table.")
    lengths = {
        "factor_names": factor_names.shape[0],
        "factor_kind": factor_kind.shape[0],
        "factor_chi_nu": factor_chi_nu.shape[0],
        "factor_beta_alpha": factor_beta_alpha.shape[0],
        "factor_beta_beta": factor_beta_beta.shape[0],
    }
    misaligned = {k: v for k, v in lengths.items() if v != n}
    if misaligned:
        raise ValueError(
            f"credit factor universe: column length mismatch vs factor_ids "
            f"({n}): {misaligned}."
        )


def validate_factor_id_sorted_unique(factor_ids: npt.NDArray[np.int32]) -> None:
    """`factor_id` must be strictly ascending — uniqueness + sorted in one check."""
    if not (np.diff(factor_ids) > 0).all():
        unique, counts = np.unique(factor_ids, return_counts=True)
        dups = unique[counts > 1][:_PREVIEW].tolist()
        if dups:
            raise ValueError(f"credit factor universe: duplicate factor_id(s) {dups}.")
        raise ValueError("credit factor universe: factor_id is not sorted ascending.")


def validate_factor_names(
    factor_ids: npt.NDArray[np.int32],
    factor_names: npt.NDArray[np.str_],
) -> None:
    """Factor names must be non-empty and unique."""
    empty = np.char.str_len(factor_names) == 0
    if empty.any():
        raise ValueError(
            f"credit factor universe: empty factor_name for factor_id(s) "
            f"{_bad_ids(factor_ids, empty)}."
        )
    unique, counts = np.unique(factor_names, return_counts=True)
    dups = unique[counts > 1][:_PREVIEW].tolist()
    if dups:
        raise ValueError(f"credit factor universe: duplicate factor_name(s) {dups}.")


def validate_factor_kind(
    factor_ids: npt.NDArray[np.int32],
    factor_kind: npt.NDArray[np.int8],
) -> None:
    """Every `factor_kind` is a defined `CreditFactorKind` value."""
    valid = np.array(sorted(VALID_CREDIT_FACTOR_KIND_IDS), dtype=np.int8)
    bad = ~np.isin(factor_kind, valid)
    if bad.any():
        raise ValueError(
            f"credit factor universe: unknown factor_kind id(s) "
            f"{np.unique(factor_kind[bad])[:_PREVIEW].tolist()} for factor_id(s) "
            f"{_bad_ids(factor_ids, bad)}."
        )


def validate_chi_nu(
    factor_ids: npt.NDArray[np.int32],
    factor_kind: npt.NDArray[np.int8],
    factor_chi_nu: npt.NDArray[np.float64],
) -> None:
    """
    nu consistency with kind.

    - CHI factors: nu must be finite and strictly positive (it is the dof the
      generator draws ``W ~ chi^2(nu)`` with; a missing or non-positive value
      breaks the t-marginal).
    - Non-CHI factors: nu must be the `NO_CHI_NU` sentinel (never NaN, never a
      stray value that a future consumer might mistake for a real dof).
    """
    is_chi = factor_kind == np.int8(CreditFactorKind.CHI)

    bad_chi = is_chi & ~(np.isfinite(factor_chi_nu) & (factor_chi_nu > 0.0))
    if bad_chi.any():
        raise ValueError(
            f"credit factor universe: CHI factor(s) with missing or non-positive "
            f"nu for factor_id(s) {_bad_ids(factor_ids, bad_chi)}."
        )

    bad_non_chi = (~is_chi) & (factor_chi_nu != NO_CHI_NU)
    if bad_non_chi.any():
        raise ValueError(
            f"credit factor universe: non-CHI factor(s) carry a nu value "
            f"(must be NO_CHI_NU={float(NO_CHI_NU)}) for factor_id(s) "
            f"{_bad_ids(factor_ids, bad_non_chi)}."
        )


def validate_beta_params(
    factor_ids: npt.NDArray[np.int32],
    factor_kind: npt.NDArray[np.int8],
    factor_beta_alpha: npt.NDArray[np.float64],
    factor_beta_beta: npt.NDArray[np.float64],
) -> None:
    """
    Beta(alpha, beta) parameter consistency with kind.

    - BETA_PD factors: both alpha and beta must be finite and strictly positive
      (they are the shape parameters the generator draws ``P_s ~ Beta`` with; a
      missing or non-positive value is an invalid Beta).
    - Non-BETA_PD factors: both must be the `NO_BETA_PARAM` sentinel (never NaN,
      never a stray value a future consumer might mistake for a real parameter).
    """
    is_beta = factor_kind == np.int8(CreditFactorKind.BETA_PD)
    is_good_beta = (
        np.isfinite(factor_beta_alpha)
        & (factor_beta_alpha > 0.0)
        & np.isfinite(factor_beta_beta)
        & (factor_beta_beta > 0.0)
    )

    is_bad_beta = is_beta & ~is_good_beta
    if is_bad_beta.any():
        raise ValueError(
            f"credit factor universe: BETA_PD factor(s) with missing or "
            f"non-positive alpha/beta for factor_id(s) "
            f"{_bad_ids(factor_ids, is_bad_beta)}."
        )

    bad_non_beta = (~is_beta) & (
        (factor_beta_alpha != NO_BETA_PARAM) | (factor_beta_beta != NO_BETA_PARAM)
    )
    if bad_non_beta.any():
        raise ValueError(
            f"credit factor universe: non-BETA_PD factor(s) carry an alpha/beta "
            f"value (must be NO_BETA_PARAM={float(NO_BETA_PARAM)}) for "
            f"factor_id(s) {_bad_ids(factor_ids, bad_non_beta)}."
        )


def validate_credit_factor_universe_arrays(
    *,
    factor_ids: npt.NDArray[np.int32],
    factor_names: npt.NDArray[np.str_],
    factor_kind: npt.NDArray[np.int8],
    factor_chi_nu: npt.NDArray[np.float64],
    factor_beta_alpha: npt.NDArray[np.float64],
    factor_beta_beta: npt.NDArray[np.float64],
) -> None:
    """Single-call entry point: every validator, fail-fast on first violation."""
    validate_lengths_aligned(
        factor_ids=factor_ids,
        factor_names=factor_names,
        factor_kind=factor_kind,
        factor_chi_nu=factor_chi_nu,
        factor_beta_alpha=factor_beta_alpha,
        factor_beta_beta=factor_beta_beta,
    )
    validate_factor_id_sorted_unique(factor_ids)
    validate_factor_names(factor_ids, factor_names)
    validate_factor_kind(factor_ids, factor_kind)
    validate_chi_nu(factor_ids, factor_kind, factor_chi_nu)
    validate_beta_params(factor_ids, factor_kind, factor_beta_alpha, factor_beta_beta)
