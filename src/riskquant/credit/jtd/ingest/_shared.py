# src/riskquant/credit/jtd/ingest/_shared.py
"""
Shared low-level ingest primitives: factor-name -> UIX resolution, typed column
extraction, Gaussian / Student-t default thresholds, and the obligor-master
PD/LGD join. Used by the per-model builders (and, for resolution, indirectly via
the builders feeding the semantic validators). No model-specific logic here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtri
from scipy.stats import t as scipy_student_t

from riskquant._arrays import searchsorted_or_raise
from riskquant.constants import PD_DUST_TOLERANCE
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit

_PREVIEW = 10


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


def _i64(df: pd.DataFrame, col: str) -> npt.NDArray[np.int64]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int64))


def _f64(df: pd.DataFrame, col: str) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.float64))


def _gaussian_thresholds(pd_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """t_i = Phi^{-1}(PD_i) with dust-clipping to keep ndtri finite."""
    out = np.empty_like(pd_arr)
    np.clip(pd_arr, PD_DUST_TOLERANCE, 1.0 - PD_DUST_TOLERANCE, out=out)
    return ndtri(out, out=out)


def _students_t_thresholds(
    pd_arr: npt.NDArray[np.float64],
    nu: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    t_i = F_t^{-1}(PD_i; nu_i)

    Ensures:
        P(T_i < t_i) = PD_i

    for Student-t latent variables.
    """
    clipped = np.clip(
        pd_arr,
        PD_DUST_TOLERANCE,
        1.0 - PD_DUST_TOLERANCE,
    )

    return np.asarray(
        scipy_student_t.ppf(clipped, df=nu),
        dtype=np.float64,
    )


def _join_pd_lgd_from_master(
    obligor_uids: npt.NDArray[np.int64],
    obligor_master: CompiledObligorCredit,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Look up PD and LGD on the obligor master in one ``searchsorted``.

    Delegates the actual lookup to the centralized
    ``_arrays.searchsorted_or_raise`` — every credit-package join goes
    through the same primitive, never inline searchsorted/clip/match.
    """
    pos = searchsorted_or_raise(
        raw=obligor_uids,
        universe=obligor_master.uid,
        ctx="compile_jtd_portfolio: JTD obligor uid(s)",
    )
    pd_arr = np.ascontiguousarray(obligor_master.pd[pos], dtype=np.float64)
    lgd_arr = np.ascontiguousarray(obligor_master.lgd[pos], dtype=np.float64)
    return pd_arr, lgd_arr
