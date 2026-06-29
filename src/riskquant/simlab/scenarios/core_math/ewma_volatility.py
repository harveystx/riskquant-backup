# src/riskquant/simlab/scenarios/core_math/ewma_volatility.py
"""
EWMA volatility-ratio kernels.

Computes the FHS Z-score series  Z_t = sigma_today / sigma_t  along
the historical T axis, per column. Two entry points:

* `kernel_ewma_vol_ratios`
      Takes raw values and a shift-type per column; computes returns
      inline.

* `kernel_ewma_vol_ratios_from_residuals`
      Takes pre-computed residuals (e.g. AR(1)-filtered) directly.

DOD properties
--------------
- **Zero allocations inside the kernel.** The previous implementation
  did `variances = np.empty(T, dtype=np.float64)` *inside* `nb.prange`
  — one heap burst per column per call. We replace it with a two-pass
  algorithm that re-derives the EWMA variance series on the fly:
  pass 1 carries `var_t` only and ends with `var_today`; pass 2 re-runs
  the EWMA recursion to recover each `var_t` and writes the ratio
  immediately. Trade O(T) extra FMAs for O(T) heap; the FMAs are
  cheap on every modern CPU.
- `math.log` / `math.sqrt` for direct libm dispatch.
- Branch hoisting around the t-loops (one shift-type test per column).
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import inline_njit, parallel_njit
from riskquant.constants import MIN_EWMA_VARIANCE
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)

_F64_2D_F = nb.float64[::1, :]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]


# ============================================================
# Shared one-column ratio pass given a return-emitter
# ============================================================
#
# We parameterise the per-column work by the shift-type id at the
# outer level (in `kernel_ewma_vol_ratios`) and inline three
# per-shift-type passes; for the residuals variant the return IS the
# residual so we have one pass.


@inline_njit
def _ewma_var_today_residuals(col_res, decay: float) -> float:
    """
    Single forward sweep computing var_today only, residuals variant.

    Seeds with var_1 = r_1^2 and runs the EWMA recursion through t=T-1.
    """
    t_max = col_res.shape[0]
    if t_max < 2:
        return 0.0

    r1 = col_res[1]
    var_t = r1 * r1
    one_minus_lam = 1.0 - decay

    for t in range(2, t_max):
        rt = col_res[t]
        var_t = one_minus_lam * (rt * rt) + decay * var_t
    return var_t


@inline_njit
def _emit_ratios_residuals(
    col_res,
    col_out,
    var_today: float,
    decay: float,
    max_vol_ratio: float,
) -> None:
    """
    Second sweep: re-derive var_t per t and emit clamped ratios.
    """
    t_max = col_res.shape[0]
    if t_max == 0:
        return

    if t_max == 1:
        col_out[0] = 1.0
        return

    r1 = col_res[1]
    var_t = r1 * r1
    one_minus_lam = 1.0 - decay

    # Indices t=0 and t=1 share the seed variance, per the original
    # algorithm.
    if var_t < MIN_EWMA_VARIANCE:
        col_out[0] = 1.0
        col_out[1] = 1.0
    else:
        ratio = math.sqrt(var_today / var_t)
        if ratio > max_vol_ratio:
            ratio = max_vol_ratio
        col_out[0] = ratio
        col_out[1] = ratio

    for t in range(2, t_max):
        rt = col_res[t]
        var_t = one_minus_lam * (rt * rt) + decay * var_t
        if var_t < MIN_EWMA_VARIANCE:
            col_out[t] = 1.0
        else:
            ratio = math.sqrt(var_today / var_t)
            if ratio > max_vol_ratio:
                ratio = max_vol_ratio
            col_out[t] = ratio


# ============================================================
# Public — residuals variant
# ============================================================


@parallel_njit(
    nb.void(
        _F64_2D_F,  # residuals    (T, K_req) F-contig
        nb.float64,  # decay_factor
        nb.float64,  # max_vol_ratio
        _F64_2D_F,  # out_ratios   (T, K_req) F-contig
    ),
)
def kernel_ewma_vol_ratios_from_residuals(
    residuals,
    decay_factor,
    max_vol_ratio,
    out_ratios,
) -> None:
    """
    Z-score series from a pre-computed residuals matrix.

    Two-pass per column, zero heap allocation inside `prange`.
    """
    k_cols = residuals.shape[1]
    for k in nb.prange(k_cols):
        col_res = residuals[:, k]
        col_out = out_ratios[:, k]
        var_today = _ewma_var_today_residuals(col_res, decay_factor)
        _emit_ratios_residuals(col_res, col_out, var_today, decay_factor, max_vol_ratio)


# ============================================================
# Public — raw-values variant (computes returns inline)
# ============================================================


@inline_njit
def _ewma_var_today_raw(col_vals, decay: float, st: int) -> float:
    """
    Forward sweep computing var_today from raw values, branch-hoisted
    by shift type.
    """
    t_max = col_vals.shape[0]
    if t_max < 2:
        return 0.0

    one_minus_lam = 1.0 - decay

    if st == SHIFT_LOG_RETURN_ID:
        r1 = math.log(col_vals[1] / col_vals[0])
        var_t = r1 * r1
        for t in range(2, t_max):
            rt = math.log(col_vals[t] / col_vals[t - 1])
            var_t = one_minus_lam * (rt * rt) + decay * var_t
        return var_t

    if st == SHIFT_SIMPLE_RETURN_ID:
        r1 = (col_vals[1] / col_vals[0]) - 1.0
        var_t = r1 * r1
        for t in range(2, t_max):
            rt = (col_vals[t] / col_vals[t - 1]) - 1.0
            var_t = one_minus_lam * (rt * rt) + decay * var_t
        return var_t

    if st == SHIFT_DIFFERENCE_ID:
        r1 = col_vals[1] - col_vals[0]
        var_t = r1 * r1
        for t in range(2, t_max):
            rt = col_vals[t] - col_vals[t - 1]
            var_t = one_minus_lam * (rt * rt) + decay * var_t
        return var_t

    return 0.0


@inline_njit
def _emit_ratios_raw(
    col_vals,
    col_out,
    var_today: float,
    decay: float,
    max_vol_ratio: float,
    st: int,
) -> None:
    """
    Second sweep over raw values, branch-hoisted by shift type.
    """
    t_max = col_vals.shape[0]
    if t_max == 0:
        return
    if t_max == 1:
        col_out[0] = 1.0
        return

    one_minus_lam = 1.0 - decay

    if st == SHIFT_LOG_RETURN_ID:
        r1 = math.log(col_vals[1] / col_vals[0])
        var_t = r1 * r1
        if var_t < MIN_EWMA_VARIANCE:
            col_out[0] = 1.0
            col_out[1] = 1.0
        else:
            ratio = math.sqrt(var_today / var_t)
            if ratio > max_vol_ratio:
                ratio = max_vol_ratio
            col_out[0] = ratio
            col_out[1] = ratio
        for t in range(2, t_max):
            rt = math.log(col_vals[t] / col_vals[t - 1])
            var_t = one_minus_lam * (rt * rt) + decay * var_t
            if var_t < MIN_EWMA_VARIANCE:
                col_out[t] = 1.0
            else:
                ratio = math.sqrt(var_today / var_t)
                if ratio > max_vol_ratio:
                    ratio = max_vol_ratio
                col_out[t] = ratio
        return

    if st == SHIFT_SIMPLE_RETURN_ID:
        r1 = (col_vals[1] / col_vals[0]) - 1.0
        var_t = r1 * r1
        if var_t < MIN_EWMA_VARIANCE:
            col_out[0] = 1.0
            col_out[1] = 1.0
        else:
            ratio = math.sqrt(var_today / var_t)
            if ratio > max_vol_ratio:
                ratio = max_vol_ratio
            col_out[0] = ratio
            col_out[1] = ratio
        for t in range(2, t_max):
            rt = (col_vals[t] / col_vals[t - 1]) - 1.0
            var_t = one_minus_lam * (rt * rt) + decay * var_t
            if var_t < MIN_EWMA_VARIANCE:
                col_out[t] = 1.0
            else:
                ratio = math.sqrt(var_today / var_t)
                if ratio > max_vol_ratio:
                    ratio = max_vol_ratio
                col_out[t] = ratio
        return

    if st == SHIFT_DIFFERENCE_ID:
        r1 = col_vals[1] - col_vals[0]
        var_t = r1 * r1
        if var_t < MIN_EWMA_VARIANCE:
            col_out[0] = 1.0
            col_out[1] = 1.0
        else:
            ratio = math.sqrt(var_today / var_t)
            if ratio > max_vol_ratio:
                ratio = max_vol_ratio
            col_out[0] = ratio
            col_out[1] = ratio
        for t in range(2, t_max):
            rt = col_vals[t] - col_vals[t - 1]
            var_t = one_minus_lam * (rt * rt) + decay * var_t
            if var_t < MIN_EWMA_VARIANCE:
                col_out[t] = 1.0
            else:
                ratio = math.sqrt(var_today / var_t)
                if ratio > max_vol_ratio:
                    ratio = max_vol_ratio
                col_out[t] = ratio
        return


@parallel_njit(
    nb.void(
        _F64_2D_F,  # values        (T, K_total) F-contig
        _I32_1D,  # cols          (K_req,)
        _I8_1D,  # shift_types   (K_req,)
        nb.float64,  # decay_factor
        _F64_2D_F,  # out_ratios    (T, K_req) F-contig
        nb.float64,  # max_vol_ratio
    ),
)
def kernel_ewma_vol_ratios(
    values,
    cols,
    shift_types,
    decay_factor,
    out_ratios,
    max_vol_ratio,
) -> None:
    """
    Z-score series Z_t = sigma_today / sigma_t per column, computed
    directly from raw values with the per-column shift type. Two-pass,
    zero-allocation.
    """
    k_cols = cols.shape[0]
    for k in nb.prange(k_cols):
        col = cols[k]
        st = shift_types[k]
        col_vals = values[:, col]
        col_out = out_ratios[:, k]
        var_today = _ewma_var_today_raw(col_vals, decay_factor, st)
        _emit_ratios_raw(col_vals, col_out, var_today, decay_factor, max_vol_ratio, st)


__all__ = [
    "kernel_ewma_vol_ratios",
    "kernel_ewma_vol_ratios_from_residuals",
]
