# src/riskquant/simlab/scenarios/core_math/ar1_filter.py
"""
AR(1) residual kernel.

Pipeline (per column)
---------------------
1. First pass — compute returns according to the column's shift type
   and accumulate OLS raw moments.
2. Solve AR(1) coefficients (alpha, beta) with a t-test on beta; drop
   to de-mean only if beta is not significant at 95%.
3. Second pass — overwrite returns with residuals.

DOD properties
--------------
- Zero allocations inside the kernel.
- The first-pass return-formula `if/elif` is split into three
  shift-type variants so the t-loop never sees a branch.
- `math.log` (libm) instead of `np.log` (ufunc) — direct call, faster
  inside the tight loop.
- Parallel across columns; AR(1) state is per-column independent.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import inline_njit, parallel_njit  # type: ignore[import]
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)

_F64_2D_F = nb.float64[::1, :]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]

_T_STAT_CRITICAL_95: float = 1.96
_VAR_X_FLOOR: float = 1e-12


# ============================================================
# Inlined per-shift-type return helpers
# ============================================================


@inline_njit
def _ret_log(curr: float, prev: float) -> float:
    return math.log(curr / prev)


@inline_njit
def _ret_simple(curr: float, prev: float) -> float:
    return (curr / prev) - 1.0


@inline_njit
def _ret_diff(curr: float, prev: float) -> float:
    return curr - prev


# ============================================================
# Per-column kernel (one shift-type variant)
# ============================================================


@inline_njit
def _ar1_one_col(col_vals, col_res, st: int) -> None:
    """
    AR(1)-filter a single column in place.

    `col_res[0]` is set to 0.0 (no return for t=0). Pass-1 fills `col_res[t]`
    with the t-th return and accumulates OLS raw moments. Pass-2 subtracts
    `(alpha + beta * r_{t-1})` to leave residuals in `col_res`.

    The shift-type branch is HOISTED above the two t-loops to keep each
    inner loop straight-line SIMD-friendly.
    """
    t_max = col_vals.shape[0]
    if t_max < 2:
        if t_max == 1:
            col_res[0] = 0.0
        return

    col_res[0] = 0.0

    # ---- Pass 1 — three branch-hoisted variants ----
    prev_val = col_vals[0]
    curr_val = col_vals[1]

    if st == SHIFT_LOG_RETURN_ID:
        r = _ret_log(curr_val, prev_val)
    elif st == SHIFT_SIMPLE_RETURN_ID:
        r = _ret_simple(curr_val, prev_val)
    elif st == SHIFT_DIFFERENCE_ID:
        r = _ret_diff(curr_val, prev_val)
    else:
        # Unknown shift-type: upstream `build_shock_universe` filters
        # NO_SHIFT, so reaching this branch indicates a contract break.
        return

    col_res[1] = r
    prev_r = r
    prev_val = curr_val

    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    sum_yy = 0.0

    if st == SHIFT_LOG_RETURN_ID:
        for t in range(2, t_max):
            curr_val = col_vals[t]
            r = _ret_log(curr_val, prev_val)
            col_res[t] = r
            sum_x += prev_r
            sum_y += r
            sum_xx += prev_r * prev_r
            sum_xy += prev_r * r
            sum_yy += r * r
            prev_r = r
            prev_val = curr_val
    elif st == SHIFT_SIMPLE_RETURN_ID:
        for t in range(2, t_max):
            curr_val = col_vals[t]
            r = _ret_simple(curr_val, prev_val)
            col_res[t] = r
            sum_x += prev_r
            sum_y += r
            sum_xx += prev_r * prev_r
            sum_xy += prev_r * r
            sum_yy += r * r
            prev_r = r
            prev_val = curr_val
    else:  # SHIFT_DIFFERENCE_ID
        for t in range(2, t_max):
            curr_val = col_vals[t]
            r = _ret_diff(curr_val, prev_val)
            col_res[t] = r
            sum_x += prev_r
            sum_y += r
            sum_xx += prev_r * prev_r
            sum_xy += prev_r * r
            sum_yy += r * r
            prev_r = r
            prev_val = curr_val

    # ---- OLS solve ----
    n_obs = t_max - 2
    if n_obs <= 0:
        return

    mean_x = sum_x / n_obs
    mean_y = sum_y / n_obs

    var_x = sum_xx - n_obs * mean_x * mean_x
    cov_xy = sum_xy - n_obs * mean_x * mean_y

    beta = 0.0 if var_x < _VAR_X_FLOOR else cov_xy / var_x

    alpha = mean_y - beta * mean_x

    # ---- t-test on beta ----
    var_y = sum_yy - n_obs * mean_y * mean_y
    rss = var_y - beta * cov_xy
    df = n_obs - 2
    if df < 1:
        df = 1
    sigma2 = rss / df

    if var_x > _VAR_X_FLOOR:
        se_beta = math.sqrt(sigma2 / var_x)
        t_stat = abs(beta / se_beta)
    else:
        t_stat = 0.0

    if t_stat < _T_STAT_CRITICAL_95:
        beta = 0.0
        alpha = mean_y  # de-mean only

    # ---- Pass 2 — overwrite returns with residuals ----
    prev_r = col_res[1]
    col_res[1] = prev_r - alpha
    for t in range(2, t_max):
        curr_r = col_res[t]
        col_res[t] = curr_r - (alpha + beta * prev_r)
        prev_r = curr_r


# ============================================================
# Parallel multi-column entry point
# ============================================================


@parallel_njit(
    nb.void(
        _F64_2D_F,  # values            (T, K_total) F-contig
        _I32_1D,  # cols              (K_req,)
        _I8_1D,  # shift_types       (K_req,)
        _F64_2D_F,  # out_residuals     (T, K_req)   F-contig
    ),
)
def kernel_ar1_residuals(
    values,
    cols,
    shift_types,
    out_residuals,
) -> None:
    """
    AR(1) residuals for K_req columns of a (T, K_total) value matrix,
    written into an aligned (T, K_req) residuals matrix.
    """
    k_cols = cols.shape[0]
    for k in nb.prange(k_cols):
        col = cols[k]
        st = shift_types[k]
        col_vals = values[:, col]
        col_res = out_residuals[:, k]
        _ar1_one_col(col_vals, col_res, st)


# Re-export for typed callers.
__all__ = ["kernel_ar1_residuals"]
