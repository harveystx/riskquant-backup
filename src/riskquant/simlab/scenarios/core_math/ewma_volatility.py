# src/riskquant/simlab/scenarios/core_math/ewma_volatility.py


from riskquant.constants import MIN_EWMA_VARIANCE
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE,
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
)


import numba as nb
import numpy as np


# =========================================================================
# EWMA VOLATILITY PRE-COMPILER (DYNAMIC TO SHIFT TYPES)
# =========================================================================


@nb.njit(parallel=True, fastmath=True, cache=True)
def kernel_ewma_vol_ratios(
    values: np.ndarray,  # Fortran-contiguous (T x K_total)
    cols: np.ndarray,  # K_req, column indices into values
    shift_types: np.ndarray,  # K_req
    decay_factor: float,  # lambda (e.g., 0.94)
    out_ratios: np.ndarray,  # Fortran-contiguous (T x K_req)
    max_vol_ratio: float,  # Maximum allowed volatility ratio (e.g., 10.0)
) -> None:
    """
    Computes the FHS Z-score matrix: Z_t = (Sigma_today / Sigma_t)
    dynamically respecting the specific shift type requested for each column.
    """
    T = values.shape[0]
    K = cols.shape[0]

    for k in nb.prange(K):
        col = cols[k]
        st = shift_types[k]

        col_vals = values[:, col]
        col_out_ratios = out_ratios[:, k]

        variances = np.empty(T, dtype=np.float64)

        # 1. Initialize first return based on strict shift type
        if st == SHIFT_LOG_RETURN:
            r1 = np.log(col_vals[1] / col_vals[0])
        elif st == SHIFT_SIMPLE_RETURN:
            r1 = (col_vals[1] / col_vals[0]) - 1.0
        elif st == SHIFT_DIFFERENCE:
            r1 = col_vals[1] - col_vals[0]
        else:
            r1 = 0.0

        var_t = r1 * r1
        variances[0] = var_t
        variances[1] = var_t

        # 2. EWMA loop matching strict shift type
        if st == SHIFT_LOG_RETURN:
            for t in range(2, T):
                rt = np.log(col_vals[t] / col_vals[t - 1])
                var_t = (1.0 - decay_factor) * (rt * rt) + decay_factor * var_t
                variances[t] = var_t

        elif st == SHIFT_SIMPLE_RETURN:
            for t in range(2, T):
                rt = (col_vals[t] / col_vals[t - 1]) - 1.0
                var_t = (1.0 - decay_factor) * (rt * rt) + decay_factor * var_t
                variances[t] = var_t

        elif st == SHIFT_DIFFERENCE:
            for t in range(2, T):
                rt = col_vals[t] - col_vals[t - 1]
                var_t = (1.0 - decay_factor) * (rt * rt) + decay_factor * var_t
                variances[t] = var_t

        # 3. Today's variance is the very last element
        var_today = variances[T - 1]

        # 4. Calculate Z-ratios (Sigma_today / Sigma_t)
        # for t in range(T):
        #     if variances[t] < 1e-12:
        #         col_out_ratios[t] = 1.0
        #     else:
        #         col_out_ratios[t] = np.sqrt(var_today / variances[t])
        for t in range(T):
            if variances[t] < MIN_EWMA_VARIANCE:
                col_out_ratios[t] = 1.0
            else:
                # Calculate raw ratio and clamp it to the max_vol_ratio
                raw_ratio = np.sqrt(var_today / variances[t])
                col_out_ratios[t] = min(raw_ratio, max_vol_ratio)


@nb.njit(parallel=True, fastmath=True, cache=True)
def kernel_ewma_vol_ratios_from_residuals(
    residuals: np.ndarray,  # Fortran-contiguous (T x K_req)
    decay_factor: float,  # lambda (e.g., 0.94)
    max_vol_ratio: float,  # Maximum allowed volatility ratio (e.g., 5.0)
    out_ratios: np.ndarray,  # Fortran-contiguous (T x K_req)
) -> None:
    """
    Computes the FHS Z-score matrix: Z_t = (Sigma_today / Sigma_t)
    Assumes `residuals` (or returns) are already computed and aligned.
    """
    T = residuals.shape[0]
    K = residuals.shape[1]

    for k in nb.prange(K):
        col_res = residuals[:, k]
        col_out_ratios = out_ratios[:, k]

        variances = np.empty(T, dtype=np.float64)

        # 1. Initialize first return
        # t=0 has no return, so we seed the variance with t=1
        r1 = col_res[1]
        var_t = r1 * r1
        variances[0] = var_t
        variances[1] = var_t

        # 2. EWMA loop
        for t in range(2, T):
            rt = col_res[t]
            var_t = (1.0 - decay_factor) * (rt * rt) + decay_factor * var_t
            variances[t] = var_t

        # 3. Today's variance is the very last element
        var_today = variances[T - 1]

        # 4. Calculate Z-ratios (Sigma_today / Sigma_t)
        for t in range(T):
            if variances[t] < MIN_EWMA_VARIANCE:
                col_out_ratios[t] = 1.0
            else:
                # Calculate raw ratio and clamp it to the max_vol_ratio
                raw_ratio = np.sqrt(var_today / variances[t])
                col_out_ratios[t] = min(raw_ratio, max_vol_ratio)
