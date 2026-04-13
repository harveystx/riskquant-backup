# src/riskquant/simlab/scenarios/core_math/ar1_filter.py
import numba as nb
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE,
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
)


@nb.njit(parallel=True, fastmath=True, cache=True)
def kernel_ar1_residuals(
    values: np.ndarray,  # Fortran-contiguous (T x K_total)
    cols: np.ndarray,  # (K_req,)
    shift_types: np.ndarray,  # (K_req,)
    out_residuals: np.ndarray,  # Fortran-contiguous (T x K_req)
) -> None:
    """
    Tier-1 optimized AR(1) residual kernel.

    Pipeline (per column):
    1. Compute returns in-place
    2. Accumulate OLS raw moments during return computation (single pass)
    3. Solve AR(1) coefficients (alpha, beta)
    4. Overwrite returns with residuals (second pass)

    Design goals:
    - Zero allocations
    - Minimal memory passes (2 total)
    - Cache-friendly streaming
    - Numerically stable OLS via raw moments
    """

    T = values.shape[0]
    K = cols.shape[0]

    for k in nb.prange(K):
        col = cols[k]
        st = shift_types[k]

        col_vals = values[:, col]
        col_res = out_residuals[:, k]

        # -------------------------------------------------------------
        # Edge cases
        # -------------------------------------------------------------
        if T < 2:
            if T == 1:
                col_res[0] = 0.0
            continue

        col_res[0] = 0.0

        # -------------------------------------------------------------
        # First pass:
        # - compute returns
        # - accumulate raw OLS moments
        # -------------------------------------------------------------
        prev_val = col_vals[0]

        # Initialize stats
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        sum_yy = 0.0

        # Compute r_1
        curr_val = col_vals[1]

        if st == SHIFT_LOG_RETURN:
            r = np.log(curr_val / prev_val)
        elif st == SHIFT_SIMPLE_RETURN:
            r = (curr_val / prev_val) - 1.0
        elif st == SHIFT_DIFFERENCE:
            r = curr_val - prev_val
        else:
            r = 0.0

        col_res[1] = r
        prev_r = r
        prev_val = curr_val

        # Main streaming loop
        for t in range(2, T):
            curr_val = col_vals[t]

            if st == SHIFT_LOG_RETURN:
                r = np.log(curr_val / prev_val)
            elif st == SHIFT_SIMPLE_RETURN:
                r = (curr_val / prev_val) - 1.0
            elif st == SHIFT_DIFFERENCE:
                r = curr_val - prev_val
            else:
                r = 0.0

            col_res[t] = r

            # Accumulate OLS stats
            sum_x += prev_r
            sum_y += r
            sum_xx += prev_r * prev_r
            sum_xy += prev_r * r
            sum_yy += r * r

            prev_r = r
            prev_val = curr_val

        # -------------------------------------------------------------
        # OLS solve (AR(1): r_t = alpha + beta * r_{t-1} + e_t)
        # -------------------------------------------------------------
        n_obs = T - 2

        if n_obs <= 0:
            continue

        mean_x = sum_x / n_obs
        mean_y = sum_y / n_obs

        var_x = sum_xx - n_obs * mean_x * mean_x
        cov_xy = sum_xy - n_obs * mean_x * mean_y

        if var_x < 1e-12:
            beta = 0.0
        else:
            beta = cov_xy / var_x

        alpha = mean_y - beta * mean_x

        # --------------------------------------------------------------
        # Statistical significance test for beta (proper t-test)
        # --------------------------------------------------------------

        # OLS standard property: RSS = SS_yy - beta * SS_xy
        # First, find the mean-centered sum of squares for Y
        var_y = sum_yy - n_obs * mean_y * mean_y

        # Compute Residual Sum of Squares directly (no loop required)
        rss = var_y - beta * cov_xy

        # Degrees of freedom for simple linear regression (1 predictor + intercept) is N - 2
        # (Your previous code used N - 1, which is for mean-only estimation)
        df = max(1, n_obs - 2)
        sigma2 = rss / df

        # Standard error of beta
        if var_x > 1e-12:
            se_beta = np.sqrt(sigma2 / var_x)
            t_stat = abs(beta / se_beta)
        else:
            t_stat = 0.0

        # Threshold (95% confidence)
        if t_stat < 1.96:
            beta = 0.0
            alpha = mean_y  # de-mean only

        # -------------------------------------------------------------
        # Second pass:
        # overwrite returns with residuals
        # -------------------------------------------------------------
        prev_r = col_res[1]
        col_res[1] = prev_r - alpha

        for t in range(2, T):
            curr_r = col_res[t]
            col_res[t] = curr_r - (alpha + beta * prev_r)
            prev_r = curr_r
