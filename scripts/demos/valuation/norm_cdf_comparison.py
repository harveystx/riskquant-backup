# scripts/demos/valuation/norm_cdf_comparison.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from riskquant.valuation.pricer_kernels._deprecated._math_v1 import (
    numba_norm_cdf,
    numba_norm_cdf_abrom_steg,
)


@nb.njit(cache=True, parallel=True)
def eval_exact(xs: np.ndarray) -> np.ndarray:
    n = xs.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in nb.prange(n):
        out[i] = numba_norm_cdf(xs[i])

    return out


@nb.njit(cache=True, parallel=True)
def eval_abrom_steg(xs: np.ndarray) -> np.ndarray:
    n = xs.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in nb.prange(n):
        out[i] = numba_norm_cdf_abrom_steg(xs[i])

    return out


def main() -> None:

    xs = np.linspace(
        -8.0,
        8.0,
        1_000_000,
        dtype=np.float64,
    )

    print("Compiling Numba kernels...")
    exact = eval_exact(xs)
    approx = eval_abrom_steg(xs)

    abs_error = np.abs(exact - approx)

    rel_error = abs_error / np.maximum(
        np.abs(exact),
        1e-16,
    )

    signed_error = exact - approx

    idx_max = np.argmax(abs_error)

    print()
    print("Accuracy Statistics")
    print("-" * 60)
    print(f"Max Abs Error : {abs_error.max():.12e}")
    print(f"Mean Abs Error: {abs_error.mean():.12e}")
    print(f"RMSE          : {np.sqrt(np.mean(abs_error**2)):.12e}")
    print(f"Max Rel Error : {rel_error.max():.12e}")
    print()
    print(f"x @ max error : {xs[idx_max]:.8f}")
    print(f"Exact         : {exact[idx_max]:.12e}")
    print(f"Approx        : {approx[idx_max]:.12e}")

    # ==========================================================
    # Combined Figure
    # ==========================================================

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 12),
    )

    # ----------------------------------------------------------
    # Panel 1: CDF Comparison
    # ----------------------------------------------------------

    axes[0, 0].plot(
        xs,
        exact,
        label="Exact (erf)",
    )

    axes[0, 0].plot(
        xs,
        approx,
        "--",
        label="Abramowitz-Stegun",
    )

    axes[0, 0].set_title("Normal CDF")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("CDF")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # ----------------------------------------------------------
    # Panel 2: Absolute Error (Log Scale)
    # ----------------------------------------------------------

    axes[0, 1].semilogy(
        xs,
        abs_error,
    )

    axes[0, 1].set_title("Absolute Error")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("|Exact - Approx|")
    axes[0, 1].grid(True)

    # ----------------------------------------------------------
    # Panel 3: Signed Error
    # ----------------------------------------------------------

    axes[1, 0].plot(
        xs,
        signed_error,
    )

    axes[1, 0].set_title("Signed Error")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Exact - Approx")
    axes[1, 0].grid(True)

    # ----------------------------------------------------------
    # Panel 4: Relative Error (Log Scale)
    # ----------------------------------------------------------

    axes[1, 1].semilogy(
        xs,
        rel_error,
    )

    axes[1, 1].set_title("Relative Error")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Relative Error")
    axes[1, 1].grid(True)

    plt.suptitle(
        "Normal CDF Approximation Validation",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
