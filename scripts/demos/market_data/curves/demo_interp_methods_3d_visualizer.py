# scripts/demos/market_data/curves/demo_interp_methods_3d_visualizer.py

import matplotlib.pyplot as plt
import numpy as np

# Import the Enums and Dispatcher (adjust paths based on your project structure)
from riskquant.market_data.curves.extrap_methods import (
    EXTRAP_FLAT,
    EXTRAP_LINEAR,
)
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.market_data.curves.interp_methods import (
    INTERP_LINEAR,
    INTERP_LOCF,
    INTERP_LOG_LINEAR,
    INTERP_NOCB,
)


def run_full_visualization_demo():
    # 1. Define a highly erratic, "kinky" input curve
    # We've added more pillars to really show off the zigzag.
    raw_tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

    # Notice how the discount factors go down, UP, down, UP.
    # A Df going up over time means negative forward rates!
    raw_dfs = np.array([0.96, 0.70, 0.90, 0.50, 0.75, 0.40, 0.60, 0.20, 0.35])

    # 2. Reshape to match Numba signatures: (C, M) and (C, M, S)
    tenors = raw_tenors.reshape(1, -1)
    values = raw_dfs.reshape(1, -1, 1)

    # Warning: If you use negative values in raw_dfs, np.log() will throw a NaN error.
    # Log-linear interpolators expect strictly positive discount factors.
    log_values = np.log(values)

    # 3. Create targets extending beyond boundaries
    targets = np.linspace(0.0, 35.0, 1000)

    # Helper function to invoke the dispatcher and handle allocation
    def get_curve(interp_id, extrap_id, use_log=False):
        # Fetch the correct Numba kernel from the registry
        kernel = CURVE_MATH_REGISTRY[(int(interp_id), int(extrap_id))]

        # Pre-allocate output array
        out = np.empty((tenors.shape[0], targets.shape[0], values.shape[2]))

        # Inject log_values for the Log-Linear methods, else raw values
        input_vals = log_values if use_log else values

        # Execute the kernel
        kernel(tenors, input_vals, targets, out)

        # Return the 1D array (Curve 0, Scenario 0) for plotting
        return out[0, :, 0]

    # 4. Generate all 8 combinations via the dispatcher
    lin_flat = get_curve(INTERP_LINEAR, EXTRAP_FLAT)
    lin_lin = get_curve(INTERP_LINEAR, EXTRAP_LINEAR)

    log_flat = get_curve(INTERP_LOG_LINEAR, EXTRAP_FLAT, use_log=True)
    log_lin = get_curve(INTERP_LOG_LINEAR, EXTRAP_LINEAR, use_log=True)

    step_l_flat = get_curve(INTERP_LOCF, EXTRAP_FLAT)
    step_l_lin = get_curve(INTERP_LOCF, EXTRAP_LINEAR)

    step_r_flat = get_curve(INTERP_NOCB, EXTRAP_FLAT)
    step_r_lin = get_curve(INTERP_NOCB, EXTRAP_LINEAR)

    # 5. Plotting in a 2x2 Grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "RiskQuant Interpolation & Extrapolation Behaviors via Dispatcher", fontsize=16
    )

    # Top-Left: Linear
    axs[0, 0].plot(targets, lin_flat, label="Flat Extrap", linewidth=2)
    axs[0, 0].plot(targets, lin_lin, label="Linear Extrap", linestyle="--")
    axs[0, 0].set_title("Linear Interpolation")

    # Top-Right: Log-Linear
    axs[0, 1].plot(targets, log_flat, label="Flat Extrap", linewidth=2)
    axs[0, 1].plot(targets, log_lin, label="Linear Extrap", linestyle="--")
    axs[0, 1].set_title("Log-Linear Interpolation (Discount Factors)")

    # Bottom-Left: Step Left
    # (Note: Flat and Linear extrapolate identically on step functions per your code)
    axs[1, 0].plot(targets, step_l_flat, label="Flat Extrap", linewidth=4, alpha=0.5)
    axs[1, 0].plot(targets, step_l_lin, label="Linear Extrap", linestyle=":")
    axs[1, 0].set_title("Step Left (LOCF)")

    # Bottom-Right: Step Right
    axs[1, 1].plot(targets, step_r_flat, label="Flat Extrap", linewidth=4, alpha=0.5)
    axs[1, 1].plot(targets, step_r_lin, label="Linear Extrap", linestyle=":")
    axs[1, 1].set_title("Step Right (NOCB)")

    # Apply common styling to all subplots
    for ax in axs.flat:
        ax.scatter(
            raw_tenors, raw_dfs, color="red", zorder=5, label="Input Pillars", s=50
        )
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=30.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Tenor (Years)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    plt.show()


if __name__ == "__main__":
    run_full_visualization_demo()
