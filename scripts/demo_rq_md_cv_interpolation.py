# scripts/demo_rq_md_cv_interpolation.py

import numpy as np
import matplotlib.pyplot as plt

# Import the kernels directly from your definitions file
from riskquant.market_data.curves.interp_defs import (
    interp_linear_flat_extrap_3d,
    interp_linear_linear_extrap_3d,
    interp_log_linear_flat_extrap_3d,
    interp_log_linear_linear_extrap_3d,
    interp_step_left_flat_extrap_3d,
    interp_step_left_linear_extrap_3d,
    interp_step_right_flat_extrap_3d,
    interp_step_right_linear_extrap_3d,
)
from riskquant.utils.printer import print_calling_file, print_success_message


def run_demo():
    print_calling_file()
    print("Setting up dummy 3D curve data...")

    # ==========================================
    # 1. SETUP DUMMY DATA
    # ==========================================
    C_curves = 1
    S_scenarios = 1

    # Tenors: 5 pillars (e.g., Year 1 to Year 5)
    tenors = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # Shape: (C, M)
    t_flat = tenors[0]

    # Standard Values: (e.g., Forward Prices)
    # Shape: (C, M, S) -> (1, 5, 1)
    values = np.array([[[2.0], [4.5], [3.0], [5.0], [4.0]]])
    v_flat = values[0, :, 0]

    # Log Values: (e.g., Discount Factors)
    # Shape: (C, M, S) -> (1, 5, 1)
    # DFs naturally decay towards 0
    dfs = np.array([[[0.96], [0.90], [0.83], [0.75], [0.66]]])
    ln_values = np.log(dfs)
    df_flat = dfs[0, :, 0]

    # Targets: High-resolution X-axis from Year 0.0 to 7.0
    # This will heavily test the extrapolation on both ends!
    targets = np.linspace(-1.0, 7.0, 500)

    print(f"Executing vectorized math for {len(targets)} targets...")

    # ==========================================
    # 2. EXECUTE KERNELS
    # ==========================================

    # --- Linear ---
    y_lin_flat = interp_linear_flat_extrap_3d(tenors, values, targets)[0, :, 0]
    y_lin_lin = interp_linear_linear_extrap_3d(tenors, values, targets)[0, :, 0]

    # --- Log-Linear (Discount Factors) ---
    # Remember: the kernel exponentiates the result automatically
    y_ll_flat = interp_log_linear_flat_extrap_3d(tenors, ln_values, targets)[
        0, :, 0
    ]
    y_ll_lin = interp_log_linear_linear_extrap_3d(tenors, ln_values, targets)[
        0, :, 0
    ]

    # --- Step Left ---
    y_sl_flat = interp_step_left_flat_extrap_3d(tenors, values, targets)[
        0, :, 0
    ]
    y_sl_lin = interp_step_left_linear_extrap_3d(tenors, values, targets)[
        0, :, 0
    ]

    # --- Step Right ---
    y_sr_flat = interp_step_right_flat_extrap_3d(tenors, values, targets)[
        0, :, 0
    ]
    y_sr_lin = interp_step_right_linear_extrap_3d(tenors, values, targets)[
        0, :, 0
    ]

    # ==========================================
    # 3. PLOT RESULTS
    # ==========================================
    print("Rendering plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "RiskQuant Vectorized Curve Interpolation & Extrapolation", fontsize=16
    )

    # Helper to plot pillars
    def plot_pillars(ax, y_data, title):
        ax.plot(
            t_flat,
            y_data,
            "ko",
            markersize=8,
            label="Known Pillars",
            zorder=5,
        )
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(5.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 1. Linear
    axs[0, 0].plot(targets, y_lin_lin, "r-", linewidth=2, label="Linear Extrap")
    axs[0, 0].plot(targets, y_lin_flat, "b--", linewidth=2, label="Flat Extrap")
    plot_pillars(axs[0, 0], v_flat, "Linear Interpolation (Prices/Rates)")

    # 2. Log-Linear
    axs[0, 1].plot(
        targets,
        y_ll_lin,
        "r-",
        linewidth=2,
        label="Linear Extrap (Constant Fwd)",
    )
    axs[0, 1].plot(
        targets,
        y_ll_flat,
        "b--",
        linewidth=2,
        label="Flat Extrap (Constant Fwd)",
    )
    plot_pillars(
        axs[0, 1], df_flat, "Log-Linear Interpolation (Discount Factors)"
    )
    axs[0, 1].set_ylabel("Discount Factor")

    # 3. Step Left
    # Note: Flat vs Linear extrap on a step function yield the same result since slope = 0
    axs[1, 0].plot(
        targets, y_sl_flat, "b-", linewidth=2, label="Step Left (LOCF)"
    )
    plot_pillars(axs[1, 0], v_flat, "Step Left Interpolation")

    # 4. Step Right
    axs[1, 1].plot(
        targets, y_sr_flat, "g-", linewidth=2, label="Step Right (NOCB)"
    )
    plot_pillars(axs[1, 1], v_flat, "Step Right Interpolation")

    plt.tight_layout()
    plt.show()

    print_success_message()


if __name__ == "__main__":
    run_demo()
