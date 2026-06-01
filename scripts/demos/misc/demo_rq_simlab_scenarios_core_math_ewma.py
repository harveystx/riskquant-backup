# scripts/demo_rq_simlab_scenarios_core_math_ewma.py
import numpy as np
import matplotlib.pyplot as plt

from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios_from_residuals,
)


# ---------------------------------------------------------
# Simulate Market Regimes
# ---------------------------------------------------------
def simulate_regime_shifting_returns(T=1000, seed=42):
    """
    Simulates a return series moving through three distinct regimes:
    1. Calm / Low Volatility
    2. Crisis / High Volatility
    3. Elevated / Current Volatility
    """
    np.random.seed(seed)

    # Regime 1: Calm (Days 0-500)
    r1 = np.random.normal(0, 0.005, 500)

    # Regime 2: Crisis (Days 500-800)
    r2 = np.random.normal(0, 0.040, 300)

    # Regime 3: Current Market (Laplace Distribution / Linear log-decay)
    # Target std dev = 0.020
    scale_param = 0.020 / np.sqrt(2)
    r3 = np.random.laplace(loc=0.0, scale=scale_param, size=200)
    r3 = r3 * np.linspace(
        0.20, 1.0, 200
    )  # Linear expansion of volatility in the current regime

    return np.concatenate([r1, r2, r3])


# ---------------------------------------------------------
# Helper: Reconstruct EWMA Volatility for plotting
# ---------------------------------------------------------
def compute_demo_ewma_volatility(
    returns: np.ndarray, decay_factor: float
) -> np.ndarray:
    """
    Mimics the exact internal variance loop of the Numba kernel to extract
    the absolute volatility (sigma) for visualization purposes.
    """
    T = len(returns)
    variances = np.zeros(T, dtype=np.float64)

    # Initialize exactly as the kernel does
    r1 = returns[1]
    var_t = r1 * r1
    variances[0] = var_t
    variances[1] = var_t

    for t in range(2, T):
        rt = returns[t]
        var_t = (1.0 - decay_factor) * (rt * rt) + decay_factor * var_t
        variances[t] = var_t

    return np.sqrt(variances)


# ---------------------------------------------------------
# Main Demo
# ---------------------------------------------------------
def run_demo_ewma_fhs_scaling():
    T = 1000
    decay_factor = 0.94
    max_vol_ratio = 5.0  # Safety cap

    # 1. Generate our synthetic returns (pure noise, no AR1 structure needed here)
    raw_returns = simulate_regime_shifting_returns(T=T)

    # Format for the Numba kernel: (T x K_req), Fortran contiguous
    residuals = np.asfortranarray(raw_returns.reshape(-1, 1))

    # 2. Run the EWMA Kernel
    out_ratios = np.zeros_like(residuals)

    kernel_ewma_vol_ratios_from_residuals(
        residuals=residuals,
        decay_factor=decay_factor,
        max_vol_ratio=max_vol_ratio,
        out_ratios=out_ratios,
    )

    # Extract 1D arrays for plotting
    ratios = out_ratios[:, 0]

    # Reconstruct the volatility series for visualization
    rolling_vol = compute_demo_ewma_volatility(raw_returns, decay_factor)

    # 3. Calculate the "Scaled Shocks" (This is what the FHS generator actually simulates)
    # Scaled Shock = Historical Return * (Sigma_today / Sigma_t)
    scaled_shocks = residuals[:, 0] * ratios

    # 4. Diagnostics & Sanity Checks
    print("\n=== EWMA VOLATILITY SCALING DEMO ===")
    print(f"Decay Factor (Lambda): {decay_factor}")
    print(f"Max Vol Ratio Cap:     {max_vol_ratio}")
    print(f"Current Volatility:    {rolling_vol[-1]*100:.2f}% (Daily)")
    print(
        f"Final Z-Ratio (t={T-1}): {ratios[-1]:.4f} (Should always be exactly 1.0)"
    )

    # Check what happened to the calm period (Regime 1)
    avg_ratio_calm = np.mean(ratios[100:400])
    print(f"Average scaling ratio during Calm Regime: {avg_ratio_calm:.2f}x")

    # Check what happened to the crisis period (Regime 2)
    avg_ratio_crisis = np.mean(ratios[600:750])
    print(
        f"Average scaling ratio during Crisis Regime: {avg_ratio_crisis:.2f}x"
    )

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Raw Historical Returns + EWMA Volatility Bands
    axes[0].plot(raw_returns, color="steelblue", alpha=0.6, label="Raw Returns")
    axes[0].plot(
        2 * rolling_vol,
        color="red",
        linestyle="--",
        alpha=0.8,
        label="+2σ EWMA Band",
    )
    axes[0].plot(
        -2 * rolling_vol,
        color="red",
        linestyle="--",
        alpha=0.8,
        label="-2σ EWMA Band",
    )

    axes[0].axvspan(0, 500, color="grey", alpha=0.1, label="Regime 1: Calm")
    axes[0].axvspan(500, 800, color="red", alpha=0.1, label="Regime 2: Crisis")
    axes[0].axvspan(
        800, 1000, color="orange", alpha=0.1, label="Regime 3: Current"
    )

    axes[0].set_title("1. Raw Historical Returns & Rolling Volatility")
    axes[0].set_ylabel("Return")
    axes[0].legend(loc="upper left")

    # Plot 2: The EWMA Z-Ratios (Sigma_today / Sigma_t)
    axes[1].plot(ratios, color="purple", linewidth=1.5)
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(
        max_vol_ratio, color="red", linestyle=":", label="Max Ratio Cap"
    )
    axes[1].set_title("2. FHS Scaling Ratios (Sigma_Today / Sigma_t)")
    axes[1].set_ylabel("Scaling Multiplier")
    axes[1].legend(loc="upper left")

    # Plot 3: The Filtered / Scaled Shocks
    axes[2].plot(scaled_shocks, color="green", alpha=0.8)
    axes[2].set_title(
        "3. Scaled Shocks Fed into Simulation (The Projected Future)"
    )
    axes[2].set_ylabel("Scaled Return")
    axes[2].set_xlabel("Time (t)")

    # Highlight how uniform the scaled shocks look compared to the raw returns
    axes[2].axvspan(0, 500, color="grey", alpha=0.1)
    axes[2].axvspan(500, 800, color="red", alpha=0.1)
    axes[2].axvspan(800, 1000, color="orange", alpha=0.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo_ewma_fhs_scaling()
