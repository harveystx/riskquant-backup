# scripts/demo_rq_simlab_scenarios_core_math_ar1_filter.py
import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals

#####################
# AR(1) FILTER DEMO #
#####################


def simulate_ar1_returns(T=1000, phi=0.7, sigma=1.0, seed=42):
    np.random.seed(seed)
    r = np.zeros(T)
    noise = sigma * np.random.randn(T)

    for t in range(1, T):
        r[t] = phi * r[t - 1] + noise[t]

    return r


def returns_to_prices(r):
    return np.exp(np.cumsum(r))


def run_kernel(values):
    T = values.shape[0]
    out = np.zeros((T, 1), dtype=np.float64, order="F")

    kernel_ar1_residuals(
        values,
        cols=np.array([0]),
        shift_types=np.array([SHIFT_LOG_RETURN_ID]),
        out_residuals=out,
    )
    return out[:, 0]


def compute_log_returns(values):
    r = np.zeros_like(values)
    for t in range(1, len(values)):
        r[t] = np.log(values[t] / values[t - 1])
    return r


def autocorr(x, lag=1):
    return np.corrcoef(x[lag:], x[:-lag])[0, 1]


# ---------------------------------------------------------
# Main demo
# ---------------------------------------------------------


def run_demo_ar1_filter_autocorrelated_series():
    T = 1500
    phi_true = 0.8

    # -----------------------------------------------------
    # 1. Simulate AR(1) returns
    # -----------------------------------------------------
    raw_returns = simulate_ar1_returns(T=T, phi=phi_true)
    prices = returns_to_prices(raw_returns)

    values = np.asfortranarray(prices.reshape(-1, 1))

    # -----------------------------------------------------
    # 2. Run kernel
    # -----------------------------------------------------
    residuals = run_kernel(values)

    # recompute returns (as kernel sees them)
    computed_returns = compute_log_returns(prices)

    # -----------------------------------------------------
    # 3. Diagnostics
    # -----------------------------------------------------
    raw_acf = autocorr(computed_returns[2:])
    res_acf = autocorr(residuals[2:])

    print("\n=== AR(1) FILTER DEMO ===")
    print(f"True phi:                  {phi_true:.3f}")
    print(f"Raw returns lag-1 ACF:     {raw_acf:.3f}")
    print(f"Residuals lag-1 ACF:       {res_acf:.3f}")
    print(f"Raw mean:                  {np.mean(computed_returns):.6f}")
    print(f"Residual mean:             {np.mean(residuals):.6f}")
    print(f"Raw std:                   {np.std(computed_returns):.6f}")
    print(f"Residual std:              {np.std(residuals):.6f}")

    # -----------------------------------------------------
    # 4. Plots
    # -----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # --- Time series ---
    axes[0, 0].plot(computed_returns, label="Raw Returns")
    axes[0, 0].set_title("Raw Returns (AR(1) structure visible)")
    axes[0, 0].legend()

    axes[0, 1].plot(residuals, label="Residuals", color="orange")
    axes[0, 1].set_title("Residuals (whitened)")
    axes[0, 1].legend()

    # --- Histograms ---
    axes[1, 0].hist(computed_returns[2:], bins=50, alpha=0.7)
    axes[1, 0].set_title("Raw Returns Distribution")

    axes[1, 1].hist(residuals[2:], bins=50, alpha=0.7, color="orange")
    axes[1, 1].set_title("Residuals Distribution")

    # --- Scatter (lag plot) ---
    axes[2, 0].scatter(
        computed_returns[:-1],
        computed_returns[1:],
        alpha=0.3,
        s=5,
    )
    axes[2, 0].set_title(f"Raw Returns Lag Plot (ACF ≈ {raw_acf:.2f})")
    axes[2, 0].set_xlabel("r(t-1)")
    axes[2, 0].set_ylabel("r(t)")

    axes[2, 1].scatter(
        residuals[:-1],
        residuals[1:],
        alpha=0.3,
        s=5,
        color="orange",
    )
    axes[2, 1].set_title(f"Residuals Lag Plot (ACF ≈ {res_acf:.2f})")
    axes[2, 1].set_xlabel("e(t-1)")
    axes[2, 1].set_ylabel("e(t)")

    plt.tight_layout()
    plt.show()


#######################################
# ILLIQUIDITY + REGIME-SWITCHING DEMO #
#######################################


# ---------------------------------------------------------
# Illiquid + Regime-Switching Time Series
# ---------------------------------------------------------


def simulate_illiquid_regime_series(
    T=2000,
    seed=123,
):
    np.random.seed(seed)

    prices = np.zeros(T)
    prices[0] = 100.0

    regimes = [
        {
            "phi": 0.98,  # Near unit root
            "sigma": 0.003,  # Lower volatility
            "p_stale": 0.8,  # Very high stale probability
            "stale_duration": 5,  # Multiple periods of staleness
        },
        {"phi": 0.90, "sigma": 0.008, "p_stale": 0.15, "stale_duration": 2},
        {"phi": 0.75, "sigma": 0.015, "p_stale": 0.05, "stale_duration": 1},
    ]

    regime = 0
    r_prev = 0.0
    stale_counter = 0

    for t in range(1, T):
        if np.random.rand() < 0.01:
            regime = np.random.randint(len(regimes))

        phi = regimes[regime]["phi"]
        sigma = regimes[regime]["sigma"]
        p_stale = regimes[regime]["p_stale"]
        stale_duration = regimes[regime]["stale_duration"]

        # Extended stale periods
        if stale_counter > 0:
            prices[t] = prices[t - 1]
            stale_counter -= 1
            continue
        elif np.random.rand() < p_stale:
            prices[t] = prices[t - 1]
            stale_counter = np.random.poisson(stale_duration - 1)
            continue

        noise = sigma * np.random.randn()
        r = phi * r_prev + noise

        if np.random.rand() < 0.02:
            r += np.random.randn() * 0.05

        prices[t] = prices[t - 1] * np.exp(r)
        r_prev = r

    return prices


# ---------------------------------------------------------
# Diagnostics for illiquid behaviour
# ---------------------------------------------------------


def compute_zero_return_fraction(returns):
    return np.mean(np.isclose(returns, 0.0))


def rolling_autocorr(x, window=100):
    out = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        out[i] = np.corrcoef(x[i - window : i - 1], x[i - window + 1 : i])[0, 1]
    return out


# ---------------------------------------------------------
# Plug into your main()
# ---------------------------------------------------------


def run_demo_ar1_filter_illiquid_series():
    T = 2000

    # 1. simulate
    prices = simulate_illiquid_regime_series(T=T)
    values = np.asfortranarray(prices.reshape(-1, 1))

    # 2. run kernel
    residuals = run_kernel(values)

    # 3. compute returns
    raw_returns = compute_log_returns(prices)

    # 4. diagnostics
    raw_acf = autocorr(raw_returns[2:])
    res_acf = autocorr(residuals[2:])

    zero_frac_raw = compute_zero_return_fraction(raw_returns)
    zero_frac_res = compute_zero_return_fraction(residuals)

    print("\n=== ILLIQUID REGIME DEMO ===")
    print(f"Raw ACF:                 {raw_acf:.3f}")
    print(f"Residual ACF:            {res_acf:.3f}")
    print(f"Zero-return fraction raw: {zero_frac_raw:.3f}")
    print(f"Zero-return fraction res: {zero_frac_res:.3f}")

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # --- Row 0: High-Level Series & Rolling ACF ---
    axes[0, 0].plot(prices)
    axes[0, 0].set_title("Illiquid Price Series (stale + jumps)")

    axes[0, 1].plot(rolling_autocorr(raw_returns), label="Raw")
    axes[0, 1].plot(rolling_autocorr(residuals), label="Residuals", color="orange")
    axes[0, 1].legend()
    axes[0, 1].set_title("Rolling Lag-1 Autocorrelation")

    # --- Row 1: Time Series ---
    axes[1, 0].plot(raw_returns)
    axes[1, 0].set_title("Raw Returns (heteroskedastic + regimes)")

    axes[1, 1].plot(residuals, color="orange")
    axes[1, 1].set_title("Residuals (after AR(1) filter)")

    # --- Row 2: Histograms ---
    axes[2, 0].hist(raw_returns[2:], bins=50, alpha=0.7)
    axes[2, 0].set_title("Raw Returns Distribution")

    axes[2, 1].hist(residuals[2:], bins=50, alpha=0.7, color="orange")
    axes[2, 1].set_title("Residuals Distribution")

    # --- Row 3: Lag plots ---
    axes[3, 0].scatter(raw_returns[:-1], raw_returns[1:], s=5, alpha=0.3)
    axes[3, 0].set_title("Raw Lag Plot")
    axes[3, 0].set_xlabel("r(t-1)")
    axes[3, 0].set_ylabel("r(t)")

    axes[3, 1].scatter(residuals[:-1], residuals[1:], s=5, alpha=0.3, color="orange")
    axes[3, 1].set_title("Residual Lag Plot")
    axes[3, 1].set_xlabel("e(t-1)")
    axes[3, 1].set_ylabel("e(t)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# White Noise (Liquid / Efficient Market) Demo
# ---------------------------------------------------------


def simulate_white_noise_series(T=1000, mu=0.005, sigma=0.015, seed=42):
    """
    Simulates a purely random walk with drift.
    There is zero structural autocorrelation (beta = 0).
    """
    np.random.seed(seed)
    r = np.random.normal(mu, sigma, T)
    return r


def run_demo_ar1_filter_white_noise_series():
    T = 1500
    mu_true = 0.005  # Noticeable drift so we can see the de-meaning
    sigma_true = 0.015

    # 1. Simulate
    raw_returns = simulate_white_noise_series(T=T, mu=mu_true, sigma=sigma_true)
    prices = returns_to_prices(raw_returns)
    values = np.asfortranarray(prices.reshape(-1, 1))

    # 2. Run kernel
    residuals = run_kernel(values)
    computed_returns = compute_log_returns(prices)

    # 3. Diagnostics
    raw_acf = autocorr(computed_returns[2:])
    res_acf = autocorr(residuals[2:])

    raw_mean = np.mean(computed_returns[2:])
    res_mean = np.mean(residuals[2:])

    print("\n=== EFFICIENT MARKET (DE-MEAN ONLY) DEMO ===")
    print("Behavior: Kernel detects t-stat < 1.96, sets beta=0, and only de-means.")
    print(f"True drift (mu):           {mu_true:.6f}")
    print(f"Raw returns lag-1 ACF:     {raw_acf:.3f} (Should be ~0)")
    print(f"Residuals lag-1 ACF:       {res_acf:.3f} (Should be ~0)")
    print(f"Raw mean:                  {raw_mean:.6f} (Should be ~mu)")
    print(f"Residual mean:             {res_mean:.6f} (Should be exactly 0)")

    # 4. Plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # --- Time series ---
    axes[0, 0].plot(
        computed_returns, label="Raw Returns (with drift)", color="steelblue"
    )
    axes[0, 0].set_title("Raw Returns (White Noise)")
    axes[0, 0].legend()

    axes[0, 1].plot(residuals, label="Residuals", color="green")
    axes[0, 1].set_title("Residuals (Strictly De-meaned)")
    axes[0, 1].legend()

    # --- Histograms (Showing the shift) ---
    axes[1, 0].hist(computed_returns[2:], bins=50, alpha=0.7, color="steelblue")
    axes[1, 0].axvline(
        raw_mean,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {raw_mean:.4f}",
    )
    axes[1, 0].set_title("Raw Returns Distribution")
    axes[1, 0].legend()

    axes[1, 1].hist(residuals[2:], bins=50, alpha=0.7, color="green")
    axes[1, 1].axvline(
        res_mean,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {res_mean:.4f}",
    )
    axes[1, 1].set_title("Residuals Distribution")
    axes[1, 1].legend()

    # --- Scatter (lag plot) ---
    axes[2, 0].scatter(
        computed_returns[:-1],
        computed_returns[1:],
        alpha=0.3,
        s=5,
        color="steelblue",
    )
    axes[2, 0].set_title(f"Raw Returns Lag Plot (ACF ≈ {raw_acf:.2f})")
    axes[2, 0].set_xlabel("r(t-1)")
    axes[2, 0].set_ylabel("r(t)")

    axes[2, 1].scatter(residuals[:-1], residuals[1:], alpha=0.3, s=5, color="green")
    axes[2, 1].set_title(f"Residuals Lag Plot (ACF ≈ {res_acf:.2f})")
    axes[2, 1].set_xlabel("e(t-1)")
    axes[2, 1].set_ylabel("e(t)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo_ar1_filter_autocorrelated_series()
    run_demo_ar1_filter_illiquid_series()
    run_demo_ar1_filter_white_noise_series()
