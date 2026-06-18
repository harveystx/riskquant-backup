# scripts/demos/credit/jtd/evaluators/demo_sf_copula_gaussian.py
"""Demonstration of the S-Tier Gaussian Copula JTD Evaluator with synthetic data."""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# ============================================================
# IMPORT PRODUCTION MODULES
# ============================================================
from riskquant.constants import RNG_SEED
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.evaluators.sf_copula_gaussian import (
    evaluate_sf_gaussian_copula_jtd,
)

# --- 1. Scale Parameters ---
M_OBLIGORS = 100  # 100 Counterparties
S_SCENARIOS = 5_000  # 50,000 Monte Carlo Paths

# 1 Systemic Factor + M Idiosyncratic Factors
F_FACTORS = 1 + M_OBLIGORS


# Mocking the state container since it sits outside the provided context
@dataclass
class MockCreditFactorMatrix:
    latent_factors: npt.NDArray[np.float64]


def run_demo() -> None:
    print("--- S-TIER GAUSSIAN COPULA EVALUATOR DEMO ---\n")
    rng = np.random.default_rng(seed=RNG_SEED)

    # --- 2. Allocate & Generate Data (The Input) ---
    print(f"Allocating Memory for {M_OBLIGORS} Obligors x {S_SCENARIOS} Scenarios...")

    # EAD Matrix: (M, S). Simulating constant $1M to $5M exposures for simplicity
    ead_base = rng.uniform(1e6, 5e6, (M_OBLIGORS, 1))
    ead = (
        np.broadcast_to(ead_base, (M_OBLIGORS, S_SCENARIOS))
        .astype(np.float64)
        .copy(order="C")
    )

    # Latent Factors: (F, S). Standard Normal Draws.
    latent_factors = rng.standard_normal((F_FACTORS, S_SCENARIOS), dtype=np.float64)

    # Instantiate the real production container
    factor_matrix = CreditFactorMatrix(latent_factors=latent_factors)
    # ...

    # Obligor Parameters
    obligor_uids = np.arange(1000, 1000 + M_OBLIGORS, dtype=np.int64)
    lgd = rng.uniform(0.4, 0.6, M_OBLIGORS).astype(np.float64)  # 40% to 60% LGD
    beta = rng.uniform(0.3, 0.7, M_OBLIGORS).astype(
        np.float64
    )  # Correlation to systemic factor

    # Thresholds: ~1% to ~5% Probability of Default mapped to inverse normal CDF
    # norm.ppf(0.01) = -2.33, norm.ppf(0.05) = -1.64
    thresholds = rng.uniform(-2.5, -1.6, M_OBLIGORS).astype(np.float64)

    # Uix (Pointers mapping obligors to factor matrix rows)
    sys_uix = np.empty(
        M_OBLIGORS, dtype=np.int32
    )  # All point to row 0 (The Market Factor)
    idio_uix = np.arange(1, 1 + M_OBLIGORS, dtype=np.int32)  # Point to rows 1 to M

    # Assemble the JTD Block
    block = SfGaussianCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd,
        thresholds=thresholds,
        factor_loading=beta,
        systemic_uix=sys_uix,
        idiosyncratic_uix=idio_uix,
    )

    print("\n--- Block Structure ---")
    block.print_block(max_items=5)

    # Pre-allocate contiguous output memory
    out_loss = np.empty((M_OBLIGORS, S_SCENARIOS), dtype=np.float64)

    # --- 3. The Evaluator Call (JIT Boundary) ---
    print("Evaluating Jump-To-Default Losses...")
    start_time = time.perf_counter()

    evaluate_sf_gaussian_copula_jtd(
        block=block,
        ead=ead,
        credit_factors=factor_matrix,
        out=out_loss,
    )

    print(f"Evaluator completed in {time.perf_counter() - start_time:.4f}s\n")

    # --- 4. Reduction & Metrics (The Output) ---
    # Collapse the (M, S) obligor losses into an (S,) array of Portfolio Losses
    portfolio_losses = np.sum(out_loss, axis=0)

    expected_loss = np.mean(portfolio_losses)
    cvar_99 = np.quantile(portfolio_losses, 0.99)
    cvar_999 = np.quantile(portfolio_losses, 0.999)

    print("--- Portfolio Credit Risk Metrics ---")
    print(f"Expected Loss (EL):  ${expected_loss:,.2f}")
    print(f"Credit VaR (99.0%):  ${cvar_99:,.2f}")
    print(f"Credit VaR (99.9%):  ${cvar_999:,.2f}")
    print(f"Worst Scenario Loss: ${np.max(portfolio_losses):,.2f}")

    # --- 5. Visualization ---
    plot_copula_results(portfolio_losses, expected_loss, cvar_99)


def plot_copula_results(
    portfolio_losses: np.ndarray, el: float, cvar_99: float
) -> None:
    """Plots the right-tailed loss distribution characteristic of copula models."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of losses (ignoring zero-loss scenarios for clarity if desired,
    # but for portfolio level, we plot all)
    counts, bins, patches = ax.hist(
        portfolio_losses / 1e6,  # Convert to Millions for readable axis
        bins=100,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.8,
    )

    # Add vertical lines for Risk Metrics
    ax.axvline(
        el / 1e6,
        color="#2ca02c",
        linestyle="dashed",
        linewidth=2,
        label=f"Expected Loss (${el / 1e6:.1f}M)",
    )
    ax.axvline(
        cvar_99 / 1e6,
        color="#d62728",
        linestyle="solid",
        linewidth=2,
        label=f"99% Credit VaR (${cvar_99 / 1e6:.1f}M)",
    )

    ax.set_title(
        "Portfolio Loss Distribution (Single-Factor Gaussian Copula)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Simulated Portfolio Loss ($ Millions)", fontsize=12)
    ax.set_ylabel("Frequency (Scenarios)", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
