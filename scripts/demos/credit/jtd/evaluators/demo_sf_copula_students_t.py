# scripts/demos/credit/jtd/evaluators/demo_sf_copula_students_t.py
"""Demonstration of the S-Tier Student-t Copula JTD Evaluator with synthetic data."""

import time

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# IMPORT PRODUCTION MODULES
# ============================================================
from riskquant.constants import RNG_SEED
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.sf_copula_students_t import SfStudentsTCopulaJtdBlock
from riskquant.credit.jtd.evaluators.sf_copula_students_t import (
    evaluate_sf_students_t_copula_jtd,
)

# --- 1. Scale Parameters ---
M_OBLIGORS = 100  # 100 Counterparties
S_SCENARIOS = 50_000  # 50,000 Monte Carlo Paths

# 1 Systemic Factor + M Idiosyncratic Factors + M Chi-Squared Factors
F_FACTORS = 1 + (2 * M_OBLIGORS)


def run_demo() -> None:
    print("--- S-TIER STUDENT-T COPULA EVALUATOR DEMO ---\n")
    rng = np.random.default_rng(seed=RNG_SEED)

    # --- 2. Allocate & Generate Data (The Input) ---
    print(f"Allocating Memory for {M_OBLIGORS} Obligors x {S_SCENARIOS} Scenarios...")

    # EAD Matrix: (M, S). Simulating constant $1M to $5M exposures
    ead_base = rng.uniform(1e6, 5e6, (M_OBLIGORS, 1))
    ead = (
        np.broadcast_to(ead_base, (M_OBLIGORS, S_SCENARIOS))
        .astype(np.float64)
        .copy(order="C")
    )

    # Obligor Parameters
    obligor_uids = np.arange(1000, 1000 + M_OBLIGORS, dtype=np.int64)
    lgd = rng.uniform(0.4, 0.6, M_OBLIGORS).astype(np.float64)
    beta = rng.uniform(0.3, 0.7, M_OBLIGORS).astype(np.float64)

    # Degrees of Freedom (nu): Lower values (e.g., 4 to 6) create extremely fat tails
    nu = rng.uniform(4.0, 8.0, M_OBLIGORS).astype(np.float64)

    # Thresholds: Mapped to inverse CDF (approximate -2.5 to -1.6)
    thresholds = rng.uniform(-2.5, -1.6, M_OBLIGORS).astype(np.float64)

    # --- Latent Factor Generation (The Mathematical Setup) ---
    # 1. Systemic Factor (Z ~ N(0,1))
    Z = rng.standard_normal((1, S_SCENARIOS), dtype=np.float64)

    # 2. Idiosyncratic Factors (Eps ~ N(0,1))
    eps = rng.standard_normal((M_OBLIGORS, S_SCENARIOS), dtype=np.float64)

    # 3. Chi-Squared Factors (W ~ Chi^2(nu))
    # Note: We pass the obligor-specific `nu` array to generate df-matched Chi-sq draws
    W = rng.chisquare(df=nu[:, None], size=(M_OBLIGORS, S_SCENARIOS))

    # CRITICAL: The kernel expects 1 / sqrt(W) precomputed to avoid inner-loop division
    inv_sqrt_w = (1.0 / np.sqrt(W)).astype(np.float64)

    # Stack into a single C-contiguous factor matrix
    latent_factors = np.vstack([Z, eps, inv_sqrt_w]).astype(np.float64).copy(order="C")
    factor_matrix = CreditFactorMatrix(latent_factors=latent_factors)

    # --- Index Pointers (UIX) ---
    sys_uix = np.empty(M_OBLIGORS, dtype=np.int32)
    idio_uix = np.arange(1, 1 + M_OBLIGORS, dtype=np.int32)
    chi_uix = np.arange(1 + M_OBLIGORS, 1 + (2 * M_OBLIGORS), dtype=np.int32)

    # Assemble the JTD Block
    block = SfStudentsTCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd,
        thresholds=thresholds,
        factor_loading=beta,
        sqrt_nu=np.sqrt(nu),
        systemic_uix=sys_uix,
        idiosyncratic_uix=idio_uix,
        chi_squared_uix=chi_uix,
    )

    print("\n--- Block Structure ---")
    block.print_block(max_items=5)

    # Pre-allocate contiguous output memory
    out_loss = np.empty((M_OBLIGORS, S_SCENARIOS), dtype=np.float64)

    # --- 3. The Evaluator Call (JIT Boundary) ---
    print("Evaluating Jump-To-Default Losses (Student-t Copula)...")
    start_time = time.perf_counter()

    evaluate_sf_students_t_copula_jtd(
        block=block,
        ead=ead,
        credit_factors=factor_matrix,
        out=out_loss,
    )

    print(f"Evaluator completed in {time.perf_counter() - start_time:.4f}s\n")

    # --- 4. Reduction & Metrics (The Output) ---
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

    counts, bins, patches = ax.hist(
        portfolio_losses / 1e6,  # Convert to Millions for readable axis
        bins=100,
        color="#8c564b",  # Distinct brown/red color for Student-t
        edgecolor="black",
        alpha=0.8,
    )

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
        "Portfolio Loss Distribution (Single-Factor Student-t Copula)",
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
