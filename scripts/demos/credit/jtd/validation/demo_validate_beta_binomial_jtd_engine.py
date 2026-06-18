# scripts/demos/credit/jtd/validation/demo_validate_beta_binomial_jtd_engine.py
"""
Beta-Binomial JTD Engine Validation Demo.

Validates the BETA_BINOMIAL model end to end: a single shared per-scenario
default probability P_s ~ Beta(alpha, beta) drives conditionally-independent
obligor defaults (u_i < P_s). The checks confirm (1) the empirical mean default
rate matches the theoretical E[P_s] = alpha / (alpha + beta), and (2) the
realized default rate is strongly overdispersed versus an independent-Bernoulli
book of the same size — the defining signature of the common-shock model.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from riskquant.constants import RNG_SEED
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.masters.cross_master.compiled import (
    CompiledCreditMasters,
)
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.factor_universe.ingest import (
    compile_credit_factor_universe,
)
from riskquant.credit.masters.obligors.ingest import compile_obligor_master
from riskquant.utils.printer import print_calling_file, print_df, print_success_message

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================

NUM_OBLIGORS = 1_000
NUM_SCENARIOS = 100_000

# Distribution of models
N_FORCED_DEFAULT = int(0.0 * NUM_OBLIGORS)  # Forced-Default (Stress/Defaulted)
N_GAUSS = int(0.0 * NUM_OBLIGORS)  # Gaussian Copula
N_T_COP = int(0.0 * NUM_OBLIGORS)  # Student-t Copula
N_INDEPENDENT_BERNOULLI = int(0.0 * NUM_OBLIGORS)  # Independent Bernoulli
N_STRESS_REGIME_BERNOULLI = int(0.0 * NUM_OBLIGORS)  # Stress Regime
N_BETA_BINOMIAL = int(1.0 * NUM_OBLIGORS)  # Beta Binomial

_ASSINGNED = (
    N_FORCED_DEFAULT
    + N_GAUSS
    + N_T_COP
    + N_INDEPENDENT_BERNOULLI
    + N_STRESS_REGIME_BERNOULLI
    + N_BETA_BINOMIAL
)

# Correct rounding remainders if any (not strictly needed for 1000, but good practice)
if _ASSINGNED != NUM_OBLIGORS:
    N_BETA_BINOMIAL += NUM_OBLIGORS - _ASSINGNED

GLOBAL_T_COP_NU = 5.0  # Degrees of freedom for the common T-Copula chi-square factor


def _plot_portfolio_losses(
    portfolio_losses: np.ndarray,
    expected_loss: float,
    credit_var_99: float,
    worst_loss: float,
    credit_es_99: float,
) -> None:
    """Plots the aggregated portfolio loss distribution and risk metrics."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    losses_m = portfolio_losses / 1e6
    el_m = expected_loss / 1e6
    credit_var_99_m = credit_var_99 / 1e6
    # worst_m = worst_loss / 1e6

    ax.hist(
        losses_m,
        bins=150,
        color="#3a7ca5",
        edgecolor="#294d69",
        alpha=0.8,
        density=True,
    )

    ax.axvline(
        el_m,
        color="#2ca02c",
        linestyle="dashed",
        linewidth=2.5,
        label=f"Expected Loss (EL): ${el_m:,.1f}M",
    )
    ax.axvline(
        credit_var_99_m,
        color="#d62728",
        linestyle="solid",
        linewidth=2.5,
        label=f"99% Credit VaR: ${credit_var_99_m:,.1f}M",
    )

    ax.axvline(
        credit_es_99 / 1e6,
        color="orange",
        linewidth=2.5,
        label=f"99% Credit ES: ${credit_es_99 / 1e6:,.1f}M",
    )

    ax.set_title("E2E JTD Portfolio Loss Distribution", fontsize=14, pad=15)
    ax.set_xlabel("Simulated Portfolio Loss ($ Millions)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}M"))
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.show()


def run_demo() -> None:
    print_calling_file()
    print("--- BETA-BINOMIAL JTD ENGINE VALIDATION DEMO ---")
    print(f"Obligors:  {NUM_OBLIGORS}")
    print(f"Scenarios: {NUM_SCENARIOS:,}")
    print(
        f"Model Mix: \n"
        f" - {N_FORCED_DEFAULT} Forced-Default \n"
        f" - {N_INDEPENDENT_BERNOULLI} Independent Bernoulli \n"
        f" - {N_STRESS_REGIME_BERNOULLI} Stress Regime Bernoulli \n"
        f" - {N_BETA_BINOMIAL} Beta Binomial \n"
        f" - {N_GAUSS} Gaussian \n"
        f" - {N_T_COP} Student-t "
    )
    print("-" * 41 + "\n")

    rng = np.random.default_rng(seed=RNG_SEED)

    # ============================================================
    # 2. INGEST: OBLIGOR MASTER
    # ============================================================
    print("[1/5] Building & Compiling Obligor Master...")

    obligor_uids = np.arange(1, NUM_OBLIGORS + 1, dtype=np.int64)
    rng.shuffle(obligor_uids)  # Shuffle to prove the compiler sorts it

    # --- OBLIGOR PDS --- #

    # Base PDs
    # Realistic synthetic PD generation for 1000 names
    _obligor_pds = rng.lognormal(mean=np.log(0.02), sigma=10.0, size=NUM_OBLIGORS)
    _obligor_pds = np.clip(_obligor_pds, 0.0001, 0.50)

    # Define a concentration parameter. Lower = more overdispersion, higher = tighter.
    # kappa = 100.0 introduces highly realistic credit-spread overdispersion
    kappa = 100.0

    # Calculate vectorized alpha and beta arrays per obligor
    _alphas = _obligor_pds * kappa
    _betas = (1.0 - _obligor_pds) * kappa

    # Draw completely smooth, continuously overdispersed probabilities
    _obligor_pds = rng.beta(_alphas, _betas)

    # --- OBLIGOR LGDS --- #

    _obligor_lgds = rng.uniform(0.25, 0.75, NUM_OBLIGORS)

    obligor_df = pd.DataFrame(
        {
            "uid": obligor_uids,
            "pd": _obligor_pds,
            "lgd": _obligor_lgds,
            "rating_id": 0,
            "sector_id": 0,
            "country_id": 0,
            "parent_uid": None,  # Ultimates
        }
    )

    obligor_master = compile_obligor_master(obligor_df)

    # ============================================================
    # 3. INGEST: JTD PORTFOLIO
    # ============================================================
    print("[2/5] Building & Compiling JTD Portfolio...")

    models = (
        [JtdModel.COPULA_SF_GAUSSIAN.name] * N_GAUSS
        + [JtdModel.COPULA_SF_STUDENTS_T.name] * N_T_COP
        + [JtdModel.FORCED_DEFAULT.name] * N_FORCED_DEFAULT
        + [JtdModel.INDEPENDENT_BERNOULLI.name] * N_INDEPENDENT_BERNOULLI
        + [JtdModel.STRESS_REGIME_BERNOULLI.name] * N_STRESS_REGIME_BERNOULLI
        + [JtdModel.BETA_BINOMIAL.name] * N_BETA_BINOMIAL
    )

    factor_names = (
        ["GLOBAL_CREDIT"]
        + [f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_CHI"]
        + [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_STRESS_REGIME"]
        + ["GLOBAL_BETA_PD"]
    )

    # --- CALIBRATE THE BETA BINOMIAL PARAMETERS ---
    _obligor_mean_pd = np.mean(_obligor_pds)
    _obligor_var_pd = np.var(_obligor_pds)

    # Guardrail 1: Prevent negative/zero 'total' values if cross-sectional variance is too high
    max_theoretical_var = _obligor_mean_pd * (1.0 - _obligor_mean_pd)
    if _obligor_var_pd >= max_theoretical_var:
        # Cap the variance at a safe 75% of the absolute mathematical maximum
        _obligor_var_pd = 0.75 * max_theoretical_var

    total = (_obligor_mean_pd * (1 - _obligor_mean_pd) / _obligor_var_pd) - 1
    alpha_mom = _obligor_mean_pd * total
    beta_mom = (1 - _obligor_mean_pd) * total

    # Guardrail 2: Prevent alpha < 1.0 which causes unstable U-shaped density spikes at the boundaries
    if alpha_mom < 1.0:
        alpha_mom = 1.5
        beta_mom = alpha_mom * ((1.0 - _obligor_mean_pd) / _obligor_mean_pd)

    print(
        f"\nCalibrated Beta-Binomial parameters to match obligor mean PD ({100 * _obligor_mean_pd:.2f}%):"
        f"\n- alpha: {alpha_mom:>8.2f}"
        f"\n- beta:  {beta_mom:>8.2f}"
    )

    factor_universe_df = pd.DataFrame(
        {
            "factor_id": range(len(factor_names)),
            "factor_name": factor_names,
            "factor_kind": [
                "SYSTEMIC"
                if n == "GLOBAL_CREDIT"
                else "CHI"
                if (n == "GLOBAL_CHI" or n.startswith("CHI_"))
                else "UNIFORM"
                if n.startswith("U_")
                else "STRESS_INDICATOR"
                if n == "GLOBAL_STRESS_REGIME"
                else "BETA_PD"
                if n == "GLOBAL_BETA_PD"
                else "IDIOSYNCRATIC"
                for n in factor_names
            ],
            "nu": [
                GLOBAL_T_COP_NU if (n == "GLOBAL_CHI" or n.startswith("CHI_")) else None
                for n in factor_names
            ],
            # Use the new shape parameters we locked in earlier to avoid beta naming collision
            "alpha": [
                alpha_mom if n == "GLOBAL_BETA_PD" else None for n in factor_names
            ],
            "beta": [beta_mom if n == "GLOBAL_BETA_PD" else None for n in factor_names],
        }
    )
    universe = compile_credit_factor_universe(factor_universe_df)

    jtd_df = pd.DataFrame(
        {
            "uid": obligor_uids,  # Unsorted, matches the raw frame
            "instrument": models,
            "factor_loading": rng.uniform(0.1, 0.5, NUM_OBLIGORS),
            "systemic_factor": "GLOBAL_CREDIT",
            "idiosyncratic_factor": [
                f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)
            ],
            "uniform_factor": [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)],
            "stress_regime_factor": "GLOBAL_STRESS_REGIME",
            "stress_pd_multiplier": rng.uniform(1.25, 3.0, NUM_OBLIGORS),
            "beta_factor": "GLOBAL_BETA_PD",  # Will broadcast. Non-Beta-Binomial builders safely ignore it.
        }
    )

    # Add T-Copula specific columns
    chi_factors = (
        [None] * N_GAUSS
        + ["GLOBAL_CHI"] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_BETA_BINOMIAL
    )
    nus = (
        [None] * N_GAUSS
        + [GLOBAL_T_COP_NU] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_BETA_BINOMIAL
    )

    jtd_df["chi_squared_factor"] = chi_factors
    jtd_df["nu"] = nus

    print_df(
        jtd_df.head(n=10),
        title="Sample (head) JTD Portfolio DataFrame (Unsorted)",
    )
    print_df(
        jtd_df.tail(n=10),
        title="Sample (tail) JTD Portfolio DataFrame (Unsorted)",
    )

    _jtd_bundle = JtdIngestBundle.from_instrument_frame(jtd_df)
    _masters = CompiledCreditMasters(
        obligor_master=obligor_master,
        factor_universe=universe,
    )
    compiled_portfolio = compile_jtd_portfolio(
        bundle=_jtd_bundle,
        masters=_masters,
    )

    compiled_portfolio.summary()

    # ============================================================
    # 4. SYNTHESIZE EAD (ALIGNED TO COMPILED PORTFOLIO)
    # ============================================================
    print("[3/5] Generating Exposure Matrix...")

    ead_by_obligor = np.empty(
        (NUM_OBLIGORS, NUM_SCENARIOS), dtype=np.float64, order="C"
    )

    base_ead = rng.uniform(1e6, 10e6, (NUM_OBLIGORS, 1))
    ead_by_obligor[:] = base_ead  # Broadcast across scenarios

    ead_by_obligor *= rng.lognormal(
        mean=0.0, sigma=0.1, size=(NUM_OBLIGORS, NUM_SCENARIOS)
    )

    # ============================================================
    # 5. SYNTHESIZE FACTOR MATRIX
    # ============================================================
    print("[4/5] Generating Credit Factor Matrix...")

    factor_generator = MixedCreditFactorGenerator.from_universe(
        universe,
        total_scenarios=NUM_SCENARIOS,
        seed=RNG_SEED,
        stress_p_by_uix={int(u): 0.05 for u in universe.stress_indicator_uix},
    )
    factor_matrix = build_credit_factor_matrix(
        factor_generator.iter_batches(batch_size=min(NUM_SCENARIOS, 25_000)),
        num_factors=universe.n_factors,
        num_scenarios=NUM_SCENARIOS,
    )

    # ============================================================
    # 6. ENGINE EXECUTION
    # ============================================================
    print("[5/5] Executing `run_jtd` Engine...")

    start_time = time.perf_counter()

    jtd_result = run_jtd(
        jtd_portfolio=compiled_portfolio,
        ead_by_obligor=ead_by_obligor,
        credit_factors=factor_matrix,
    )

    elapsed = time.perf_counter() - start_time
    print(
        f"Engine completed {NUM_OBLIGORS * NUM_SCENARIOS:,} "
        f"evaluations in {elapsed:.4f}s"
    )

    # ============================================================
    # 7. METRICS & ANALYSIS
    # ============================================================
    print("\n--- Portfolio Credit Risk Metrics ---")

    portfolio_losses = np.sum(jtd_result.loss, axis=0)

    expected_loss = np.mean(portfolio_losses)
    credit_var_99 = np.quantile(portfolio_losses, 0.99)
    credit_es_99 = portfolio_losses[portfolio_losses >= credit_var_99].mean()

    print(f"Expected Loss (EL):  ${expected_loss:,.2f}")
    print(f"Credit VaR (99.0%):  ${credit_var_99:,.2f}")
    print(f"Worst Scenario Loss: ${np.max(portfolio_losses):,.2f}")
    print(f"Credit ES (99.0%): ${credit_es_99:,.2f}")

    # 8. PLOTS
    _plot_portfolio_losses(
        portfolio_losses,
        expected_loss,
        credit_var_99,
        np.max(portfolio_losses),
        credit_es_99,
    )

    # ============================================================
    # DEFAULT STATS
    # ============================================================

    default_indicators = jtd_result.loss > 0.0
    defaults_per_scenario = default_indicators.sum(axis=0)

    print("\n--- Default Statistics ---")
    print(f"Average Defaults: {defaults_per_scenario.mean():.2f}")
    print(f"99% Default Count: {np.quantile(defaults_per_scenario, 0.99):.0f}")
    print(f"Max Defaults:      {defaults_per_scenario.max():.0f}")

    print("\n--- Default Clustering Statistics ---")
    print(f"Average Defaults:      {defaults_per_scenario.mean():.2f}")
    print(f"95% Default Count:     {np.quantile(defaults_per_scenario, 0.95):.0f}")
    print(f"99% Default Count:     {np.quantile(defaults_per_scenario, 0.99):.0f}")
    print(f"99.9% Default Count:   {np.quantile(defaults_per_scenario, 0.999):.0f}")
    print(f"Max Defaults:          {defaults_per_scenario.max():.0f}")

    # ============================================================
    # VaR Attribution
    # ============================================================

    var_idx = np.argmin(np.abs(portfolio_losses - credit_var_99))
    obligor_contributions = jtd_result.loss[:, var_idx]
    sorted_idx = np.argsort(obligor_contributions)[::-1]

    top_n = 10
    print(f"\n--- Top {top_n} Contributors to 99% Credit VaR ---")
    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        uid = compiled_portfolio.obligor_uids[idx]
        contrib = obligor_contributions[idx]
        print(f"{rank:2d}. Obligor {uid:<4d} Contribution: ${contrib:,.2f}")

    print(f"\nTotal Portfolio Loss in VaR Scenario: ${portfolio_losses[var_idx]:,.2f}")

    # ============================================================
    # Expected Shortfall (99%) Attribution
    # ============================================================

    tail_mask = portfolio_losses >= credit_var_99
    obligor_contributions = jtd_result.loss[:, tail_mask].mean(axis=1)
    sorted_idx = np.argsort(obligor_contributions)[::-1]

    top_n = 10
    print(f"\n--- Top {top_n} Contributors to 99% Credit Expected Shortfall ---")
    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        uid = compiled_portfolio.obligor_uids[idx]
        contrib = obligor_contributions[idx]
        print(f"{rank:2d}. Obligor {uid:<4d} Contribution: ${contrib:,.2f}")

    credit_es_99 = portfolio_losses[tail_mask].mean()
    print(f"\n99% Credit Expected Shortfall: ${credit_es_99:,.2f}")

    top10_share = (
        obligor_contributions[sorted_idx[:10]].sum() / obligor_contributions.sum()
    )
    print(f"Top 10 names explain {100 * top10_share:.1f}% of Credit Expected Shortfall")

    # ============================================================
    # BETA-BINOMIAL VALIDATION
    # ============================================================
    # Theoretical common-PD mean: E[P_s] = alpha / (alpha + beta).
    theoretical_mean_pd = alpha_mom / (alpha_mom + beta_mom)

    # Per-scenario realized default rate across the (100% beta-binomial) book.
    default_indicators = jtd_result.loss > 0.0
    default_rate_per_scenario = default_indicators.mean(axis=0)
    empirical_mean_pd = float(default_rate_per_scenario.mean())

    # Signature of the common-shock model: the default RATE is itself stochastic
    # (it IS P_s), so its cross-scenario variance is large. An independent-
    # Bernoulli book of the same size would pin the rate near its mean, with
    # variance ~ p(1 - p) / N.
    independent_rate_var = (
        theoretical_mean_pd * (1.0 - theoretical_mean_pd) / NUM_OBLIGORS
    )
    realized_rate_var = float(default_rate_per_scenario.var())

    print("\n--- Beta-Binomial Validation ---")
    print(f"Theoretical mean PD  E[P_s] = a/(a+b): {theoretical_mean_pd:.4%}")
    print(f"Empirical mean default rate:           {empirical_mean_pd:.4%}")
    print(f"Realized default-rate variance:        {realized_rate_var:.3e}")
    print(f"Independent-Bernoulli baseline var:    {independent_rate_var:.3e}")
    print(
        f"Overdispersion ratio:                  "
        f"{realized_rate_var / independent_rate_var:,.1f}x"
    )

    print_success_message()


if __name__ == "__main__":
    run_demo()