# scripts/demos/credit/jtd/e2e/demo_e2e_full_jtd_portfolio.py
"""
End-to-End Mixed JTD Credit Engine Demo.

Exercises the full JTD model set in one portfolio (forced-default, independent
and stress-regime Bernoulli, beta-binomial, single-factor Gaussian and Student-t
copulas, and the multi-factor Gaussian and Student-t copulas) through the
bundle ingest path:
    Raw DataFrames -> JtdIngestBundle -> Compilation -> Matrix -> Engine -> Metrics

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
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.cross_master.compiled import (
    CompiledCreditMasters,
)
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
N_FORCED_DEFAULT = int(0.05 * NUM_OBLIGORS)  # Forced Default
_OBLIGORS_REMAINING = NUM_OBLIGORS - N_FORCED_DEFAULT

# Distribute the remaining across 5 models
_FACTOR = 1 / 7
N_INDEPENDENT_BERNOULLI = int(_FACTOR * _OBLIGORS_REMAINING)  # Independent Bernoulli
N_STRESS_REGIME_BERNOULLI = int(_FACTOR * _OBLIGORS_REMAINING)  # Stress Regime
N_BETA_BINOMIAL = int(_FACTOR * _OBLIGORS_REMAINING)  # Beta Binomial
N_GAUSS = int(_FACTOR * _OBLIGORS_REMAINING)  # SF Gaussian Copula
N_T_COP = int(_FACTOR * _OBLIGORS_REMAINING)  # SF Student-t Copula
N_MF_GAUSSIAN = int(_FACTOR * _OBLIGORS_REMAINING)  # Multi-Factor Gaussian Copula
N_MF_STUDENTS_T = int(_FACTOR * _OBLIGORS_REMAINING)  # Multi-Factor Student's T Copula

_ASSINGNED = (
    N_FORCED_DEFAULT
    + N_INDEPENDENT_BERNOULLI
    + N_STRESS_REGIME_BERNOULLI
    + N_BETA_BINOMIAL
    + N_GAUSS
    + N_T_COP
    + N_MF_GAUSSIAN
    + N_MF_STUDENTS_T
)

# Correct rounding remainders if any (not strictly needed for 1000, but good practice)
if _ASSINGNED != NUM_OBLIGORS:
    N_BETA_BINOMIAL += NUM_OBLIGORS - _ASSINGNED

GLOBAL_T_COP_NU = 5.0  # Degrees of freedom for the common T-Copula chi-square factor

# Systemic sector factors the multi-factor obligors load on (independent normals
# in the universe; any cross-sector correlation is absorbed into the loadings).
MF_SECTOR_FACTORS = ("SECTOR_ENERGY", "SECTOR_FINANCIALS", "SECTOR_TECH")

MF_CHI_SECTOR_NUS = {
    "CHI_SECTOR_ENERGY": 5.0,
    "CHI_SECTOR_FINANCIALS": 5.0,
    "CHI_SECTOR_TECH": 5.0,
}


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
    worst_m = worst_loss / 1e6

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


def _plot_model_distributions(model_losses: dict) -> None:
    """Plots the loss distributions of the individual JTD models on separate axes."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.style.use("dark_background")
    # Grid sized to the model count (4 per row).
    n_models = len(model_losses)
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )
    fig.suptitle("Component Loss Distributions by Model", fontsize=18)

    axes = axes.flatten()
    for ax in axes[n_models:]:
        ax.set_visible(False)

    for ax, (model_name, losses) in zip(axes, model_losses.items()):
        losses_m = losses / 1e6

        # Plot the histogram (gracefully handling zero-variance/no-loss edge cases)
        ax.hist(
            losses_m,
            bins=100 if np.var(losses_m) > 0 else 10,
            color="#3a7ca5",
            edgecolor="#294d69",
            alpha=0.8,
            density=True,
        )

        # Calculate metrics for the lines
        el_m = losses_m.mean()
        var_99_m = np.quantile(losses_m, 0.99)
        es_99_m = (
            losses_m[losses_m >= var_99_m].mean()
            if np.any(losses_m >= var_99_m)
            else 0.0
        )

        # Draw risk metric lines
        ax.axvline(
            el_m,
            color="#2ca02c",
            linestyle="dashed",
            linewidth=2.5,
            label=f"EL: ${el_m:,.1f}M",
        )
        ax.axvline(
            var_99_m,
            color="#d62728",
            linestyle="solid",
            linewidth=2.5,
            label=f"99% VaR: ${var_99_m:,.1f}M",
        )
        ax.axvline(
            es_99_m,
            color="orange",
            linestyle="solid",
            linewidth=2.5,
            label=f"99% ES: ${es_99_m:,.1f}M",
        )

        ax.set_title(model_name, fontsize=14, pad=10)
        ax.set_xlabel("Loss ($ Millions)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}M"))

        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper right", fontsize=10)

    # Call standard tight_layout first to handle the internal subplot labels
    plt.tight_layout()

    # Then explicitly carve out 10% of the top of the figure for the suptitle,
    # and add a bit of breathing room between the rows/columns.
    fig.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)

    plt.show()


def run_demo() -> None:
    print_calling_file()
    print("--- S-TIER COPULA JTD ENGINE E2E DEMO ---")
    print(f"Obligors:  {NUM_OBLIGORS}")
    print(f"Scenarios: {NUM_SCENARIOS:,}")
    print(
        f"Model Mix: \n"
        f" - {N_FORCED_DEFAULT} Forced-Default \n"
        f" - {N_INDEPENDENT_BERNOULLI} Independent Bernoulli \n"
        f" - {N_STRESS_REGIME_BERNOULLI} Stress Regime Bernoulli \n"
        f" - {N_BETA_BINOMIAL} Beta Binomial \n"
        f" - {N_GAUSS} Single-Factor Gaussian \n"
        f" - {N_T_COP} Single-Factor Student-t \n"
        f" - {N_MF_GAUSSIAN} Multi-Factor Gaussian \n"
        f" - {N_MF_STUDENTS_T} Multi-Factor Student's T"
    )
    print("-" * 41 + "\n")

    rng = np.random.default_rng(seed=RNG_SEED)

    # ============================================================
    # 2. INGEST: OBLIGOR MASTER
    # ============================================================
    print("[1/5] Building & Compiling Obligor Master...")

    obligor_uids = np.arange(1, NUM_OBLIGORS + 1, dtype=np.int64)
    rng.shuffle(obligor_uids)  # Shuffle to prove the compiler sorts it

    # Define models early so we can isolate the Non-Forced-Defaults
    models = (
        [JtdModel.COPULA_SF_GAUSSIAN.name] * N_GAUSS
        + [JtdModel.COPULA_SF_STUDENTS_T.name] * N_T_COP
        + [JtdModel.FORCED_DEFAULT.name] * N_FORCED_DEFAULT
        + [JtdModel.INDEPENDENT_BERNOULLI.name] * N_INDEPENDENT_BERNOULLI
        + [JtdModel.STRESS_REGIME_BERNOULLI.name] * N_STRESS_REGIME_BERNOULLI
        + [JtdModel.BETA_BINOMIAL.name] * N_BETA_BINOMIAL
        + [JtdModel.COPULA_MF_GAUSSIAN.name] * N_MF_GAUSSIAN
        + [JtdModel.COPULA_MF_STUDENTS_T.name] * N_MF_STUDENTS_T
    )
    models_array = np.array(models)
    non_fd_mask = models_array != JtdModel.FORCED_DEFAULT.name

    # --- OBLIGOR PDS --- #

    # Base PDs (Individual random generation, kept for forced defaults)
    _obligor_pds = rng.lognormal(mean=np.log(0.02), sigma=10.0, size=NUM_OBLIGORS)
    _obligor_pds = np.clip(_obligor_pds, 0.0001, 0.50)

    # Define a concentration parameter.
    kappa = 100.0

    # Calculate vectorized alpha and beta arrays per obligor
    _alphas = _obligor_pds * kappa
    _betas = (1.0 - _obligor_pds) * kappa

    # Draw completely smooth, continuously overdispersed probabilities
    _obligor_pds = rng.beta(_alphas, _betas)

    # --- OBLIGOR LGDS --- #
    _obligor_lgds = rng.uniform(0.25, 0.75, NUM_OBLIGORS)

    # --- APPLY HOMOGENEOUS NON-FORCED DEFAULT OVERRIDE --- #
    # Generate exactly ONE safe, strictly positive random PD and LGD
    shared_pd = rng.uniform(0.01, 0.05)  # e.g., a random PD between 1% and 5%
    shared_lgd = rng.uniform(0.25, 0.75)

    # Apply the single PD and LGD to everyone who is NOT a forced default
    _obligor_pds[non_fd_mask] = shared_pd
    _obligor_lgds[non_fd_mask] = shared_lgd

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
        + [JtdModel.COPULA_MF_GAUSSIAN.name] * N_MF_GAUSSIAN
        + [JtdModel.COPULA_MF_STUDENTS_T.name] * N_MF_STUDENTS_T
    )

    factor_names = (
        ["GLOBAL_CREDIT"]
        + [f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_CHI"]
        + [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_STRESS_REGIME"]
        + ["GLOBAL_BETA_PD"]
        + list(MF_SECTOR_FACTORS)
        + list(MF_CHI_SECTOR_NUS.keys())
    )

    # --- CALIBRATE THE BETA BINOMIAL PARAMETERS ---
    _obligor_mean_pd = float(shared_pd)

    # Guardrail 1: Define the absolute mathematical ceiling for variance
    max_theoretical_var = _obligor_mean_pd * (1.0 - _obligor_mean_pd)

    # Guardrail 2: Inject synthetic variance to prevent divide-by-zero.
    # Because all non-forced PDs are identical, actual variance is 0.0.
    # We set variance to 2% of the maximum limit to create moderate contagion.
    _obligor_var_pd = 0.02 * max_theoretical_var

    total = (max_theoretical_var / _obligor_var_pd) - 1
    alpha_mom = _obligor_mean_pd * total
    beta_mom = (1.0 - _obligor_mean_pd) * total

    # Guardrail 3: Prevent alpha < 1.0 which causes unstable U-shaped density spikes
    if alpha_mom < 1.0:
        alpha_mom = 1.5
        beta_mom = alpha_mom * ((1.0 - _obligor_mean_pd) / _obligor_mean_pd)

    print(
        f"\nCalibrated Beta-Binomial parameters to match shared obligor PD ({100 * _obligor_mean_pd:.2f}%):"
        f"\n- alpha: {alpha_mom:>8.2f}"
        f"\n- beta:  {beta_mom:>8.2f}"
    )

    factor_universe_df = pd.DataFrame(
        {
            "factor_id": range(len(factor_names)),
            "factor_name": factor_names,
            "factor_kind": [
                "SYSTEMIC"
                if (n == "GLOBAL_CREDIT" or n.startswith("SECTOR_"))
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
                GLOBAL_T_COP_NU if n == "GLOBAL_CHI" else MF_CHI_SECTOR_NUS.get(n)
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
            "systemic_factor": "GLOBAL_CREDIT",  # Broadcast to all
            "idiosyncratic_factor": [
                f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)
            ],
            "uniform_factor": [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)],
            "stress_regime_factor": "GLOBAL_STRESS_REGIME",
            "stress_pd_multiplier": rng.uniform(1.25, 3.0, NUM_OBLIGORS),
            "beta_factor": "GLOBAL_BETA_PD",  # Will broadcast. Non-Beta-Binomial builders safely ignore it.
        }
    )

    # Multi-factor Student-t obligors: each gets the sector chi-square factor for
    # the sector it loads on (per-sector tail dependence), with nu reconciling to
    # that chi factor's reference nu in the universe.
    _mft_sectors = [
        MF_SECTOR_FACTORS[i % len(MF_SECTOR_FACTORS)] for i in range(N_MF_STUDENTS_T)
    ]
    _mft_chi = ["CHI_" + sector for sector in _mft_sectors]
    _mft_nu = [MF_CHI_SECTOR_NUS[chi] for chi in _mft_chi]

    # Add T-Copula specific columns
    chi_factors = (
        [None] * N_GAUSS
        + ["GLOBAL_CHI"] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_BETA_BINOMIAL
        + [None] * N_MF_GAUSSIAN
        + _mft_chi
    )
    nus = (
        [None] * N_GAUSS
        + [GLOBAL_T_COP_NU] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_BETA_BINOMIAL
        + [None] * N_MF_GAUSSIAN
        + _mft_nu
    )

    jtd_df["chi_squared_factor"] = chi_factors
    jtd_df["nu"] = nus

    # --- MULTI-FACTOR LOADINGS (long format, one row per obligor x systemic) ---
    # Every MF obligor loads on GLOBAL_CREDIT plus one sector; sum beta^2 < 1
    # (room for the idiosyncratic residual), with the sector assigned round-robin.
    mf_uids = jtd_df.loc[
        jtd_df["instrument"] == JtdModel.COPULA_MF_GAUSSIAN.name, "uid"
    ].to_numpy()
    _n_mf = mf_uids.shape[0]
    _global_betas = rng.uniform(0.20, 0.40, _n_mf)
    _sector_betas = rng.uniform(0.30, 0.60, _n_mf)
    _sectors = np.array(
        [MF_SECTOR_FACTORS[i % len(MF_SECTOR_FACTORS)] for i in range(_n_mf)],
        dtype=object,
    )
    mf_loadings_df = pd.DataFrame(
        {
            "obligor_uid": np.concatenate([mf_uids, mf_uids]),
            "factor_name": np.concatenate(
                [np.array(["GLOBAL_CREDIT"] * _n_mf, dtype=object), _sectors]
            ),
            "factor_loading": np.concatenate([_global_betas, _sector_betas]),
        }
    )

    # Multi-factor Student-t obligors load the same way (GLOBAL_CREDIT + one
    # sector); their sector here must match the one driving their chi factor.
    mft_uids = jtd_df.loc[
        jtd_df["instrument"] == JtdModel.COPULA_MF_STUDENTS_T.name, "uid"
    ].to_numpy()
    _n_mft = mft_uids.shape[0]
    _mft_global_betas = rng.uniform(0.20, 0.40, _n_mft)
    _mft_sector_betas = rng.uniform(0.30, 0.60, _n_mft)
    _mft_sectors_arr = np.array(_mft_sectors, dtype=object)
    mft_loadings_df = pd.DataFrame(
        {
            "obligor_uid": np.concatenate([mft_uids, mft_uids]),
            "factor_name": np.concatenate(
                [np.array(["GLOBAL_CREDIT"] * _n_mft, dtype=object), _mft_sectors_arr]
            ),
            "factor_loading": np.concatenate([_mft_global_betas, _mft_sector_betas]),
        }
    )
    mf_loadings_df = pd.concat([mf_loadings_df, mft_loadings_df], ignore_index=True)

    print_df(
        jtd_df.head(n=10),
        title="Sample (head) JTD Portfolio DataFrame (Unsorted)",
    )
    print_df(
        jtd_df.tail(n=10),
        title="Sample (tail) JTD Portfolio DataFrame (Unsorted)",
    )

    bundle = JtdIngestBundle.from_instrument_frame(
        jtd_df, mf_loadings_df=mf_loadings_df
    )
    _masters = CompiledCreditMasters(
        obligor_master=obligor_master,
        factor_universe=universe,
    )
    compiled_portfolio = compile_jtd_portfolio(
        bundle=bundle,
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
    # MODEL COMPARISON: FORCED-DEFAULT VS OTHERS
    # ============================================================

    model_series = jtd_df.set_index("uid")["instrument"]
    aligned_models = model_series.loc[
        compiled_portfolio.obligor_uids.tolist()
    ].to_numpy()

    forced_default_mask = aligned_models == JtdModel.FORCED_DEFAULT.name
    independent_bernoulli_mask = aligned_models == JtdModel.INDEPENDENT_BERNOULLI.name
    stress_regime_bernoulli_mask = (
        aligned_models == JtdModel.STRESS_REGIME_BERNOULLI.name
    )
    beta_binomial_mask = aligned_models == JtdModel.BETA_BINOMIAL.name
    gaussian_mask = aligned_models == JtdModel.COPULA_SF_GAUSSIAN.name
    t_mask = aligned_models == JtdModel.COPULA_SF_STUDENTS_T.name
    mf_gaussian_mask = aligned_models == JtdModel.COPULA_MF_GAUSSIAN.name
    mf_students_t_mask = aligned_models == JtdModel.COPULA_MF_STUDENTS_T.name

    forced_default_losses = jtd_result.loss[forced_default_mask].sum(axis=0)
    independent_bernoulli_losses = jtd_result.loss[independent_bernoulli_mask].sum(
        axis=0
    )
    stress_regime_bernoulli_losses = jtd_result.loss[stress_regime_bernoulli_mask].sum(
        axis=0
    )
    beta_binomial_losses = jtd_result.loss[beta_binomial_mask].sum(axis=0)
    gaussian_losses = jtd_result.loss[gaussian_mask].sum(axis=0)
    t_losses = jtd_result.loss[t_mask].sum(axis=0)
    mf_gaussian_losses = jtd_result.loss[mf_gaussian_mask].sum(axis=0)
    mf_students_t_losses = jtd_result.loss[mf_students_t_mask].sum(axis=0)

    def _calc_metrics(losses):
        el = losses.mean()
        var = np.quantile(losses, 0.99)
        es = losses[losses >= var].mean()
        return el, var, es

    fd_el, fd_var, fd_es = _calc_metrics(forced_default_losses)
    ib_el, ib_var, ib_es = _calc_metrics(independent_bernoulli_losses)
    srb_el, srb_var, srb_es = _calc_metrics(stress_regime_bernoulli_losses)
    bb_el, bb_var, bb_es = _calc_metrics(beta_binomial_losses)
    gauss_el, gauss_var, gauss_es = _calc_metrics(gaussian_losses)
    t_el, t_var, t_es = _calc_metrics(t_losses)
    mfg_el, mfg_var, mfg_es = _calc_metrics(mf_gaussian_losses)
    mft_el, mft_var, mft_es = _calc_metrics(mf_students_t_losses)

    print("\n" + "=" * 91)
    print(f"{'MODEL COMPARISON SUMMARY':^91}")
    print("=" * 91)
    print(
        f"{'Model Type':<26} | {'Expected Loss (EL)':>18} | {'99% Credit VaR':>18} | {'99% Credit ES':>18}"
    )
    print("-" * 91)
    print(
        f"{'Forced-Default':<26} | ${fd_el:>17,.2f} | ${fd_var:>17,.2f} | ${fd_es:>17,.2f}"
    )
    print(
        f"{'Independent Bernoulli':<26} | ${ib_el:>17,.2f} | ${ib_var:>17,.2f} | ${ib_es:>17,.2f}"
    )
    print(
        f"{'Stress Regime Bernoulli':<26} | ${srb_el:>17,.2f} | ${srb_var:>17,.2f} | ${srb_es:>17,.2f}"
    )
    print(
        f"{'Beta Binomial':<26} | ${bb_el:>17,.2f} | ${bb_var:>17,.2f} | ${bb_es:>17,.2f}"
    )
    print(
        f"{'Gaussian Copula':<26} | ${gauss_el:>17,.2f} | ${gauss_var:>17,.2f} | ${gauss_es:>17,.2f}"
    )
    print(
        f"{'Student-t Copula':<26} | ${t_el:>17,.2f} | ${t_var:>17,.2f} | ${t_es:>17,.2f}"
    )
    print(
        f"{'Multi-Factor Gaussian':<26} | ${mfg_el:>17,.2f} | ${mfg_var:>17,.2f} | ${mfg_es:>17,.2f}"
    )
    print(
        f"{'Multi-Factor Student-t':<26} | ${mft_el:>17,.2f} | ${mft_var:>17,.2f} | ${mft_es:>17,.2f}"
    )

    print("=" * 91)

    # --- PLOT DISTRIBUTIONS PER MODEL ---
    model_losses_dict = {
        "Forced-Default": forced_default_losses,
        "Independent Bernoulli": independent_bernoulli_losses,
        "Stress Regime Bernoulli": stress_regime_bernoulli_losses,
        "Beta Binomial": beta_binomial_losses,
        "Gaussian Copula": gaussian_losses,
        "Student-t Copula": t_losses,
        "Multi-Factor Gaussian": mf_gaussian_losses,
        "Multi-Factor Student-t": mf_students_t_losses,
    }
    _plot_model_distributions(model_losses_dict)

    print_success_message()


if __name__ == "__main__":
    run_demo()
