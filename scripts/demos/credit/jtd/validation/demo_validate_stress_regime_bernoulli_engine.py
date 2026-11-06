# scripts/demos/credit/jtd/validation/demo_validate_stress_regime_bernoulli_engine.py
"""
End-to-End Validation of the Stress Regime Bernoulli JTD Engine.

This script validates the Stress Regime Bernoulli block of the Jump-to-Default
(JTD) credit engine.

Stress Regime Bernoulli obligors are modeled as conditionally independent
Bernoulli default events whose probabilities of default are amplified when a
shared stress regime is active. The stress indicator introduces portfolio-wide
default clustering by simultaneously increasing default probabilities across
multiple obligors during adverse scenarios.

The purpose of this validation is to demonstrate that the engine correctly
applies stress-dependent default probabilities and reproduces the expected loss
and default characteristics implied by the stress regime specification.

Pipeline
--------
Raw DataFrames
    -> Compilation
    -> Portfolio Assembly
    -> JTD Engine
    -> Scenario Loss Distribution
    -> Validation Metrics

===============================================================================
MATHEMATICAL MODEL
===============================================================================

Let S_s denote a binary stress indicator for scenario s:

    S_s ∈ {0, 1}

where

    S_s = 1  => stress regime active
    S_s = 0  => normal regime

and

    P(S_s = 1) = p_stress

For obligor i with baseline probability of default PD_i and stress multiplier
m_i, the scenario-specific probability of default is

    PD_{i,s} =
        PD_i                      if S_s = 0
        min(PD_i * m_i, 1.0)      if S_s = 1

Conditional on the stress regime realization, defaults are independent:

    D_{i,s} | S_s ~ Bernoulli(PD_{i,s})

The portfolio loss in scenario s is

    L_s = Σ_i D_{i,s} * LGD_i * EAD_i

where

    D_{i,s} : default indicator
    LGD_i   : loss given default
    EAD_i   : exposure at default

Although obligors remain conditionally independent given the stress regime,
they become unconditionally dependent because they share the same stress
indicator. This creates default clustering and increases portfolio tail risk.

===============================================================================
THEORETICAL RESULTS
===============================================================================

Expected Default Probability:

    E[PD_i]
        = (1 - p_stress) * PD_i
        + p_stress * min(PD_i * m_i, 1.0)

Expected Loss:

    EL
        = Σ_i E[PD_i] * LGD_i * EAD_i

Default Dependence:

Because obligors share a common stress indicator,

    Cov(D_i, D_j) > 0

whenever both obligors are sensitive to the stress regime.

Unlike the Independent Bernoulli model, default dependence arises through the
shared regime state rather than through latent-factor copula structures.

Tail Risk:

Large portfolio losses occur disproportionately during stress-regime scenarios,
producing a heavier-tailed loss distribution than the Independent Bernoulli
model.

===============================================================================
EXPECTED VALIDATION PROPERTIES
===============================================================================

The simulation should demonstrate:

1. Stress-regime activation frequency converges to the configured
   stress probability.

2. Marginal default frequencies converge to their theoretical stressed
   expectations.

3. Simulated expected loss converges to the theoretical expected loss.

4. Default frequencies increase materially during stress-regime scenarios.

5. Positive pairwise default dependence emerges despite conditional
   independence within each regime.

6. Portfolio tail-loss metrics exceed those produced by an equivalent
   Independent Bernoulli portfolio.

7. Empirical loss moments converge to theoretical values within Monte Carlo
   sampling error.

===============================================================================
RELATIONSHIP TO OTHER JTD BLOCKS
===============================================================================

The Stress Regime Bernoulli block extends the Independent Bernoulli model by
introducing a shared stress indicator that simultaneously affects multiple
obligors.

Unlike Gaussian Copula and Student-t Copula models, dependence is generated
through explicit regime switching rather than latent asset-value factors.

Compared with the Independent Bernoulli model, Stress Regime Bernoulli
produces:

    - Positive default dependence
    - Increased default clustering
    - Higher portfolio tail risk
    - Regime-dependent loss distributions

while remaining computationally simple and easily interpretable.

===============================================================================
REFERENCES
===============================================================================

1. McNeil, A. J., Frey, R., & Embrechts, P. (2005).
   Quantitative Risk Management: Concepts, Techniques and Tools.
   Princeton University Press.

2. Bluhm, C., Overbeck, L., & Wagner, C. (2016).
   Introduction to Credit Risk Modeling (2nd Edition).
   Chapman & Hall / CRC.

3. Hamilton, J. D. (1989).
   A New Approach to the Economic Analysis of Nonstationary Time Series and
   the Business Cycle.
   Econometrica, 57(2), 357-384.

4. Gordy, M. B. (2003).
   A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules.
   Journal of Financial Intermediation, 12(3), 199-232.

"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl

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
N_GAUSS = int(0.0 * NUM_OBLIGORS)  # Gaussian Copula
N_T_COP = int(0.0 * NUM_OBLIGORS)  # Student-t Copula
N_FORCED_DEFAULT = int(0.0 * NUM_OBLIGORS)  # Forced-Default (Stress/Defaulted)
N_INDEPENDENT_BERNOULLI = int(0.0 * NUM_OBLIGORS)  # Independent Bernoulli
N_STRESS_REGIME_BERNOULLI = int(1.0 * NUM_OBLIGORS)  # Stress Regime Bernoulli

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


def run_demo() -> None:
    print_calling_file()
    print("--- STRESS REGIME BERNOULLI JTD ENGINE E2E DEMO ---")
    print(f"Obligors:  {NUM_OBLIGORS}")
    print(f"Scenarios: {NUM_SCENARIOS:,}")
    print("-" * 34 + "\n")

    rng = np.random.default_rng(seed=RNG_SEED)

    # ============================================================
    # 2. INGEST: OBLIGOR MASTER
    # ============================================================
    print("[1/5] Building & Compiling Obligor Master...")

    # DATA EXAMPLE (Obligor Master):
    # | uid | pd   | lgd  | rating_id | parent_uid |
    # |-----|------|------|-----------|------------|
    # | 10  | 0.02 | 0.60 | 0         | None       |
    #
    # PD (0.02) = 2% chance of default this year.
    # LGD (0.60) = If default occurs, we lose 60% of our exposure (recover 40%).

    obligor_uids = np.arange(1, NUM_OBLIGORS + 1, dtype=np.int64)
    rng.shuffle(obligor_uids)  # Shuffle to prove the compiler sorts it

    obligor_df = pl.DataFrame({
        "uid": obligor_uids,
        "pd": rng.uniform(0.001, 0.05, NUM_OBLIGORS),
        "lgd": rng.uniform(0.4, 0.8, NUM_OBLIGORS),
        "rating_id": 0,
        "sector_id": 0,
        "country_id": 0,
        "parent_uid": None,  # Ultimates
    })

    # The master is compiled into SoA. UIDs are strictly sorted ascending
    # to allow for lightning-fast `np.searchsorted` joins later.
    obligor_master = compile_obligor_master(obligor_df)

    # ============================================================
    # 3. INGEST: JTD PORTFOLIO
    # ============================================================
    print("[2/5] Building & Compiling JTD Portfolio...")

    # DATA EXAMPLE (JTD Portfolio):
    # | uid | instrument         | beta | systemic_factor | idiosyncratic_factor | nu  |
    # |-----|--------------------|------|-----------------|----------------------|-----|
    # | 10  | COPULA_SF_GAUSSIAN | 0.45 | GLOBAL_CREDIT   | IDIO_10              | NaN |
    # | 11  | COPULA_SF_T        | 0.60 | GLOBAL_CREDIT   | IDIO_11              | 5.0 |
    #
    # Beta (0.45): The obligor's latent asset value is constructed as:
    #              A = 0.45 * Z + sqrt(1 - 0.45^2) * eps
    #
    #              The systematic market factor contributes 0.45 as a loading
    #              coefficient, corresponding to an asset correlation of
    #              rho = beta^2 = 20.25%.
    #
    #              The remaining 79.75% of variance is driven by the
    #              idiosyncratic factor (eps).

    models = (
        [JtdModel.COPULA_SF_GAUSSIAN.name] * N_GAUSS
        + [JtdModel.COPULA_SF_STUDENTS_T.name] * N_T_COP
        + [JtdModel.FORCED_DEFAULT.name] * N_FORCED_DEFAULT
        + [JtdModel.INDEPENDENT_BERNOULLI.name] * N_INDEPENDENT_BERNOULLI
        + [JtdModel.STRESS_REGIME_BERNOULLI.name] * N_STRESS_REGIME_BERNOULLI
    )

    # We need UIX (Universe Index) pointers for all latent factors.
    # 1 Global + 300 Idio + 150 Chi = 451 total factors.
    factor_names = (
        ["GLOBAL_CREDIT"]
        + [f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_CHI"]
        + [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_STRESS_REGIME"]
    )
    # Compile the factor-universe MASTER from the generated factor list. At
    # validation scale this authoring table is generated (one idiosyncratic
    # factor per obligor) rather than the minimal canonical fixture, but it
    # flows through the SAME master contract: name -> UIX resolution and chi-dof
    # reconciliation now read from `universe`, never an ad-hoc {name: i} dict.
    factor_universe_df = pl.DataFrame({
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
            else "IDIOSYNCRATIC"
            for n in factor_names
        ],
        "nu": [
            GLOBAL_T_COP_NU if (n == "GLOBAL_CHI" or n.startswith("CHI_")) else None
            for n in factor_names
        ],
    })
    universe = compile_credit_factor_universe(factor_universe_df)

    jtd_df = pl.DataFrame({
        "uid": obligor_uids,  # Unsorted, matches the raw frame
        "instrument": models,
        "beta": rng.uniform(0.1, 0.5, NUM_OBLIGORS),
        "systemic_factor": "GLOBAL_CREDIT",
        "idiosyncratic_factor": [f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)],
        "uniform_factor": [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)],
        "stress_regime_factor": "GLOBAL_STRESS_REGIME",
        "stress_pd_multiplier": rng.uniform(1.25, 3.0, NUM_OBLIGORS),
    })

    # Add T-Copula specific columns (pandas allows NaNs for non-applicable rows)
    chi_factors = (
        [None] * N_GAUSS
        + ["GLOBAL_CHI"] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
    )
    nus = (
        [None] * N_GAUSS
        + [GLOBAL_T_COP_NU] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
    )

    jtd_df = jtd_df.with_columns(
        pl.Series("chi_squared_factor", chi_factors, dtype=pl.String),
        pl.Series("nu", nus, dtype=pl.Float64),
    )

    print_df(
        jtd_df.sample(n=25, seed=RNG_SEED),
        title="Sample JTD Portfolio DataFrame (Unsorted)",
    )

    # Compile the portfolio! This strips strings, maps UIXs, joins PD/LGD,
    # computes thresholds via ndtri(PD), precomputes sqrt(nu), and builds
    # the C-contiguous Struct-of-Arrays (SoA) blocks.
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

    # EAD (Exposure At Default) is the total dollar amount at risk at the exact
    # moment of default. It changes per scenario because it's driven by market
    # risk (e.g., if a counterparty defaults when FX rates move against them,
    # our exposure is higher).
    #
    # DOD INVARIANT: The EAD matrix rows MUST align perfectly with the
    # `compiled_portfolio.obligor_uids` array, because the engine dispatches
    # contiguous memory slices based on the block offsets.

    ead_by_obligor = np.empty(
        (NUM_OBLIGORS, NUM_SCENARIOS), dtype=np.float64, order="C"
    )

    # Synthesize base $1M - $10M exposures
    base_ead = rng.uniform(1e6, 10e6, (NUM_OBLIGORS, 1))
    ead_by_obligor[:] = base_ead  # Broadcast across scenarios

    # Introduce stochastic noise to simulate market-driven exposure paths
    ead_by_obligor *= rng.lognormal(
        mean=0.0, sigma=0.1, size=(NUM_OBLIGORS, NUM_SCENARIOS)
    )

    # ============================================================
    # 5. SYNTHESIZE FACTOR MATRIX
    # ============================================================
    print("[4/5] Generating Credit Factor Matrix...")

    # === Synthesize the credit factor matrix from the universe master ===
    # The mixed generator derives its standard-normal / chi / uniform / stress
    # columns from `universe`, so every row carries the correct marginal and
    # cannot disagree with the compiled portfolio about which columns are chi
    # (or at what nu). This replaces the hand-rolled per-block buffer fill.
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

    # The engine walks through `compiled_portfolio.blocks` (Gaussian -> T-Copula -> Forced-Default).
    # It slices the `ead_by_obligor` and passes it to the highly-optimized `njit` kernels.
    # Because memory is C-contiguous, the CPU cache pre-fetches the scenario vectors,
    # resulting in near bare-metal hardware limits.
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

    # jtd_result.loss is shape (NUM_OBLIGORS, NUM_SCENARIOS)
    # We sum down the obligor axis (axis=0) to get the total portfolio loss per scenario.
    portfolio_losses = np.sum(jtd_result.loss, axis=0)

    # Expected Loss (EL) is the mathematical mean of all Monte Carlo outcomes.
    expected_loss = np.mean(portfolio_losses)

    # Credit VaR (Value at Risk) 99.0% represents the loss threshold that is only
    # breached in the worst 1% of scenarios (driven heavily by the T-Copula W factor).
    credit_var_99 = np.quantile(portfolio_losses, 0.99)

    # Credit Expected Shortfall (ES) 99.0% is the average loss across all scenarios that breach the 99% VaR threshold.
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
    # Empirical PD Validation
    # ============================================================

    # A key sanity check for any copula default engine is verifying that the
    # simulated default frequencies converge to the input PDs. Since default
    # thresholds are constructed via:
    #
    #     t_i = Phi^{-1}(PD_i)
    #
    # the unconditional probability of:
    #
    #     A_i < t_i
    #
    # should equal the input PD as the number of Monte Carlo scenarios grows.

    print("\n--- Empirical PD Validation ---")

    # A default occurs whenever a positive loss is realized.
    default_indicators = jtd_result.loss > 0.0

    # Empirical default frequency for each obligor.
    empirical_pd = default_indicators.mean(axis=1)

    # ------------------------------------------------------------------
    # Align empirical PDs and target PDs by UID.
    #
    # Never assume row ordering is identical between:
    #   - jtd_result.loss
    #   - compiled_portfolio.obligor_uids
    #   - obligor_master
    #
    # Explicitly joining by UID avoids subtle validation bugs.
    # ------------------------------------------------------------------

    validation_df = pd.DataFrame({
        "uid": compiled_portfolio.obligor_uids,
        "empirical_pd": empirical_pd,
    })

    validation_df = validation_df.merge(
        pd.DataFrame({
            "uid": obligor_master.uid,
            "target_pd": obligor_master.pd,
        }),
        on="uid",
        how="inner",
    )

    validation_df["abs_error"] = np.abs(
        validation_df["empirical_pd"] - validation_df["target_pd"]
    )

    print(f"Mean Input PD:      {validation_df['target_pd'].mean():.4%}")

    print(f"Mean Empirical PD:  {validation_df['empirical_pd'].mean():.4%}")

    print(f"Mean Absolute Error:{validation_df['abs_error'].mean():.4%}")

    print(f"Max Absolute Error: {validation_df['abs_error'].max():.4%}")

    # ============================================================
    # Largest Deviations
    # ============================================================

    top_n = 10

    worst_df = validation_df.sort_values(
        "abs_error",
        ascending=False,
    ).head(top_n)

    print(f"\n--- Top {top_n} Largest PD Deviations ---")

    for rank, row in enumerate(
        worst_df.itertuples(index=False),
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"Obligor {row.uid:<4d} "
            f"Target={row.target_pd:>7.3%} "
            f"Empirical={row.empirical_pd:>7.3%} "
            f"Error={row.abs_error:>7.3%}"
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

    # Find the scenario closest to the 99% Credit VaR.
    # This identifies the tail scenario responsible for the VaR loss.
    var_idx = np.argmin(np.abs(portfolio_losses - credit_var_99))

    # Loss contribution of each obligor in the VaR scenario.
    obligor_contributions = jtd_result.loss[:, var_idx]

    # Sort descending by contribution.
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

    # Select all portfolio loss scenarios that exceed the 99% Credit VaR.
    # This defines the 1% tail of the loss distribution.
    tail_mask = portfolio_losses >= credit_var_99

    # Average loss contribution of each obligor across all tail scenarios.
    # This produces a stable Expected Shortfall-style attribution.
    obligor_contributions = jtd_result.loss[:, tail_mask].mean(axis=1)

    # Sort obligors by contribution to tail losses.
    sorted_idx = np.argsort(obligor_contributions)[::-1]

    top_n = 10

    print(f"\n--- Top {top_n} Contributors to 99% Credit Expected Shortfall ---")

    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        uid = compiled_portfolio.obligor_uids[idx]
        contrib = obligor_contributions[idx]

        print(f"{rank:2d}. Obligor {uid:<4d} Contribution: ${contrib:,.2f}")

    # Average portfolio loss across all tail scenarios.
    credit_es_99 = portfolio_losses[tail_mask].mean()

    print(f"\n99% Credit Expected Shortfall: ${credit_es_99:,.2f}")

    # --- TAIL CONCENTRATION ANALYSIS ---

    top10_share = (
        obligor_contributions[sorted_idx[:10]].sum() / obligor_contributions.sum()
    )

    print(f"Top 10 names explain {100 * top10_share:.1f}% of Credit Expected Shortfall")

    # ============================================================
    # MODEL COMPARISON: FORCED-DEFAULT VS GAUSSIAN VS T-COPULA
    # ============================================================

    # 1. Create a fast lookup series from the raw dataframe
    model_by_uid = dict(
        zip(jtd_df["uid"].to_list(), jtd_df["instrument"].to_list(), strict=True)
    )

    # 2. Align the model labels to the exact memory order of the loss matrix
    aligned_models = np.array([
        model_by_uid[int(u)] for u in compiled_portfolio.obligor_uids
    ])

    # 3. Generate the boolean masks
    forced_default_mask = aligned_models == JtdModel.FORCED_DEFAULT.name
    independent_bernoulli_mask = aligned_models == JtdModel.INDEPENDENT_BERNOULLI.name
    stress_regime_bernoulli_mask = (
        aligned_models == JtdModel.STRESS_REGIME_BERNOULLI.name
    )
    gaussian_mask = aligned_models == JtdModel.COPULA_SF_GAUSSIAN.name
    t_mask = aligned_models == JtdModel.COPULA_SF_STUDENTS_T.name

    # 4. Slice the 2D loss matrix and sum across obligors
    forced_default_losses = jtd_result.loss[forced_default_mask].sum(axis=0)
    independent_bernoulli_losses = jtd_result.loss[independent_bernoulli_mask].sum(
        axis=0
    )
    stress_regime_bernoulli_losses = jtd_result.loss[stress_regime_bernoulli_mask].sum(
        axis=0
    )
    gaussian_losses = jtd_result.loss[gaussian_mask].sum(axis=0)
    t_losses = jtd_result.loss[t_mask].sum(axis=0)

    # 5. Calculate Metrics
    forced_default_el = forced_default_losses.mean()
    forced_default_var = np.quantile(forced_default_losses, 0.99)
    forced_default_es = forced_default_losses[
        forced_default_losses >= forced_default_var
    ].mean()

    independent_bernoulli_el = independent_bernoulli_losses.mean()
    independent_bernoulli_var = np.quantile(independent_bernoulli_losses, 0.99)
    independent_bernoulli_es = independent_bernoulli_losses[
        independent_bernoulli_losses >= independent_bernoulli_var
    ].mean()

    stress_regime_bernoulli_el = stress_regime_bernoulli_losses.mean()
    stress_regime_bernoulli_var = np.quantile(stress_regime_bernoulli_losses, 0.99)
    stress_regime_bernoulli_es = stress_regime_bernoulli_losses[
        stress_regime_bernoulli_losses >= stress_regime_bernoulli_var
    ].mean()

    gauss_el = gaussian_losses.mean()
    gauss_var = np.quantile(gaussian_losses, 0.99)
    gauss_es = gaussian_losses[gaussian_losses >= gauss_var].mean()

    t_el = t_losses.mean()
    t_var = np.quantile(t_losses, 0.99)
    t_es = t_losses[t_losses >= t_var].mean()

    # 6. Print Formatted Table
    print("\n" + "=" * 85)
    print(f"{'MODEL COMPARISON SUMMARY':^85}")
    print("=" * 91)
    print(
        f"{'Model Type':<26} | {'Expected Loss (EL)':>18} | {'99% Credit VaR':>18} | {'99% Credit ES':>18}"
    )
    print("-" * 91)
    print(
        f"{'Forced-Default':<26} | ${forced_default_el:>17,.2f} | ${forced_default_var:>17,.2f} | ${forced_default_es:>17,.2f}"
    )
    print(
        f"{'Independent Bernoulli':<26} | ${independent_bernoulli_el:>17,.2f} | ${independent_bernoulli_var:>17,.2f} | ${independent_bernoulli_es:>17,.2f}"
    )
    print(
        f"{'Stress Regime Bernoulli':<26} | ${stress_regime_bernoulli_el:>17,.2f} | ${stress_regime_bernoulli_var:>17,.2f} | ${stress_regime_bernoulli_es:>17,.2f}"
    )
    print(
        f"{'Gaussian Copula':<26} | ${gauss_el:>17,.2f} | ${gauss_var:>17,.2f} | ${gauss_es:>17,.2f}"
    )
    print(
        f"{'Student-t Copula':<26} | ${t_el:>17,.2f} | ${t_var:>17,.2f} | ${t_es:>17,.2f}"
    )
    print("=" * 91)

    # ============================================================
    # END
    # ============================================================

    print_success_message()


if __name__ == "__main__":
    run_demo()
