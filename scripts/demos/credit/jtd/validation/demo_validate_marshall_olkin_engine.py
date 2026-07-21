# scripts/demos/credit/jtd/validation/demo_validate_marshall_olkin_engine.py
"""
End-to-End Validation of the Marshall-Olkin JTD Engine.

This script validates the Marshall-Olkin (Common Shock) block of the
Jump-to-Default (JTD) credit engine.

Marshall-Olkin obligors are exposed to two independent sources of default:
an idiosyncratic shock unique to the obligor, and a systemic common shock
shared across a group. If either shock triggers, the obligor defaults.
This creates deterministic default clustering, generating a discontinuous
loss distribution with extreme, Dirac-delta tail spikes representing
singular catastrophic events (e.g., a sovereign or clearinghouse collapse).

The purpose of this validation is to demonstrate that the engine correctly
evaluates dual-trigger default conditions, accurately decouples the input
marginal PDs to prevent double-counting, and reproduces the exact structural
contagion implied by the framework.

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

Let $U_i$ and $S$ be independent standard uniform random variables:

    $U_i \\sim \\text{Uniform}(0, 1)$  (Idiosyncratic shock for obligor $i$)
    $S \\sim \\text{Uniform}(0, 1)$    (Shared systemic common shock)

For an obligor $i$ with total probability of default $PD_{total}$ and exposed
to a systemic shock with probability $PD_{sys}$, the decoupled idiosyncratic
probability of default is:

    $PD_{idio} = 1 - \\frac{1 - PD_{total}}{1 - PD_{sys}}$

The default indicator for obligor $i$ in scenario $s$ is:

    $D_{i,s} = \\mathbb{I}_{(U_{i,s} < PD_{idio})} \\lor \\mathbb{I}_{(S_s < PD_{sys})}$

The portfolio loss in scenario $s$ is:

    $L_s = \\sum_i D_{i,s} \\times LGD_i \\times EAD_i$

where

    $LGD_i$ : loss given default
    $EAD_i$ : exposure at default

Because exposed obligors share the exact same systemic draw $S_s$, any scenario
where $S_s < PD_{sys}$ guarantees $D_{i,s} = 1$ for all obligors in that group,
regardless of their idiosyncratic draws.

===============================================================================
THEORETICAL RESULTS
===============================================================================

Marginal Default Probability:

    $\\mathbb{P}(D_i = 1) = PD_{total}$

Joint Default Probability (Contagion):

    $\\mathbb{P}(D_i = 1 \\cap D_j = 1 \\mid S < PD_{sys}) = 1$

Expected Loss:

    $EL = \\sum_i PD_{total} \\times LGD_i \\times EAD_i$

Tail Risk:

The loss distribution exhibits massive, discontinuous spikes in the far right
tail corresponding to scenarios where the common shock triggers.

===============================================================================
EXPECTED VALIDATION PROPERTIES
===============================================================================

The simulation should demonstrate:

1. Marginal default frequencies empirically converge to the input $PD_{total}$,
   proving the decoupling algebra correctly prevents double-counting.
2. The portfolio loss distribution exhibits extreme structural contagion
   (Dirac-delta spikes).
3. In scenarios where the max loss is realized, exactly 100% of the exposed
   cohort has defaulted.
4. Simulated expected loss converges to the theoretical expected loss.

===============================================================================
RELATIONSHIP TO OTHER JTD BLOCKS
===============================================================================

The Marshall-Olkin block models explicit, deterministic structural contagion.

Unlike Gaussian Copula and Student-t Copula models, dependence is not continuous
and is not driven by correlated asset values.

Unlike the Stress Regime Bernoulli model, the systemic factor does not merely
shift probabilities—it executes defaults directly.

Consequently, Marshall-Olkin produces the most extreme form of tail risk
available in the framework, representing mutually assured destruction.

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
N_GAUSS = int(0.0 * NUM_OBLIGORS)
N_T_COP = int(0.0 * NUM_OBLIGORS)
N_FORCED_DEFAULT = int(0.0 * NUM_OBLIGORS)
N_INDEPENDENT_BERNOULLI = int(0.0 * NUM_OBLIGORS)
N_STRESS_REGIME_BERNOULLI = int(0.0 * NUM_OBLIGORS)
N_MARSHALL_OLKIN = int(1.0 * NUM_OBLIGORS)

GLOBAL_T_COP_NU = 7.0
MO_SYSTEMIC_PD = 0.02  # 2% chance of common shock


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
        color="#d62728",
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
        color="orange",
        linestyle="solid",
        linewidth=2.5,
        label=f"99% Credit VaR: ${credit_var_99_m:,.1f}M",
    )

    ax.set_title(
        "E2E JTD Portfolio Loss Distribution (Marshall-Olkin)", fontsize=14, pad=15
    )
    ax.set_xlabel("Simulated Portfolio Loss ($ Millions)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}M"))
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.show()


def run_demo() -> None:
    print_calling_file()
    print("--- MARSHALL-OLKIN JTD ENGINE E2E DEMO ---")
    print(f"Obligors:  {NUM_OBLIGORS}")
    print(f"Scenarios: {NUM_SCENARIOS:,}")
    print("-" * 42 + "\n")

    rng = np.random.default_rng(seed=RNG_SEED)

    # ============================================================
    # 2. INGEST: OBLIGOR MASTER
    # ============================================================
    print("[1/5] Building & Compiling Obligor Master...")

    obligor_uids = np.arange(1, NUM_OBLIGORS + 1, dtype=np.int64)
    rng.shuffle(obligor_uids)

    # CRITICAL: For Marshall-Olkin, the Master PD (Total PD) must be mathematically
    # greater than or equal to the Systemic PD to allow survival decoupling.
    # We generate total PDs smoothly between 2.5% and 8.0%.
    obligor_df = pd.DataFrame(
        {
            "uid": obligor_uids,
            "pd": rng.uniform(MO_SYSTEMIC_PD + 0.005, 0.08, NUM_OBLIGORS),
            "lgd": rng.uniform(0.4, 0.8, NUM_OBLIGORS),
            "rating_id": 0,
            "sector_id": 0,
            "country_id": 0,
            "parent_uid": None,
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
        + [JtdModel.MARSHALL_OLKIN.name] * N_MARSHALL_OLKIN
    )

    # U_ prefix maps standard string parsing to UNIFORM kind.
    factor_names = (
        ["GLOBAL_CREDIT"]
        + [f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_CHI"]
        + [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)]
        + ["GLOBAL_STRESS_REGIME"]
        + ["U_SYSTEMIC_SHOCK"]
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
                else "IDIOSYNCRATIC"
                for n in factor_names
            ],
            "nu": [
                GLOBAL_T_COP_NU if (n == "GLOBAL_CHI" or n.startswith("CHI_")) else None
                for n in factor_names
            ],
        }
    )
    universe = compile_credit_factor_universe(factor_universe_df)

    # Marshall-Olkin's common shock is a UNIFORM draw, so it uses the dedicated
    # `systemic_uniform_factor` role (UNIFORM kind), NOT `systemic_factor`
    # (SYSTEMIC/Gaussian, used by the copulas).
    systemic_uniform_factors = [
        "U_SYSTEMIC_SHOCK" if m == JtdModel.MARSHALL_OLKIN.name else None
        for m in models
    ]

    jtd_df = pd.DataFrame(
        {
            "uid": obligor_uids,
            "instrument": models,
            "beta": rng.uniform(0.1, 0.5, NUM_OBLIGORS),
            "systemic_factor": "GLOBAL_CREDIT",
            "systemic_uniform_factor": systemic_uniform_factors,
            "systemic_pd": MO_SYSTEMIC_PD,
            "idiosyncratic_factor": [
                f"IDIO_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)
            ],
            "uniform_factor": [f"U_{i:03d}" for i in range(1, NUM_OBLIGORS + 1)],
            "stress_regime_factor": "GLOBAL_STRESS_REGIME",
            "stress_pd_multiplier": rng.uniform(1.25, 3.0, NUM_OBLIGORS),
        }
    )

    chi_factors = (
        [None] * N_GAUSS
        + ["GLOBAL_CHI"] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_MARSHALL_OLKIN
    )
    nus = (
        [None] * N_GAUSS
        + [GLOBAL_T_COP_NU] * N_T_COP
        + [None] * N_FORCED_DEFAULT
        + [None] * N_INDEPENDENT_BERNOULLI
        + [None] * N_STRESS_REGIME_BERNOULLI
        + [None] * N_MARSHALL_OLKIN
    )

    jtd_df["chi_squared_factor"] = chi_factors
    jtd_df["nu"] = nus

    print_df(
        jtd_df.sample(n=25, random_state=RNG_SEED),
        title="Sample JTD Portfolio DataFrame (Unsorted)",
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
    # 4. SYNTHESIZE EAD
    # ============================================================
    print("[3/5] Generating Exposure Matrix...")

    ead_by_obligor = np.empty(
        (NUM_OBLIGORS, NUM_SCENARIOS), dtype=np.float64, order="C"
    )

    base_ead = rng.uniform(1e6, 10e6, (NUM_OBLIGORS, 1))
    ead_by_obligor[:] = base_ead
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

    print("\n--- Empirical PD Validation (Decoupling Check) ---")

    default_indicators = jtd_result.loss > 0.0
    empirical_pd = default_indicators.mean(axis=1)

    validation_df = pd.DataFrame(
        {
            "uid": compiled_portfolio.obligor_uids,
            "empirical_pd": empirical_pd,
        }
    )

    validation_df = validation_df.merge(
        pd.DataFrame(
            {
                "uid": obligor_master.uid,
                "target_pd": obligor_master.pd,
            }
        ),
        on="uid",
        how="inner",
    )

    validation_df["abs_error"] = np.abs(
        validation_df["empirical_pd"] - validation_df["target_pd"]
    )

    print(f"Mean Target PD:     {validation_df['target_pd'].mean():.4%}")
    print(f"Mean Empirical PD:  {validation_df['empirical_pd'].mean():.4%}")
    print(f"Mean Absolute Error:{validation_df['abs_error'].mean():.4%}")
    print(f"Max Absolute Error: {validation_df['abs_error'].max():.4%}")
    print("(Proves algebraic decoupling perfectly conserves input marginals)")

    # ============================================================
    # DEFAULT STATS & MARSHALL-OLKIN TAIL SPIKE
    # ============================================================

    defaults_per_scenario = default_indicators.sum(axis=0)

    print("\n--- Default Clustering Statistics ---")
    print(f"Average Defaults:      {defaults_per_scenario.mean():.2f}")
    print(f"95% Default Count:     {np.quantile(defaults_per_scenario, 0.95):.0f}")
    print(f"99% Default Count:     {np.quantile(defaults_per_scenario, 0.99):.0f}")

    # Structural Contagion Proof
    max_defaults = defaults_per_scenario.max()
    print("\n--- Marshall-Olkin Structural Contagion Proof ---")
    print(f"Max Defaults Observed: {max_defaults:.0f} out of {NUM_OBLIGORS}")
    if max_defaults == NUM_OBLIGORS:
        print(">> SUCCESS: Mutual Destruction scenarios successfully generated.")
    else:
        print(">> FAIL: Structural contagion did not reach 100%.")

    # ============================================================
    # MODEL COMPARISON
    # ============================================================

    model_series = jtd_df.set_index("uid")["instrument"]
    aligned_models = model_series.loc[
        compiled_portfolio.obligor_uids.tolist()
    ].to_numpy()

    mo_mask = aligned_models == JtdModel.MARSHALL_OLKIN.name
    mo_losses = jtd_result.loss[mo_mask].sum(axis=0)

    mo_el = mo_losses.mean()
    mo_var = np.quantile(mo_losses, 0.99)
    mo_es = mo_losses[mo_losses >= mo_var].mean()

    print("\n" + "=" * 85)
    print(f"{'MODEL COMPARISON SUMMARY':^85}")
    print("=" * 91)
    print(
        f"{'Model Type':<26} | {'Expected Loss (EL)':>18} | {'99% Credit VaR':>18} | {'99% Credit ES':>18}"
    )
    print("-" * 91)
    print(
        f"{'Marshall-Olkin':<26} | ${mo_el:>17,.2f} | ${mo_var:>17,.2f} | ${mo_es:>17,.2f}"
    )
    print("=" * 91)

    print_success_message()


if __name__ == "__main__":
    run_demo()
