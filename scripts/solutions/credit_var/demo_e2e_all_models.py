# scripts/solutions/credit_var/demo_e2e_all_models.py

import numpy as np
import pandas as pd

from riskquant.solutions.credit_var.api import (
    CreditVarRequest,
    model_entry,
    run_credit_var,
    simulate_portfolio_losses,
    supported_model_keys,
)
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant_lab.fixtures.credit_var import generate_credit_portfolio_data
from riskquant_lab.plotting.credit_var import (
    plot_ecdf_comparison,
    plot_loss_distribution_comparison,
    plot_var_es_comparison,
)

NUM_OBLIGORS: int = 500
RNG_SEED: int = 99
# Lower than the single-model demos (1M): this harness runs every model twice
# (authoritative metrics via run_credit_var, plotting vector via
# simulate_portfolio_losses), so 500k keeps the full sweep practical while still
# giving a stable 99.9% tail. Raise for a production comparison.
NUM_SCENARIOS: int = 1_000_000
NUM_SCENARIOS: int = 10_000
ALPHAS: list[float] = [0.9500, 0.9750, 0.9900, 0.9990]

# Concentration (alpha + beta) for the beta-binomial pools; the mean is calibrated
# to the book so the model sits on a comparable EL. Smaller -> more dispersion.
BETA_CONCENTRATION: float = 5.0

# Fixed scalar parameters. Beta-binomial (alpha/beta) is calibrated to the shared
# book at run time; grouped models take GROUP_PARAMS; MF models take loadings.
MODEL_PARAMS: dict[str, dict[str, float]] = {
    "forced_default": {},
    "independent_bernoulli": {},
    "beta_binomial": {},  # alpha/beta injected from the book's average PD
    "beta_binomial_grouped": {},
    "stress_regime_bernoulli": {"stress_prob": 0.20, "stress_pd_multiplier": 4.0},
    "stress_regime_bernoulli_grouped": {},
    "sf_gaussian": {"rho": 0.15},
    "sf_gaussian_basel": {},
    "sf_students_t": {"rho": 0.15, "nu": 7.0},
    "sf_students_t_basel": {"nu": 7.0},
    "mf_gaussian": {},
    "mf_students_t": {"nu": 7.0},
}

# Per-group stress parameters (grouped by credit quality; see _assign_groups).
STRESS_GROUP_PARAMS: dict[str, dict[str, float]] = {
    "IG": {"stress_prob": 0.05, "stress_pd_multiplier": 2.0},
    "HY": {"stress_prob": 0.40, "stress_pd_multiplier": 5.0},
}

# Deterministic point mass -> excluded from the distribution overlays (kept in the table).
OVERLAY_EXCLUDE: set[str] = {"forced_default"}

_IG_RATINGS = ["AAA", "AA", "A", "BBB"]


def _alpha_key(alpha: float) -> str:
    return f"{alpha * 100:.2f}%"


def _assign_groups(obligor_df: pd.DataFrame) -> np.ndarray:
    """Split the book into IG / HY pools by rating (used by all grouped models)."""
    return np.where(obligor_df["rating"].isin(_IG_RATINGS), "IG", "HY")


def _beta_params(mean_pd: float, concentration: float) -> dict[str, float]:
    """Beta(alpha, beta) with the given mean and concentration alpha + beta."""
    return {"alpha": mean_pd * concentration, "beta": (1.0 - mean_pd) * concentration}


def _weighted_avg_pd(frame: pd.DataFrame) -> float:
    weight = frame["ead"] * frame["lgd"]
    return float((weight * frame["pd"]).sum() / weight.sum())


def _build_loadings(obligor_df: pd.DataFrame) -> pd.DataFrame:
    """Common MARKET factor + each obligor's sector factor (genuine multi-loading)."""
    return pd.DataFrame(
        [
            {"obligor_uid": int(u), "factor": "MARKET", "loading": 0.30}
            for u in obligor_df["obligor_uid"]
        ]
        + [
            {"obligor_uid": int(u), "factor": s, "loading": 0.50}
            for u, s in zip(
                obligor_df["obligor_uid"], obligor_df["sector"], strict=True
            )
        ]
    )


def _build_request(
    model_key: str,
    obligor_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
) -> CreditVarRequest:
    """Construct a request for one model on the shared book, honouring its flags."""
    entry = model_entry(model_key)
    frame = obligor_df
    params = dict(MODEL_PARAMS[model_key])
    group_params: dict[str, dict[str, float]] | None = None
    loadings_table: pd.DataFrame | None = None

    if entry.requires_groups:
        frame = obligor_df.copy()
        frame["group"] = _assign_groups(frame)
        if model_key == "beta_binomial_grouped":
            group_params = {
                grp: _beta_params(
                    _weighted_avg_pd(frame[frame["group"] == grp]), BETA_CONCENTRATION
                )
                for grp in ("IG", "HY")
            }
        else:  # stress_regime_bernoulli_grouped
            group_params = STRESS_GROUP_PARAMS

    if entry.requires_loadings:
        loadings_table = loadings_df

    if model_key == "beta_binomial":
        # Calibrate the single pool to the book so its EL is comparable.
        params = _beta_params(_weighted_avg_pd(obligor_df), BETA_CONCENTRATION)

    return CreditVarRequest.from_dataframe(
        obligor_table=frame,
        model=model_key,
        loadings_table=loadings_table,
        group_params=group_params,
        n_scenarios=NUM_SCENARIOS,
        alphas=ALPHAS,
        seed=RNG_SEED,
        **params,
    )


def _print_comparison_table(results: dict) -> None:
    k99 = _alpha_key(0.99)
    k999 = _alpha_key(0.999)
    width = 122
    print("-" * width)
    print("Credit VaR — Model Comparison (identical obligor portfolio)")
    print("-" * width)
    print(
        f"{'Model':32} {'EL':>13} {'Loss Std. Dev.':>13} {'Skew':>7} {'ExKurt':>8} "
        f"{'VaR 99%':>14} {'ES 99%':>14} {'VaR 99.9%':>14}"
    )
    print("-" * width)
    for key, res in results.items():
        s = res.summary
        print(
            f"{key:32} {s.loss_mean:>13,.0f} {s.loss_std_dev:>13,.0f} "
            f"{s.loss_skewness:>7.2f} {s.loss_excess_kurtosis:>8.2f} "
            f"{s.var.get(k99, float('nan')):>14,.0f} {s.es.get(k99, float('nan')):>14,.0f} "
            f"{s.var.get(k999, float('nan')):>14,.0f}"
        )
    print("-" * width)


def main():
    """
    Run every credit-VaR model on ONE shared obligor portfolio and compare.

    Produces a metrics table (EL, UL, higher moments, VaR/ES at each level) and
    three overlay plots: loss-distribution KDEs, empirical CDFs, and a VaR/ES bar
    chart — so the models can be compared like for like on identical inputs.
    """
    print_calling_file()

    # 1. One shared obligor portfolio (identical EAD / PD / LGD for every model)
    # ----------------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    obligor_df = pd.DataFrame(
        generate_credit_portfolio_data(n_obligors=NUM_OBLIGORS, rng=rng)
    )
    loadings_df = _build_loadings(obligor_df)

    # 2. Run every model on that portfolio
    # ----------------------------------------------------------------------------------
    results: dict = {}
    losses: dict[str, np.ndarray] = {}
    for model_key in supported_model_keys():
        req = _build_request(model_key, obligor_df, loadings_df)
        results[model_key] = run_credit_var(req)
        losses[model_key] = simulate_portfolio_losses(req)
        print(
            f"  -> Ran: {model_key:34} (EL={results[model_key].summary.loss_mean:>14,.0f})"
        )

    # 3. Comparison table
    # ----------------------------------------------------------------------------------
    print()
    _print_comparison_table(results)

    # 4. Visual comparisons (exclude the deterministic point mass from the overlays)
    # ----------------------------------------------------------------------------------
    overlay = {k: v for k, v in losses.items() if k not in OVERLAY_EXCLUDE}

    plot_loss_distribution_comparison(
        overlay,
        title="Loss Distribution (KDE) — All Models, Shared Portfolio",
        log_y=True,
    )
    plot_ecdf_comparison(
        overlay,
        title="Empirical CDF — All Models, Shared Portfolio",
    )

    headline = _alpha_key(max(ALPHAS))
    labels = list(results)
    var_values = [results[k].summary.var.get(headline, float("nan")) for k in labels]
    es_values = [results[k].summary.es.get(headline, float("nan")) for k in labels]
    plot_var_es_comparison(
        labels,
        var_values,
        es_values,
        alpha_label=headline,
        title=f"VaR & ES by Model at {headline} — Shared Portfolio",
    )

    print_success_message()


if __name__ == "__main__":
    main()
