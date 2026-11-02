# scripts/solutions/credit_var/demo_e2e_mf_gaussian.py

import numpy as np
import pandas as pd

from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var
from riskquant.solutions.credit_var.run import simulate_portfolio_losses
from riskquant.utils.printer import print_calling_file, print_df, print_success_message
from riskquant_lab.fixtures.credit_var import (
    generate_credit_portfolio_data,
)
from riskquant_lab.plotting.credit_var import plot_credit_loss_distribution

NUM_OBLIGORS: int = 100
RNG_SEED: int = 99
NUM_SCENARIOS: int = 1_000_000
MODEL_KEY: str = "mf_gaussian"
ALPHAS: list[float] = [0.9500, 0.9750, 0.9900, 0.9990]
MODEL_PARAMS: dict[str, float] = {}
GROUP_PARAMS: dict[str, dict[str, float]] = {}
KDE_DISPLAY_PERCENTILE: float = 99.90


def main() -> None:
    """
    End-to-end demonstration of the `mf_gaussian` model.

    This script generates a synthetic portfolio of obligors, constructs a
    `CreditVarRequest`, runs the credit VaR engine, and prints a summary of
    the results.

    The purpose is to illustrate the full workflow in a self-contained manner.
    """

    print_calling_file()

    # 1. Generate random obligor data
    # ----------------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)

    obligor_df = pd.DataFrame(
        generate_credit_portfolio_data(
            n_obligors=NUM_OBLIGORS,
            rng=rng,
        )
    )

    # 1b. Build the factor loadings: a common MARKET factor plus each obligor's
    # sector factor (this is the genuine multi-factor case -- names load on more
    # than one systematic factor).
    # ----------------------------------------------------------------------------------
    loadings_df = pd.DataFrame(
        [
            {
                "obligor_uid": int(uid),
                "factor": "MARKET",
                "loading": rng.integers(low=10, high=20) / 100,
            }
            for uid in obligor_df["obligor_uid"]
        ]
        + [
            {
                "obligor_uid": int(uid),
                "factor": sector,
                "loading": rng.integers(low=25, high=75) / 100,
            }
            for uid, sector in zip(
                obligor_df["obligor_uid"], obligor_df["sector"], strict=True
            )
        ]
    )

    grouped_loadings_df = loadings_df.groupby(["factor"]).agg(
        num_obligors=pd.NamedAgg(column="obligor_uid", aggfunc="count"),
        loading_mean=pd.NamedAgg(column="loading", aggfunc="mean"),
        loading_std_dev=pd.NamedAgg(column="loading", aggfunc="std"),
        loading_min=pd.NamedAgg(column="loading", aggfunc="min"),
        loading_max=pd.NamedAgg(column="loading", aggfunc="max"),
    )

    print_df(
        grouped_loadings_df,
        title="Factor Loadings Table",
        float_format="_.2f",
        index=True,
        add_space_before=False,
    )

    # 2. Construct the request
    # ----------------------------------------------------------------------------------
    req = CreditVarRequest.from_dataframe(
        obligor_table=obligor_df,
        model=MODEL_KEY,
        loadings_table=loadings_df,
        group_params=GROUP_PARAMS,
        n_scenarios=NUM_SCENARIOS,
        alphas=ALPHAS,
        seed=RNG_SEED,
        **MODEL_PARAMS,
    )

    # 3. Run the engine
    # ----------------------------------------------------------------------------------
    result = run_credit_var(req)

    # 4. Display the results
    # ----------------------------------------------------------------------------------
    print("-" * 81)
    print("Credit VaR Run Results")
    print("-" * 81)
    print(f"Model:                  {result.model}")
    print(f"Obligors:               {result.provenance.n_obligors:,}")
    print(f"Scenarios:              {result.provenance.n_scenarios:,}")
    print(f"Expected Loss:          {result.summary.loss_mean:,.2f}")
    print(f"Loss Std Dev:           {result.summary.loss_std_dev:,.2f}")
    print(f"Loss Skewness:          {result.summary.loss_skewness:,.2f}")
    print(f"Loss Excess Kurtosis:   {result.summary.loss_excess_kurtosis:,.2f}")
    print(f"Loss Probability:       {result.summary.loss_probability * 100:.2f}%")
    print(f"Loss Min:               {result.summary.loss_min:,.2f}")
    print(f"Loss Max:               {result.summary.loss_max:,.2f}")
    print("-" * 81)

    for record in result.summary.by_alpha:
        print(
            f"VaR ({record.alpha:.2%}):  {record.var:>18,.0f} | ES: {record.es:>18,.0f}"
        )
    print("-" * 81)

    # 5. Simulate losses for the request
    # ----------------------------------------------------------------------------------
    portfolio_losses = simulate_portfolio_losses(req)

    # 6. Plot loss distribution with KDE overlay
    # ----------------------------------------------------------------------------------
    # The plot takes dict[str, float] keyed by a display label. The label is for
    # display only; the result carries each level as its alpha float (O-1).
    # Two levels with one label would drop a level from the plot: raise.
    var_levels = {f"{r.alpha:.2%}": r.var for r in result.summary.by_alpha}
    if len(var_levels) != len(result.summary.by_alpha):
        raise ValueError(f"confidence levels share a display label: {ALPHAS}")
    plot_credit_loss_distribution(
        hist_bin_edges=list(result.loss_histogram.bin_edges),
        hist_counts=list(result.loss_histogram.counts),
        expected_loss=result.summary.loss_mean,
        var_levels=var_levels,
        losses=portfolio_losses,
        model_name=req.model,
        n_obligors=req.n_obligors,
        n_scenarios=req.n_scenarios,
        kde_display_percentile=KDE_DISPLAY_PERCENTILE,
    )

    # 7. Validate expected loss
    # ----------------------------------------------------------------------------------
    # Copulas are marginal-preserving: each obligor still defaults with probability
    # PD_i, so EL = sum(EAD * LGD * PD) exactly -- independent of the correlation.
    # (Std dev is correlation-inflated and has no simple closed form, so only the
    # mean is checked here.)
    mean_analytical = np.sum(obligor_df["ead"] * obligor_df["lgd"] * obligor_df["pd"])
    el_engine = result.summary.loss_mean
    abs_error = abs(mean_analytical - el_engine)
    rel_error = (abs_error / mean_analytical) if mean_analytical != 0 else 0.0

    print("\n" + "-" * 81)
    print("Validation of Analytical vs Engine Metrics")
    print("-" * 81)
    print(f"Loss Mean (Analytical):    {mean_analytical:,.0f}")
    print(f"Loss Mean (Engine):        {el_engine:,.0f}")
    print(f"Loss Mean Abs Error:       {abs_error:,.0f}")
    print(f"Loss Mean Rel Error:       {rel_error:.4%}")
    print("-" * 81)

    print_success_message()


if __name__ == "__main__":
    main()
