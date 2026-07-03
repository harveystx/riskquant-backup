# scripts/solutions/credit_var/demo_e2e_sf_gaussian_basel.py

import numpy as np
import pandas as pd

from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var
from riskquant.solutions.credit_var.run import simulate_portfolio_losses
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant_lab.fixtures.credit_var import (
    generate_credit_portfolio_data,
)
from riskquant_lab.plotting.credit_var import plot_credit_loss_distribution

NUM_OBLIGORS: int = 100
RNG_SEED: int = 99
NUM_SCENARIOS: int = 1_000_000
MODEL_KEY: str = "sf_gaussian_basel"
ALPHAS: list[float] = [0.9500, 0.9750, 0.9900, 0.9990]
MODEL_PARAMS: dict[str, float] = {
    # Basel IRB corporate R(PD): rho is derived per-obligor from PD.
    # No rho parameter is supplied.
}
GROUP_PARAMS: dict[str, dict[str, float]] = {}
KDE_DISPLAY_PERCENTILE: float = 99.90


def main():
    """
    End-to-end demonstration of the `sf_gaussian_basel` model.

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

    # 2. Construct the request
    # ----------------------------------------------------------------------------------
    req = CreditVarRequest.from_dataframe(
        obligor_table=obligor_df,
        model=MODEL_KEY,
        loadings_table=None,
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

    for alpha, var in result.summary.var.items():
        es = result.summary.es[alpha]
        print(f"VaR ({alpha}):  {var:>18,.0f} | ES: {es:>18,.0f}")
    print("-" * 81)

    # 5. Simulate losses for the request
    # ----------------------------------------------------------------------------------
    portfolio_losses = simulate_portfolio_losses(req)

    # 6. Plot loss distribution with KDE overlay
    # ----------------------------------------------------------------------------------
    plot_credit_loss_distribution(
        hist_bin_edges=result.loss_histogram.bin_edges,
        hist_counts=result.loss_histogram.counts,
        expected_loss=result.summary.loss_mean,
        var_levels=result.summary.var,
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
