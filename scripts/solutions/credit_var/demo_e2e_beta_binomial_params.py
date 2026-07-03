# scripts/solutions/credit_var/demo_e2e_beta_binomial_params.py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var
from riskquant.solutions.credit_var.run import simulate_portfolio_losses
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant_lab.fixtures.credit_var import (
    generate_credit_portfolio_data,
)
from riskquant_lab.plotting.credit_var import plot_loss_distribution_comparison

NUM_OBLIGORS: int = 500
RNG_SEED: int = 99
NUM_SCENARIOS: int = 1_000_000
MODEL_KEY: str = "beta_binomial"
ALPHAS: list[float] = [0.9500, 0.9750, 0.9900, 0.9990]

# ------------------------ #
# STRESS TESTING SETUP     #
# ------------------------ #

# Precision represents the concentration of the distribution (alpha + beta).
# Keeping this constant isolates the effect of shifting the mean PD.
# Definition: alpha=1.0, beta=9.0 -> Precision = 10.0
PRECISION_STRESS_VALUES: list[float] = [
    5.0,  # More dispersed (fatter tails)
    10.0,  # Base case
    20.0,  # Less dispersed (thinner tails)
]

# Stressing the target Mean PD directly.
MEAN_PD_STRESS_VALUES: list[float] = [
    0.010,  # 1.0%
    0.025,  # 2.5%
    0.050,  # 5.0%
    0.100,  # 10.0%
    0.250,  # 25.0%
    0.500,  # 50.0%
    0.750,  # 75.0%
]

# Other
GROUP_PARAMS: dict[str, dict[str, float]] = {}
KDE_DISPLAY_PERCENTILE: float = 99.90


def _calc_alpha_beta(mean_pd: float, precision: float) -> tuple[float, float]:
    """
    Calculate alpha and beta for a beta distribution given a target mean and precision.

    mean_pd = alpha / (alpha + beta)
    precision = alpha + beta
    """
    if not (0 < mean_pd < 1):
        raise ValueError("mean_pd must be strictly between 0 and 1.")
    if precision <= 0:
        raise ValueError("precision must be greater than 0.")

    alpha = mean_pd * precision
    beta = precision * (1.0 - mean_pd)
    return alpha, beta


def main():

    print_calling_file(add_space_after=False)

    clean_model_key_str = MODEL_KEY.upper().replace("_", " ")

    # 1. Generate random obligor data (Generated ONCE for consistency)
    # ----------------------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)

    obligor_df = pd.DataFrame(
        generate_credit_portfolio_data(
            n_obligors=NUM_OBLIGORS,
            rng=rng,
        )
    )

    # 2. Iterate over the stress values
    # ---------------------------------------------------------------------------------

    for precision in PRECISION_STRESS_VALUES:
        # Preallocate to store results and losses for each stress scenario
        results: list = []
        losses_dict: dict[str, np.ndarray] = {}
        model_labels: list[str] = []

        print(
            f"\n=== Running stress tests for {clean_model_key_str} model with constant precision (alpha + beta) = {precision} ===\n"
        )

        for target_mean_pd in MEAN_PD_STRESS_VALUES:
            # Calculate the required alpha and beta for this scenario
            current_alpha, current_beta = _calc_alpha_beta(
                target_mean_pd,
                precision,
            )

            current_model_params = {"alpha": current_alpha, "beta": current_beta}

            current_model_label = (
                # f"{clean_model_key_str} | "
                f"alpha={current_alpha:7.4f}, beta={current_beta:7.4f} | "
                f"Mean PD={target_mean_pd * 100:5.2f}%"
            )

            # Construct the request for this specific scenario
            req = CreditVarRequest.from_dataframe(
                obligor_table=obligor_df,
                model=MODEL_KEY,
                loadings_table=None,
                group_params=GROUP_PARAMS,
                n_scenarios=NUM_SCENARIOS,
                alphas=ALPHAS,
                seed=RNG_SEED,
                **current_model_params,
            )

            current_result = run_credit_var(req)
            current_losses = simulate_portfolio_losses(req)

            current_el = current_result.summary.loss_mean
            current_var99 = current_result.summary.get_var(alpha=0.99)
            current_es99 = current_result.summary.get_es(alpha=0.99)
            current_es_to_var_ratio = (
                current_es99 / current_var99 if current_var99 != 0 else float("nan")
            )

            current_model_label += (
                f" | "
                f"EL={current_el:12,.0f}, "
                f"VaR 99%={current_var99:12,.0f}, "
                f"ES 99%={current_es99:12,.0f}, "
                f"ES/VaR Ratio={current_es_to_var_ratio:3.2f}"
            )

            results.append(current_result)
            model_labels.append(current_model_label)

            losses_dict[current_model_label] = current_losses

            print(f"--> Running model: {current_model_label}")

        fig, ax = plot_loss_distribution_comparison(
            losses_dict,
            title=f"Credit Loss KDE Distributions (Model: {clean_model_key_str}, Constant Precision (alpha + beta): {precision})",
            log_y=True,
            linthresh_y=1e-8,
            display_percentile=KDE_DISPLAY_PERCENTILE,
        )

        # Remove y-ticks and labels as requested earlier to clean up the symlog scaling aesthetics
        ax.set_yticks([])

        plt.show()

    print_success_message()


if __name__ == "__main__":
    main()
