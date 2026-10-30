# scripts/solutions/credit_var/demo_e2e_stress_regime_bernoulli_params.py

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
MODEL_KEY: str = "stress_regime_bernoulli"
ALPHAS: list[float] = [0.9500, 0.9750, 0.9900, 0.9990]

# ---------------------------------- #
# STRESS TESTING SETUP               #
# ---------------------------------- #

# Lock the probability of entering the stress regime at 5%
CONSTANT_STRESS_PROB: float = 0.05

# Sweep the multiplier to see how worsening crisis conditions impact the tail
MULTIPLIER_STRESS_VALUES: list[float] = [
    1.0,  # Baseline (No stress impact)
    1.2,  # Very mild stress
    1.5,  # Mild stress
    2.5,  # Moderate stress
    3.5,  # High stress
    5.0,  # Severe stress
    7.5,  # Extreme stress
    10.0,  # Catastrophic stress (everything goes to 100% PD effectively)
]

GROUP_PARAMS: dict[str, dict[str, float]] = {}
KDE_DISPLAY_PERCENTILE: float = 99.90


def main():

    print_calling_file()

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

    results: list = []
    losses_dict: dict[str, np.ndarray] = {}
    model_labels: list[str] = []

    # 2. Iterate over the stress values
    # ---------------------------------------------------------------------------------

    for target_multiplier in MULTIPLIER_STRESS_VALUES:
        current_model_params = {
            "stress_prob": CONSTANT_STRESS_PROB,
            "stress_pd_multiplier": target_multiplier,
        }

        current_model_label = (
            # f"{clean_model_key_str} | "
            f"P(Stress)={CONSTANT_STRESS_PROB * 100:7.4f}% | "
            f"PD Multiplier={target_multiplier:>5.2f}x"
        )

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
        current_at99 = current_result.summary.at(0.99)
        current_var99 = current_at99.var
        current_es99 = current_at99.es
        # The result has no ES/VaR ratio (O-5). The ratio is undefined at
        # VaR = 0; print "n/a" there, never a NaN.
        if current_var99 == 0.0:
            current_es_to_var_ratio = "n/a"
        else:
            current_es_to_var_ratio = f"{current_es99 / current_var99:3.2f}"

        current_model_label += (
            f" | "
            f"EL={current_el:12,.0f}, "
            f"VaR 99%={current_var99:12,.0f}, "
            f"ES 99%={current_es99:12,.0f}, "
            f"ES/VaR Ratio={current_es_to_var_ratio}"
        )

        results.append(current_result)
        model_labels.append(current_model_label)

        losses_dict[current_model_label] = current_losses

        print(f"--> Running model: {current_model_label}")

    # 3. Plot the comparative distributions
    # ---------------------------------------------------------------------------------
    fig, ax = plot_loss_distribution_comparison(
        losses_dict,
        title=f"Credit Loss KDE Distributions (Constant Stress Prob: {CONSTANT_STRESS_PROB * 100:.1f}%)",
        log_y=True,
        linthresh_y=1e-8,
        display_percentile=KDE_DISPLAY_PERCENTILE,
    )

    ax.set_yticks([])
    plt.show()

    print_success_message()


if __name__ == "__main__":
    main()
