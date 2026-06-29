# scripts/solutions/credit_var/demo_e2e_independent_bernoulli.py

import numpy as np
import pandas as pd

from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var
from riskquant.utils.printer import print_calling_file, print_success_message

NUM_OBLIGORS = 10
RNG_SEED = 42
NUM_SCENARIOS = 500


def main():
    """
    End-to-end demonstration of the `independent_bernoulli` model.

    This script generates a synthetic portfolio of obligors, constructs a
    `CreditVarRequest`, runs the credit VaR engine, and prints a summary of
    the results.

    The purpose is to illustrate the full workflow in a self-contained manner.
    """

    print_calling_file()

    # 1. Generate random obligor data
    # ----------------------------------------------------------------------------------

    # Pushing 10,000 names to give the JIT-compiled kernels something to chew on.
    rng = np.random.default_rng(RNG_SEED)

    obligor_df = pd.DataFrame(
        {
            # Unique identifier for each obligor
            "uid": np.arange(NUM_OBLIGORS, dtype=np.int64),
            # EAD: Uniformly distributed between 10k and 1M
            "ead": rng.uniform(10_000, 1_000_000, size=NUM_OBLIGORS),
            # PD: Skewed toward realistic lower PDs
            "pd": rng.beta(a=1.5, b=20.0, size=NUM_OBLIGORS),
            # LGD: Uniformly distributed between 10% and 90%
            "lgd": rng.uniform(0.1, 0.9, size=NUM_OBLIGORS),
        }
    )

    # 2. Construct the request using the dataframe ergonomic bridge
    # ----------------------------------------------------------------------------------
    req = CreditVarRequest.from_dataframe(
        obligor_table=obligor_df,
        model="independent_bernoulli",
        n_scenarios=NUM_SCENARIOS,  # Default [cite: 138]
        alphas=[0.9500, 0.9750, 0.9900, 0.9990],  # Default [cite: 129]
        seed=RNG_SEED,
    )

    # 3. Fire the engine
    # ----------------------------------------------------------------------------------
    result = run_credit_var(req)

    # 4. Display the minimalist output
    # ----------------------------------------------------------------------------------
    print("-" * 81)
    print("Credit VaR Run Results")
    print("-" * 81)
    print(f"Model:          {result.model}")
    print(f"Obligors:       {result.provenance.n_obligors:,}")
    print(f"Scenarios:      {result.provenance.n_scenarios:,}")
    print(f"Expected Loss:  {result.summary.expected_loss:,.2f}")
    print("-" * 81)

    for alpha, var in result.summary.var.items():
        es = result.summary.es[alpha]
        print(f"VaR ({alpha}):  {var:>18,.0f} | ES: {es:>18,.0f}")
    print("-" * 81)

    print_success_message()


if __name__ == "__main__":
    main()
