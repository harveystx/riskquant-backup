# src/riskquant/ingest_fixtures/credit_factors/minimal_factor_universe.py
"""
Canonical credit-factor REGISTRY for tests and demos.

This file is structural metadata, NOT historical data. It defines:
- the set of factor *names* that obligor JTD ingest cases may reference,
- a name -> int32 UIX mapping (`factor_name_to_uix`) that
  `compile_jtd_portfolio_from_df` uses to dereference factor names into
  `CreditFactorMatrix` row indices,
- which UIX positions carry a chi-square mixing variable (t-copula), and
  the degrees of freedom for each one.

The credit code path is currently parametric-simulation only — the
copula generators draw fresh latent factors every run from
`np.random.default_rng`. No part of the credit layer consumes a
historical time series of credit-factor levels.

If/when historical or FHS scenario generation lands for credit, a
parallel `ingest_cases/credit_factors/historical/` would supply
`(T, F_credit)` factor observations through the existing
`market_data/risk_factors/matrix.py::RiskFactorMatrix` container —
F-contiguous, int32 ordinal dates, int32 factor ids — which is already
the right shape for credit history too. Until then, only the registry
metadata below is required.

Chi/nu alignment (mathematical invariant)
-----------------------------------------
Each chi-square factor in the registry has its own dof
(`nu_by_chi[c]`). Every obligor that references a chi factor MUST
have `obligor.nu == nu_by_chi[chi_uix]` — otherwise the kernel's
`sqrt(nu_i / W_i)` transform breaks the t-marginal. The canonical
t-copula ingest case is built per-obligor-chi to satisfy this:
obligor i references CHI_i with dof matching its own nu_i.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

# ==============================================================================
# FACTOR NAMES (UIX = position in this tuple)
# ==============================================================================

CANONICAL_CREDIT_FACTOR_NAMES: Final[tuple[str, ...]] = (
    "GLOBAL_CREDIT",  # 0  — systemic Z (Gaussian, t)
    "IDIO_1",  # 1  — idiosyncratic eps (obligor 1)
    "IDIO_2",  # 2  — idiosyncratic eps (obligor 2)
    "IDIO_3",  # 3  — idiosyncratic eps (obligor 3)
    "CHI_1",  # 4  — chi-square W_1 for obligor 1 (nu=5)
    "CHI_2",  # 5  — chi-square W_2 for obligor 2 (nu=7)
    "CHI_3",  # 6  — chi-square W_3 for obligor 3 (nu=10)
)

CANONICAL_CREDIT_FACTOR_COUNT: Final[int] = len(CANONICAL_CREDIT_FACTOR_NAMES)

CANONICAL_CREDIT_FACTOR_NAME_TO_UIX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(CANONICAL_CREDIT_FACTOR_NAMES)
}

CANONICAL_CREDIT_FACTOR_IDS: Final[npt.NDArray[np.int32]] = np.arange(
    CANONICAL_CREDIT_FACTOR_COUNT, dtype=np.int32
)

# ==============================================================================
# T-COPULA CHI CONFIGURATION
# ==============================================================================
# One chi-square factor per obligor (uids 1, 2, 3). Each obligor's nu_i
# matches the dof of its chi factor — the only configuration that yields
# a true Student-t marginal under `sqrt(nu_i / W_i) * (beta*Z + ... )`.

CANONICAL_T_COPULA_CHI_COLUMNS: Final[npt.NDArray[np.int32]] = np.array(
    [
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["CHI_1"],
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["CHI_2"],
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["CHI_3"],
    ],
    dtype=np.int32,
)


CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY: Final[dict[str, float]] = {
    "CHI_1": 5.0,
    "CHI_2": 7.0,
    "CHI_3": 10.0,
}


# Aligned to CANONICAL_T_COPULA_CHI_COLUMNS by index. The canonical
# t-copula ingest case uses the SAME values per-obligor.
CANONICAL_T_COPULA_NU_BY_CHI_FACTOR: Final[npt.NDArray[np.float64]] = np.array(
    list(CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY.values()),
    dtype=np.float64,
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Credit-Factor Registry - Factor Name to UIX:")
    print_dict(CANONICAL_CREDIT_FACTOR_NAME_TO_UIX)

    print("\nT-Copula Chi-Factor to Nu Registry:")
    print_dict(CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY)
    print(f"  chi_factor_uix : {CANONICAL_T_COPULA_CHI_COLUMNS.tolist()}")
    print(f"  nu_by_chi          : {CANONICAL_T_COPULA_NU_BY_CHI_FACTOR.tolist()}")

    print_success_message(add_space_before=True)


if __name__ == "__main__":
    _debug_print()
