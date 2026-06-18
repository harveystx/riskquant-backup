# src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe.py
"""
Canonical credit-factor REGISTRY for tests and demos.

This file is structural metadata, NOT historical data. It acts as a derived
view of the canonical factor master.

It defines:
- the set of factor *names* that obligor JTD ingest cases may reference.
- a name -> int32 UIX mapping.
- which UIX positions carry a chi-square mixing variable (t-copula).
- which UIX positions carry beta shape parameters for stochastic PDs.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe_master import (
    build_canonical_credit_factor_universe,
)

# Compile the single-source-of-truth master to derive registry constants
_universe = build_canonical_credit_factor_universe()
_uix_to_name = {uix: name for name, uix in _universe.factor_name_to_uix.items()}

# ==============================================================================
# FACTOR NAMES (UIX = position in this tuple)
# ==============================================================================

CANONICAL_CREDIT_FACTOR_NAME_TO_UIX: Final[dict[str, int]] = (
    _universe.factor_name_to_uix
)

CANONICAL_CREDIT_FACTOR_NAMES: Final[tuple[str, ...]] = tuple(
    name for _, name in sorted(_uix_to_name.items())
)

CANONICAL_CREDIT_FACTOR_COUNT: Final[int] = len(CANONICAL_CREDIT_FACTOR_NAMES)

CANONICAL_CREDIT_FACTOR_IDS: Final[npt.NDArray[np.int32]] = np.arange(
    CANONICAL_CREDIT_FACTOR_COUNT, dtype=np.int32
)

# ==============================================================================
# GAUSSIAN-COPULA PORTFOLIO FIXTURE DATA
# (Mock values used to construct the obligor DataFrame, not universal truths)
# ==============================================================================

CANONICAL_GAUSSIAN_COPULA_IDIO_COLUMNS_UIX: Final[npt.NDArray[np.int32]] = np.array(
    [
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_01"],
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_02"],
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX["IDIO_03"],
    ],
    dtype=np.int32,
)

CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_TO_FACTOR_LOADING_REGISTRY: Final[
    dict[str, float]
] = {
    "IDIO_01": 0.25,
    "IDIO_02": 0.35,
    "IDIO_03": 0.15,
}

CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_NAME_BY_IDIO_FACTOR: tuple[str, ...] = tuple(
    CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_TO_FACTOR_LOADING_REGISTRY.keys()
)

CANONICAL_GAUSSIAN_COPULA_FACTOR_LOADING_BY_IDIO_FACTOR: Final[
    npt.NDArray[np.float64]
] = np.array(
    list(CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_TO_FACTOR_LOADING_REGISTRY.values()),
    dtype=np.float64,
)

# ==============================================================================
# T-COPULA CHI CONFIGURATION (Derived dynamically from Master)
# ==============================================================================

CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY: Final[dict[str, float]] = {
    _uix_to_name[uix]: float(nu) for uix, nu in _universe.chi_nu_by_uix.items()
}

CANONICAL_T_COPULA_CHI_FACTOR_NAME_BY_CHI_FACTOR: tuple[str, ...] = tuple(
    CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY.keys()
)

CANONICAL_T_COPULA_CHI_COLUMNS_UIX: Final[npt.NDArray[np.int32]] = np.array(
    [
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX[name]
        for name in CANONICAL_T_COPULA_CHI_FACTOR_NAME_BY_CHI_FACTOR
    ],
    dtype=np.int32,
)

CANONICAL_T_COPULA_NU_BY_CHI_FACTOR: Final[npt.NDArray[np.float64]] = np.array(
    list(CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY.values()),
    dtype=np.float64,
)

# ==============================================================================
# BETA-BINOMIAL CONFIGURATION (Derived dynamically from Master)
# ==============================================================================

CANONICAL_BETA_BINOMIAL_FACTOR_TO_PARAMS_REGISTRY: Final[
    dict[str, tuple[float, float]]
] = {
    _uix_to_name[uix]: (float(params[0]), float(params[1]))
    for uix, params in _universe.beta_params_by_uix.items()
}

CANONICAL_BETA_BINOMIAL_FACTOR_NAMES: tuple[str, ...] = tuple(
    CANONICAL_BETA_BINOMIAL_FACTOR_TO_PARAMS_REGISTRY.keys()
)

CANONICAL_BETA_BINOMIAL_COLUMNS_UIX: Final[npt.NDArray[np.int32]] = np.array(
    [
        CANONICAL_CREDIT_FACTOR_NAME_TO_UIX[name]
        for name in CANONICAL_BETA_BINOMIAL_FACTOR_NAMES
    ],
    dtype=np.int32,
)

CANONICAL_BETA_BINOMIAL_SHAPE_A: Final[npt.NDArray[np.float64]] = np.array(
    [
        params[0]
        for params in CANONICAL_BETA_BINOMIAL_FACTOR_TO_PARAMS_REGISTRY.values()
    ],
    dtype=np.float64,
)

CANONICAL_BETA_BINOMIAL_SHAPE_B: Final[npt.NDArray[np.float64]] = np.array(
    [
        params[1]
        for params in CANONICAL_BETA_BINOMIAL_FACTOR_TO_PARAMS_REGISTRY.values()
    ],
    dtype=np.float64,
)

# ==============================================================================
# DEBUG PRINT
# ==============================================================================


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
    print(f"  chi_factor_uix : {CANONICAL_T_COPULA_CHI_COLUMNS_UIX.tolist()}")
    print(f"  nu_by_chi      : {CANONICAL_T_COPULA_NU_BY_CHI_FACTOR.tolist()}")

    print("\nBeta-Binomial Parameter Registry:")
    print_dict(CANONICAL_BETA_BINOMIAL_FACTOR_TO_PARAMS_REGISTRY)
    print(f"  beta_factor_uix: {CANONICAL_BETA_BINOMIAL_COLUMNS_UIX.tolist()}")
    print(f"  shape_a        : {CANONICAL_BETA_BINOMIAL_SHAPE_A.tolist()}")
    print(f"  shape_b        : {CANONICAL_BETA_BINOMIAL_SHAPE_B.tolist()}")

    print_success_message(add_space_before=True)


if __name__ == "__main__":
    _debug_print()
