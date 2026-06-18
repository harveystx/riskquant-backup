# src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py
"""
Canonical credit factor-universe master for tests and demos.

This is the single-source-of-truth authoring table for the credit factor
space — the credit analogue of the `risk_static_config` canonical fixture,
but a single flat table (one row per factor) rather than a dict-of-tables,
because the factor universe has no entity hierarchy.

Compiling it with `compile_credit_factor_universe` yields a
`CompiledCreditFactorUniverse`, from which every downstream consumer DERIVES
what it needs: obligor JTD frames resolve `factor_name -> UIX` against it, the
latent-factor generators read their chi / uniform / indicator column config
from it, and the JTD ingest reconciles obligor `nu` against `chi_nu_by_uix`.

Canonical factor set
--------------------
A full-taxonomy superset covering every `CreditFactorKind`, sized as the union
of what the canonical JTD ingest cases reference:

    UIX  factor_name      kind               nu
    0    GLOBAL_CREDIT    SYSTEMIC           -      systemic Z (Gaussian + t)
    1    IDIO_01          IDIOSYNCRATIC      -      obligor 1 idiosyncratic eps
    2    IDIO_02          IDIOSYNCRATIC      -      obligor 2
    3    IDIO_03          IDIOSYNCRATIC      -      obligor 3
    4    GLOBAL_CHI       CHI                5.0    shared mixing W (global t-copula)
    5    CHI_SECTOR_01    CHI                7.0    grouped-t (Daul et al.) sector W
    6    CHI_SECTOR_02    CHI                9.0
    7    CHI_SECTOR_03    CHI                11.0
    8    U_RETAIL         UNIFORM            -      independent-Bernoulli draw
    9    STRESS_GLOBAL    STRESS_INDICATOR   -      stress-regime indicator

factor_id == UIX here (dense, ascending) — the canonical case keeps identity
and coordinate aligned; the schema permits them to diverge (e.g. a sparse DB
id) without touching any consumer.

The chi degrees of freedom match the legacy
`CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY` so the canonical Student-t JTD
case (which references GLOBAL_CHI at nu=5) reconciles against this master
unchanged.
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.masters.factor_universe.compiled import (
    CompiledCreditFactorUniverse,
)
from riskquant.credit.masters.factor_universe.ingest import (
    compile_credit_factor_universe,
)
from riskquant.ingest_fixtures.schemas import CreditFactorUniverseIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================

CANONICAL_CREDIT_FACTOR_IDS: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
]

CANONICAL_CREDIT_FACTOR_NAMES: list[str] = [
    "GLOBAL_CREDIT",  # 0 systemic factor (Gaussian + t)
    "IDIO_01",  # 1 obligor 1 idiosyncratic eps
    "IDIO_02",  # 2 obligor 2 idiosyncratic eps
    "IDIO_03",  # 3 obligor 3 idiosyncratic eps
    "GLOBAL_CHI",  # 4 shared mixing W (global t-copula)
    "CHI_SECTOR_01",  # 5 grouped-t (Daul et al.) sector W
    "CHI_SECTOR_02",  # 6 grouped-t (Daul et al.) sector W
    "CHI_SECTOR_03",  # 7 grouped-t (Daul et al.) sector W
    "U_01",  # 8 obligor 1 independent-Bernoulli uniform (per-obligor: conditional independence)
    "U_02",  # 9 obligor 2 independent-Bernoulli uniform
    "U_03",  # 10 obligor 3 independent-Bernoulli uniform
    "STRESS_GLOBAL",  # 11 shared stress-regime indicator
    "GLOBAL_BETA_PD",  # 12 shared Beta(alpha, beta) common-PD factor (beta-binomial)
    "SECTOR_ENERGY",  # 13 multi-factor Gaussian copula systemic (sector)
    "SECTOR_FINANCIALS",  # 14 multi-factor Gaussian copula systemic (sector)
]

CANONICAL_CREDIT_FACTOR_KINDS: list[str] = [
    "SYSTEMIC",
    "IDIOSYNCRATIC",
    "IDIOSYNCRATIC",
    "IDIOSYNCRATIC",
    "CHI",
    "CHI",
    "CHI",
    "CHI",
    "UNIFORM",
    "UNIFORM",
    "UNIFORM",
    "STRESS_INDICATOR",
    "BETA_PD",
    "SYSTEMIC",
    "SYSTEMIC",
]

# nu present only on CHI rows (UIX 4-7); None elsewhere -> NO_CHI_NU sentinel.
CANONICAL_CREDIT_FACTOR_NUS: list[float | None] = [
    None,  # 0 GLOBAL_CREDIT
    None,  # 1 IDIO_01
    None,  # 2 IDIO_02
    None,  # 3 IDIO_03
    5.0,  # 4 GLOBAL_CHI
    7.0,  # 5 CHI_SECTOR_01
    9.0,  # 6 CHI_SECTOR_02
    11.0,  # 7 CHI_SECTOR_03
    None,  # 8 U_01
    None,  # 9 U_02
    None,  # 10 U_03
    None,  # 11 STRESS_GLOBAL
    None,  # 12 GLOBAL_BETA_PD
    None,  # 13 SECTOR_ENERGY
    None,  # 14 SECTOR_FINANCIALS
]

CANONICAL_CREDIT_FACTOR_BETA_PD_PARAMS: list[tuple[float, float] | None] = [
    None,  # 0 GLOBAL_CREDIT
    None,  # 1 IDIO_01
    None,  # 2 IDIO_02
    None,  # 3 IDIO_03
    None,  # 4 GLOBAL_CHI
    None,  # 5 CHI_SECTOR_01
    None,  # 6 CHI_SECTOR_02
    None,  # 7 CHI_SECTOR_03
    None,  # 8 U_01
    None,  # 9 U_02
    None,  # 10 U_03
    None,  # 11 STRESS_GLOBAL
    (2.0, 18.0),  # 12 GLOBAL_BETA_PD
    None,  # 13 SECTOR_ENERGY
    None,  # 14 SECTOR_FINANCIALS
]

CANONICAL_CREDIT_FACTOR_ALPHAS = [
    params[0] if params is not None else None
    for params in CANONICAL_CREDIT_FACTOR_BETA_PD_PARAMS
]

CANONICAL_CREDIT_FACTOR_BETAS = [
    params[1] if params is not None else None
    for params in CANONICAL_CREDIT_FACTOR_BETA_PD_PARAMS
]


def _get_data() -> CreditFactorUniverseIngestData:
    return {
        "factor_id": CANONICAL_CREDIT_FACTOR_IDS,
        "factor_name": CANONICAL_CREDIT_FACTOR_NAMES,
        "factor_kind": CANONICAL_CREDIT_FACTOR_KINDS,
        "nu": CANONICAL_CREDIT_FACTOR_NUS,
        "alpha": CANONICAL_CREDIT_FACTOR_ALPHAS,
        "beta": CANONICAL_CREDIT_FACTOR_BETAS,
    }


CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DATA: CreditFactorUniverseIngestData = (
    _get_data()
)

validate_column_lengths(CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DATA)

CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DF = pd.DataFrame(
    CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DATA
)


def build_canonical_credit_factor_universe() -> CompiledCreditFactorUniverse:
    """Compile the canonical authoring table into the factor-universe master.

    Recompiles on each call (cheap; the table is tiny). Consumers that want a
    shared instance should compile once and pass it through, exactly as demos
    do with the obligor and netting-set masters.
    """
    return compile_credit_factor_universe(CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DF)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(
        CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DATA,
        title="Canonical Credit Factor-Universe Ingest Data",
    )

    print_df(
        CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DF,
        title="Canonical Credit Factor-Universe Ingest Data DataFrame",
    )

    universe = build_canonical_credit_factor_universe()
    print_dict(
        universe.factor_name_to_uix,
        title="Compiled factor_name -> UIX",
    )
    print_dict(
        universe.chi_nu_by_uix,
        title="Compiled chi UIX -> nu",
    )
    print_dict(
        universe.beta_params_by_uix,
        title="Compiled Beta Params UIX -> (alpha, beta)",
    )

    print_success_message(add_space_before=True)


if __name__ == "__main__":
    _debug_print()
