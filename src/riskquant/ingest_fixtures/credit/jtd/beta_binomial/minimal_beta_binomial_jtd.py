# src/riskquant/ingest_fixtures/credit/jtd/beta_binomial/minimal_beta_binomial_jtd.py
"""
Canonical beta-binomial JTD ingest case.

Three obligors (uids 1-3): a single shared per-scenario default probability
``P_s ~ Beta(alpha, beta)`` (drawn into the GLOBAL_BETA_PD factor row) drives
conditionally-independent defaults ``u_i < P_s``. Uniform factors are
per-obligor distinct (U_01 / U_02 / U_03 — conditional independence given the
common shock); the beta factor is shared (GLOBAL_BETA_PD, the common shock; its
alpha/beta live on the universe master). PD is NOT carried — it is stochastic;
only LGD is joined off the canonical obligor master at compile time.
"""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.schemas import BetaBinomialJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> BetaBinomialJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.BETA_BINOMIAL.name] * 3,
        "uniform_factor": ["U_01", "U_02", "U_03"],
        "beta_factor": ["GLOBAL_BETA_PD", "GLOBAL_BETA_PD", "GLOBAL_BETA_PD"],
    }


CANONICAL_BETA_BINOMIAL_JTD_INGEST_DATA: BetaBinomialJTDIngestData = _get_data()

validate_column_lengths(CANONICAL_BETA_BINOMIAL_JTD_INGEST_DATA)

CANONICAL_BETA_BINOMIAL_JTD_INGEST_DF = pl.DataFrame(
    CANONICAL_BETA_BINOMIAL_JTD_INGEST_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Beta-Binomial JTD Ingest Data:")
    print_dict(CANONICAL_BETA_BINOMIAL_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_BETA_BINOMIAL_JTD_INGEST_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
