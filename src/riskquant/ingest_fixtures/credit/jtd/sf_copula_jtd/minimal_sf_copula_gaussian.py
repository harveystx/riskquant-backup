# src/riskquant/ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_gaussian.py
"""
Canonical single-factor Gaussian-copula JTD ingest case.

Three obligors (uids 1-3), referencing the canonical credit-factor
universe (one systemic factor + three idiosyncratic factors). PD and
LGD are NOT carried here — they are joined off the canonical obligor
master at compile time.
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe import (
    CANONICAL_GAUSSIAN_COPULA_FACTOR_LOADING_BY_IDIO_FACTOR,
    CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_NAME_BY_IDIO_FACTOR,
)
from riskquant.ingest_fixtures.schemas import GaussianCopulaJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> GaussianCopulaJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.COPULA_SF_GAUSSIAN.name] * 3,
        "factor_loading": list(CANONICAL_GAUSSIAN_COPULA_FACTOR_LOADING_BY_IDIO_FACTOR),
        "systemic_factor": [
            "GLOBAL_CREDIT",
            "GLOBAL_CREDIT",
            "GLOBAL_CREDIT",
        ],
        "idiosyncratic_factor": list(
            CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_NAME_BY_IDIO_FACTOR
        ),
    }


CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA: GaussianCopulaJTDIngestData = _get_data()

validate_column_lengths(CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA)

CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF = pd.DataFrame(
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Gaussian-Copula JTD Ingest Data:")
    print_dict(CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF, float_format="{:>_.2f}")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
