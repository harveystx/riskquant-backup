# src/riskquant/ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_students_t_global_chi.py
"""
Canonical single-factor Student-t-copula JTD ingest case.

Same three obligors as the Gaussian case (uids 1-3), with per-obligor
`nu` (degrees of freedom) and a per-obligor chi factor whose dof
matches the obligor's nu. PD and LGD live on the obligor master.

Mathematical invariant: for obligor i, `nu_i` must equal the
generator's `nu_by_chi` entry for the chi factor `chi_squared_factor[i]`
points at. The canonical registry encodes this 1:1 alignment in
`CANONICAL_T_COPULA_NU_BY_CHI` — any change here must mirror there.
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe import (
    CANONICAL_GAUSSIAN_COPULA_FACTOR_LOADING_BY_IDIO_FACTOR,
    CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_NAME_BY_IDIO_FACTOR,
    CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY,
)
from riskquant.ingest_fixtures.schemas import TCopulaJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> TCopulaJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.COPULA_SF_STUDENTS_T.name] * 3,
        "factor_loading": list(CANONICAL_GAUSSIAN_COPULA_FACTOR_LOADING_BY_IDIO_FACTOR),
        "systemic_factor": [
            "GLOBAL_CREDIT",
            "GLOBAL_CREDIT",
            "GLOBAL_CREDIT",
        ],
        "idiosyncratic_factor": list(
            CANONICAL_GAUSSIAN_COPULA_IDIO_FACTOR_NAME_BY_IDIO_FACTOR
        ),
        "chi_squared_factor": ["GLOBAL_CHI"] * 3,
        "nu": [CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY["GLOBAL_CHI"]] * 3,
    }


CANONICAL_T_COPULA_JTD_INGEST_DATA: TCopulaJTDIngestData = _get_data()

validate_column_lengths(CANONICAL_T_COPULA_JTD_INGEST_DATA)

CANONICAL_T_COPULA_JTD_INGEST_DF = pd.DataFrame(CANONICAL_T_COPULA_JTD_INGEST_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical T-Copula JTD Ingest Data:")
    print_dict(CANONICAL_T_COPULA_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_T_COPULA_JTD_INGEST_DF, float_format="{:>_.2f}")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
