# src/riskquant/ingest_fixtures/jtd/copula_jtd/sf_t_minimal.py
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
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
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
        "instrument": [JtdModel.COPULA_SF_T.name] * 3,
        "beta": [0.25, 0.35, 0.15],
        "systemic_factor": ["GLOBAL_CREDIT"]
        * 3,  # systemic factor is the same for all obligors
        "idiosyncratic_factor": [
            "IDIO_1",
            "IDIO_2",
            "IDIO_3",
        ],
        # Per-obligor chi factor. Each obligor's chi has its own dof
        # in the generator (CANONICAL_T_COPULA_NU_BY_CHI) and `nu`
        # below MUST match that dof. The kernel uses both: chi_uix
        # selects W_i, nu_i scales it via sqrt(nu_i / W_i).
        "chi_squared_factor": list(CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY.keys()),
        "nu": list(CANONICAL_T_COPULA_CHI_FACTOR_TO_NU_REGISTRY.values()),
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
    print_df(CANONICAL_T_COPULA_JTD_INGEST_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
