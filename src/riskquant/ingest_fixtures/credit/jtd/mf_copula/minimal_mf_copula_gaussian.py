# src/riskquant/ingest_fixtures/credit/jtd/mf_copula/minimal_mf_copula_gaussian.py
"""
Canonical multi-factor Gaussian-copula JTD ingest case.

Three obligors (uids 1-3). The per-obligor frame carries only the dispatch keys
and the idiosyncratic factor; the systemic loadings are ragged (each obligor
loads on GLOBAL_CREDIT plus a different sector factor), so they live in a
SEPARATE long-format loadings frame — one row per obligor x systemic factor —
packed into CSR at compile time. PD and LGD are joined off the canonical obligor
master.

    A_i = sum_k beta_ik * Z_k + sqrt(1 - sum beta^2) * eps_i,  default if A_i < t_i

with the systemic factors (GLOBAL_CREDIT, SECTOR_*) independent standard normals
in the canonical universe. sum beta^2 < 1 for every obligor (room for idio).
"""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.schemas import (
    MfCopulaLoadingsData,
    MfGaussianCopulaJTDIngestData,
)
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# PER-OBLIGOR FRAME (one row per obligor)
# ============================================================


def _get_data() -> MfGaussianCopulaJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.COPULA_MF_GAUSSIAN.name] * 3,
        "idiosyncratic_factor": ["IDIO_01", "IDIO_02", "IDIO_03"],
    }


CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DATA: MfGaussianCopulaJTDIngestData = (
    _get_data()
)

validate_column_lengths(CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DATA)

CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF = pl.DataFrame(
    CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DATA
)


# ============================================================
# LONG-FORMAT LOADINGS (one row per obligor x systemic factor)
# ============================================================
# obligor 1: GLOBAL_CREDIT 0.30, SECTOR_ENERGY     0.70  -> sum b^2 = 0.58
# obligor 2: GLOBAL_CREDIT 0.40, SECTOR_ENERGY     0.60  -> sum b^2 = 0.52
# obligor 3: GLOBAL_CREDIT 0.60, SECTOR_FINANCIALS 0.40  -> sum b^2 = 0.52


def _get_loadings() -> MfCopulaLoadingsData:
    return {
        "obligor_uid": [1, 1, 2, 2, 3, 3],
        "factor_name": [
            "GLOBAL_CREDIT",
            "SECTOR_ENERGY",
            "GLOBAL_CREDIT",
            "SECTOR_ENERGY",
            "GLOBAL_CREDIT",
            "SECTOR_FINANCIALS",
        ],
        "factor_loading": [0.30, 0.70, 0.40, 0.60, 0.60, 0.40],
    }


CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DATA: MfCopulaLoadingsData = _get_loadings()

validate_column_lengths(CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DATA)

CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF = pl.DataFrame(
    CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Multi-Factor Gaussian-Copula JTD Ingest Data (per obligor):")
    print_dict(CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DATA)
    print_df(CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF)

    print("Long-format loadings (one row per obligor x systemic factor):")
    print_dict(CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DATA)
    print_df(CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF, float_format="{:>_.2f}")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
