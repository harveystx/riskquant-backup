# src/riskquant/ingest_fixtures/credit/obligors/minimal_obligors_master.py
"""
Canonical obligor master for tests and demos.

Twelve obligors covering the full PD spectrum from AAA (PD=0.0001) to
distressed CCC- (PD=0.18). Sized as the *union* of every canonical JTD
case's obligor uids: obligors 1-3 are also referenced by the canonical
Gaussian and Student-t JTD ingest cases; obligors 1-12 collectively
power the canonical deterministic JTD stress test.

PD and LGD values are calibrated to exercise the sub-IG filter at the
default 0.005 threshold, including:
- Three IG names (obligors 4, 7, 11) that the sub-IG mask should drop.
- One distressed Sub-IG name (obligor 5) with extreme PD/LGD.
- One Sub-IG name (obligor 6) with deep collateralization (small LGD).
"""

from __future__ import annotations

import polars as pl

from riskquant.ingest_fixtures.schemas import ObligorCreditIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================

CANONICAL_OBLIGOR_UIDS: list[int] = list(range(1, 13))

CANONICAL_OBLIGOR_NAMES: list[str] = [
    f"OBLIGOR_{uid:02.0f}" for uid in CANONICAL_OBLIGOR_UIDS
]

CANONICAL_OBLIGOR_PDS: list[float] = [
    0.0200,  # 1.  BB
    0.0500,  # 2.  B+
    0.0100,  # 3.  BB+
    0.0003,  # 4.  AA-  (IG — sub-IG filter drops)
    0.1800,  # 5.  CCC- (distressed)
    0.0350,  # 6.  B
    0.0015,  # 7.  A    (IG)
    0.0400,  # 8.  B
    0.0040,  # 9.  BBB- (borderline IG)
    0.0800,  # 10. B-
    0.0001,  # 11. AAA  (IG)
    0.0650,  # 12. B-
]

CANONICAL_OBLIGOR_LGDS: list[float] = [
    0.60,  # 1
    0.40,  # 2
    0.50,  # 3
    0.15,  # 4  (IG, low LGD)
    0.85,  # 5  (distressed, unsecured)
    0.10,  # 6  (sub-IG, heavily collateralized)
    0.35,  # 7  (IG)
    0.65,  # 8
    0.20,  # 9  (IG)
    0.55,  # 10
    0.10,  # 11 (IG)
    0.90,  # 12 (sub-IG, deep subordination)
]

CANONCIAL_OBLIGOR_RATING_IDS: list[int] = [
    0
] * 12  # Placeholder until we have a canonical rating taxonomy
CANONCIAL_OBLIGOR_SECTOR_IDS: list[int] = [
    0
] * 12  # Placeholder until we have a canonical sector taxonomy
CANONCIAL_OBLIGOR_COUNTRY_IDS: list[int] = [
    0
] * 12  # Placeholder until we have a canonical country taxonomy
CANONCIAL_OBLIGOR_PARENT_UIDS: list[None] = [
    None
] * 12  # All obligors are ultimates here


def _get_data() -> ObligorCreditIngestData:
    return {
        "uid": CANONICAL_OBLIGOR_UIDS,
        "name": CANONICAL_OBLIGOR_NAMES,
        # Diverse PD spectrum: AAA -> CCC- (matches the original
        # deterministic-stress-test calibration).
        "pd": CANONICAL_OBLIGOR_PDS,
        "lgd": CANONICAL_OBLIGOR_LGDS,
        # rating_id, sector_id, country_id: placeholder int codes. The
        # canonical IntEnum taxonomies (credit ratings, sectors, ISO
        # country codes) are pending; use 0 = unknown for now.
        "rating_id": CANONCIAL_OBLIGOR_RATING_IDS,
        "sector_id": CANONCIAL_OBLIGOR_SECTOR_IDS,
        "country_id": CANONCIAL_OBLIGOR_COUNTRY_IDS,
        # All canonical obligors are ultimates here. None -> NO_PARENT_OBLIGOR
        # at compile time. To exercise the parent_uid path in tests, mutate
        # a copy of this dict (e.g. set obligor 8's parent to obligor 2).
        "parent_uid": CANONCIAL_OBLIGOR_PARENT_UIDS,
    }


CANONICAL_OBLIGOR_CREDIT_INGEST_DATA: ObligorCreditIngestData = _get_data()

validate_column_lengths(CANONICAL_OBLIGOR_CREDIT_INGEST_DATA)

CANONICAL_OBLIGOR_CREDIT_INGEST_DF = pl.DataFrame(CANONICAL_OBLIGOR_CREDIT_INGEST_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(
        CANONICAL_OBLIGOR_CREDIT_INGEST_DATA,
        title="Canonical Obligor Credit Ingest Data",
    )

    print_df(
        CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
        title="Canonical Obligor Credit Ingest Data DataFrame",
        float_format="{:_.5f}",  # PD and LGD columns
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
