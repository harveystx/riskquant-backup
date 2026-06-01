# src/riskquant/ingest_fixtures/netting_set/minimal_netting_set_master.py
"""
Canonical netting-set master for tests and demos.

Expanded to include 15 netting sets mapped across all 12 distinct
obligors in `minimal_obligors_master.py`, demonstrating 1:1 and N:1
netting-set-to-obligor mappings across the full PD spectrum.

- 101, 102, 103: Original canonical cases (small, zero, over-collateralized).
- 104: Provides a second netting set for obligor 2.
- 105: Maps to obligor 4 (IG) with negative collateral (posted collateral).
- 106, 107: Mixed collateral amounts for obligor 5 (distressed CCC-).
- 108: Zero collateral baseline for obligor 6.
- 109-115: Coverage for obligors 7 through 12, mixing IG and sub-IG
  collateralization scenarios (including a multi-set mapping for obligor 12).
"""

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_UIDS,
)
from riskquant.ingest_fixtures.schemas import NettingSetMasterIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================

# Netting set UIDs 101 through 115
CANONICAL_NETTING_SET_UIDS: tuple[int, ...] = tuple(set(range(101, 116)))

CANONICAL_NETTING_SET_COLLATERAL_AMOUNTS: tuple[float, ...] = (
    250.0,  # 101: Obligor 1 (small collateral)
    0.0,  # 102: Obligor 2 (zero collateral)
    1_000_000.0,  # 103: Obligor 3 (over-collateralized)
    500.0,  # 104: Obligor 2 (tests aggregation with 102)
    -100.0,  # 105: Obligor 4 (IG, negative/posted collateral)
    10_000.0,  # 106: Obligor 5 (distressed CCC-, N:1 test)
    50_000.0,  # 107: Obligor 5 (distressed CCC-, N:1 test)
    0.0,  # 108: Obligor 6 (sub-IG, zero collateral)
    5_000.0,  # 109: Obligor 7 (IG, moderate collateral)
    0.0,  # 110: Obligor 8 (sub-IG, zero collateral)
    -500.0,  # 111: Obligor 9 (IG, posted collateral)
    1_000.0,  # 112: Obligor 10 (sub-IG, small collateral)
    2_000_000.0,  # 113: Obligor 11 (AAA, massively over-collateralized)
    0.0,  # 114: Obligor 12 (distressed B-, zero collateral)
    250.0,  # 115: Obligor 12 (distressed B-, N:1 test)
)


def _get_data() -> NettingSetMasterIngestData:
    return {
        "uid": CANONICAL_NETTING_SET_UIDS,
        # Mapped to the 12 canonical obligors.
        # Obligors 2, 5, and 12 have multiple netting sets.
        "obligor_uid": [1, 2, 3, 2, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 12],
        "net_collateral": CANONICAL_NETTING_SET_COLLATERAL_AMOUNTS,
    }


CANONICAL_NETTING_SET_MASTER_INGEST_DATA: NettingSetMasterIngestData = _get_data()

validate_column_lengths(CANONICAL_NETTING_SET_MASTER_INGEST_DATA)

CANONICAL_NETTING_SET_MASTER_INGEST_DF = pd.DataFrame(
    CANONICAL_NETTING_SET_MASTER_INGEST_DATA
)


def _validate_obligor_coverage() -> None:
    from riskquant.utils.printer import print_success_message

    obligor_uids = set(CANONICAL_NETTING_SET_MASTER_INGEST_DATA["obligor_uid"])
    expected_obligors = set(
        CANONICAL_OBLIGOR_UIDS
    )  # Expecting obligor UIDs 1 through 12

    missing_obligors = expected_obligors - obligor_uids
    extra_obligors = obligor_uids - expected_obligors

    assert not missing_obligors, (
        f"Missing coverage for obligor UIDs: {missing_obligors}"
    )
    assert not extra_obligors, f"Unexpected obligor UIDs found: {extra_obligors}"

    print_success_message("Obligor coverage validation passed!")


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    # Validate obligor UIDs
    _validate_obligor_coverage()

    print_dict(
        CANONICAL_NETTING_SET_MASTER_INGEST_DATA,
        title="Canonical Netting-Set Master Ingest Data",
    )

    print_df(
        CANONICAL_NETTING_SET_MASTER_INGEST_DF,
        title="Canonical Netting-Set Master Ingest Data DataFrame",
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
