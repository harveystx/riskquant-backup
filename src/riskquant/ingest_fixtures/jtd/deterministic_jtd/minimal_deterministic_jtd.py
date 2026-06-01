# src/riskquant/ingest_fixtures/jtd/deterministic_jtd/minimal_deterministic_jtd.py
"""
Canonical deterministic JTD ingest case.

Twelve obligors (uids 1-12), all routed through the deterministic JTD
model. The frame carries only `uid` and `instrument` — LGD is joined off
the obligor master at compile time, and EAD flows in per scenario from
the exposure engine (or from a stress-overlay vector for standalone
stress tests). The previous version inlined `ead`, `lgd`, and `pd`;
those have been removed to make the obligor master the single source of
truth and the exposure pipeline the single source of EAD.

To run the legacy `calc_deterministic_jtd.py` stress test against this
case, supply EAD externally via the obligor-keyed stress overlay
(see `CANONICAL_DETERMINISTIC_JTD_EAD_OVERLAY` below).
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.schemas import DeterministicJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> DeterministicJTDIngestData:
    return {
        "uid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "instrument": [JtdModel.DETERMINISTIC.name] * 12,
    }


CANONICAL_DETERMINISTIC_JTD_INGEST_DATA: DeterministicJTDIngestData = _get_data()

validate_column_lengths(CANONICAL_DETERMINISTIC_JTD_INGEST_DATA)

CANONICAL_DETERMINISTIC_JTD_INGEST_DF = pd.DataFrame(
    CANONICAL_DETERMINISTIC_JTD_INGEST_DATA
)

# ------------------------------------------------------------
# STRESS-OVERLAY EAD
# ------------------------------------------------------------
# Stress-test EAD per obligor uid, aligned to the canonical obligor
# universe. This is NOT JTD model state — it's a stress overlay supplied
# externally to `run_jtd` (the production path would derive these from
# the exposure engine instead). Keeping the canonical values colocated
# with the deterministic case so the legacy stress test can reproduce
# its sub-IG concentration result without surgery.

CANONICAL_DETERMINISTIC_JTD_EAD_OVERLAY: dict[int, float] = {
    1: 5_000_000.0,  # Standard Sub-IG
    2: 12_000_000.0,  # Standard Sub-IG
    3: 8_500_000.0,  # Standard Sub-IG
    4: 150_000_000.0,  # Massive IG (Sovereign / Clearinghouse) — sub-IG mask drops
    5: 2_500_000.0,  # Distressed Sub-IG (Small EAD)
    6: 45_000_000.0,  # Large Sub-IG, but highly collateralized
    7: 80_000_000.0,  # Large IG (A-rated corporate)
    8: 22_000_000.0,  # Mid Sub-IG
    9: 3_000_000.0,  # Tiny IG
    10: 35_000_000.0,  # Large Sub-IG (High risk)
    11: 200_000_000.0,  # Massive IG (AAA equivalent)
    12: 14_000_000.0,  # Mid Sub-IG (Very high LGD)
}


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Deterministic JTD Ingest Data:")
    print_dict(CANONICAL_DETERMINISTIC_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_DETERMINISTIC_JTD_INGEST_DF)

    print("Stress-overlay EAD (obligor_uid -> ead):")
    print_dict(CANONICAL_DETERMINISTIC_JTD_EAD_OVERLAY)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
