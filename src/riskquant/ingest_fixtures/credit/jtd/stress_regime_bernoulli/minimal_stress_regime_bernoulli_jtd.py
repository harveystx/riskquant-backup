# src/riskquant/ingest_fixtures/credit/jtd/stress_regime_bernoulli/minimal_stress_regime_bernoulli_jtd.py
"""
Canonical stress-regime Bernoulli JTD ingest case.

Three obligors (uids 1-3): independent-Bernoulli defaults with a single shared
STRESS_INDICATOR factor (STRESS_GLOBAL) that, when on, scales each obligor's PD
by its ``stress_pd_multiplier``. Uniform factors are per-obligor distinct
(U_01 / U_02 / U_03); the stress regime factor is global (shared across all
obligors — the builder enforces a single stress UIX). PD and LGD are joined off
the canonical obligor master at compile time.
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.schemas import StressRegimeBernoulliJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> StressRegimeBernoulliJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.STRESS_REGIME_BERNOULLI.name] * 3,
        "uniform_factor": ["U_01", "U_02", "U_03"],
        "stress_regime_factor": ["STRESS_GLOBAL", "STRESS_GLOBAL", "STRESS_GLOBAL"],
        "stress_pd_multiplier": [2.0, 2.5, 3.0],
    }


CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DATA: (
    StressRegimeBernoulliJTDIngestData
) = _get_data()

validate_column_lengths(CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DATA)

CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DF = pd.DataFrame(
    CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Stress-Regime Bernoulli JTD Ingest Data:")
    print_dict(CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_STRESS_REGIME_BERNOULLI_JTD_INGEST_DF, float_format="{:>_.2f}")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
