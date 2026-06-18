# src/riskquant/ingest_fixtures/credit/jtd/bernoulli/minimal_independent_bernoulli_jtd.py
"""
Canonical independent-Bernoulli JTD ingest case.

Three obligors (uids 1-3), each defaulting if its OWN uniform draw falls below
its (fixed, master-sourced) PD: ``u_i < pd_i``. The uniform factors are
per-obligor distinct (U_01 / U_02 / U_03 in the canonical universe) — the
independence the model name promises requires distinct uniform rows. PD and LGD
are joined off the canonical obligor master at compile time.
"""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.ingest_fixtures.schemas import IndependentBernoulliJTDIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# ============================================================
# DATA
# ============================================================


def _get_data() -> IndependentBernoulliJTDIngestData:
    return {
        "uid": [1, 2, 3],
        "instrument": [JtdModel.INDEPENDENT_BERNOULLI.name] * 3,
        "uniform_factor": ["U_01", "U_02", "U_03"],
    }


CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DATA: IndependentBernoulliJTDIngestData = (
    _get_data()
)

validate_column_lengths(CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DATA)

CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DF = pd.DataFrame(
    CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Independent-Bernoulli JTD Ingest Data:")
    print_dict(CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DATA)

    print("As DataFrame:")
    print_df(CANONICAL_INDEPENDENT_BERNOULLI_JTD_INGEST_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
