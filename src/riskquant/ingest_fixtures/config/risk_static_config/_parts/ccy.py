# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/ccy.py

from __future__ import annotations

import polars as pl

from riskquant.ingest_fixtures.schemas import RiskStaticConfigCcy
from riskquant.ingest_fixtures.utils import validate_column_lengths

CANONICAL_CCY_NAME = ["AUD", "CHF", "EUR", "GBP", "USD"]
CANONICAL_CCY_ID = [1, 2, 3, 4, 5]


def _get_data() -> RiskStaticConfigCcy:
    return {
        "ccy_name": CANONICAL_CCY_NAME,
        "ccy_id": CANONICAL_CCY_ID,
    }


CANONICAL_CCY_DATA: RiskStaticConfigCcy = _get_data()


validate_column_lengths(CANONICAL_CCY_DATA)

_CANONICAL_CCY_DF = pl.DataFrame(CANONICAL_CCY_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(CANONICAL_CCY_DATA, title="Canonical Currencies Dictionary")

    print("Canonical Currencies DataFrame:")
    print(_CANONICAL_CCY_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
