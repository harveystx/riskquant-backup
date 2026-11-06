# src/riskquant/ingest_fixtures/credit/credit_exposure/minimal_spot_with_netting_sets.py
"""
Canonical market-risk portfolio for the credit-exposure pipeline.

Eight SPOT trades against the canonical static-config underlyings:
- Six are linked to one of three netting sets (101, 102, 103); two carry
  no netting linkage and are excluded from the exposure path by
  `compile_exposure`.
- Quantities are mixed-sign so the RC kernel exercises both the V > C
  branch (set 101 nets long) and the RC = 0 floor (set 102 nets short).

This is a *market-risk* ingest frame — type `MarketRiskIngestData` —
that happens to carry the `netting_set_id` linkage column. Replacement-
cost-specific parameters live on the netting-set master
(`net_collateral`); nothing model-specific is on the trade rows.
"""

from __future__ import annotations

import polars as pl

from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths
from riskquant.portfolio.instruments import InstrumentType


def _get_data() -> MarketRiskIngestData:
    return {
        "uid": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "instrument": [InstrumentType.SPOT.name] * 8,
        # Mixed signs so some netting sets net negative -> RC floors to 0.
        "quantity": [
            10.0,  # 1.  NS 101
            -4.0,  # 2.  NS 101
            6.0,  # 3.  NS 101  -> set 101 nets long
            8.0,  # 4.  NS 102
            -9.0,  # 5.  NS 102  -> set 102 nets short, RC = 0
            5.0,  # 6.  NS 103  -> single-trade netting set
            3.0,  # 7.  NS 106  -> single-trade netting set
            -2.0,  # 8.  NS 107  -> single-trade netting set
        ],
        # Curve-point relative name from the canonical static config.
        # All eight rows point at the same risk factor so the exposure
        # path resolves them to a single source.
        "underlying": ["BRENT_CRUDE_FWD[M+0]"] * 8,
        "settlement_ccy": ["USD"] * 8,
        # Effective-netting-set linkage. None -> NO_NETTING_SET (dropped
        # by compile_exposure). Trades 1-3 share 101, 4-5 share 102,
        # 6, 7, 8 are alone.
        "netting_set_id": [101, 101, 101, 102, 102, 103, 106, 107],
        # Irrelevant for SPOT.
        "strike_price": [None] * 8,
        "strike_ccy": [None] * 8,
        "is_call": [None] * 8,
        "expiry_date": [None] * 8,
        "delivery_date": [None] * 8,
        "settlement_date": [None] * 8,
        "under_discount_curve": [None] * 8,
        "strike_discount_curve": [None] * 8,
        "volatility_container_name": [None] * 8,
    }


CANONICAL_CREDIT_EXPOSURE_INGEST_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(CANONICAL_CREDIT_EXPOSURE_INGEST_DATA)

# Polars at the edge (D094: the portfolio ingest takes pl.DataFrame only). The
# frame is built column by column, so every column dtype comes from its full
# list; the all-None columns are pl.Null.
CANONICAL_CREDIT_EXPOSURE_INGEST_DF = pl.DataFrame(
    CANONICAL_CREDIT_EXPOSURE_INGEST_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(
        CANONICAL_CREDIT_EXPOSURE_INGEST_DATA,
        title="Canonical Credit-Exposure Ingest Data (SPOT + netting sets)",
    )

    print_df(
        CANONICAL_CREDIT_EXPOSURE_INGEST_DF,
        title="Canonical Credit-Exposure Ingest Data as DataFrame (SPOT + netting sets)",
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
