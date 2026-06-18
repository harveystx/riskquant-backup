# src/riskquant/ingest_fixtures/portfolio/vanilla_irs.py
"""
Canonical VANILLA_IRS ingest fixture.
"""

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import (
    generate_netting_set_ids,
    validate_column_lengths,
)
from riskquant.portfolio.instruments import InstrumentType


def _get_data() -> MarketRiskIngestData:
    return {
        "uid": [500, 501, 502, 503, 504, 505],
        "instrument": [InstrumentType.VANILLA_IRS.name] * 6,
        "notional": [
            +10_000_000.0,
            -15_000_000.0,
            -25_000_000.0,
            +18_000_000.0,
            +15_000_000.0,
            +13_000_000.0,
        ],
        "pay_receive_fixed": [
            "RECEIVE_FIXED",
            "PAY_FIXED",
            "RECEIVE_FIXED",
            "PAY_FIXED",
            "RECEIVE_FIXED",
            "PAY_FIXED",
        ],
        "settlement_ccy": [
            "USD",
            "EUR",
            "GBP",
            "USD",
            "EUR",
            "USD",
        ],
        "fixed_rate": [
            0.0450,
            0.0320,
            0.0550,
            0.0400,
            0.0280,
            0.0500,
        ],
        "float_spread": [
            +0.0000,
            +0.0015,
            -0.0000,
            +0.0025,
            -0.0010,
            +0.0000,
        ],
        "effective_date": [
            "2026-06-15",
            "2026-06-15",
            "2026-06-15",
            "2026-06-15",
            "2026-06-15",
            "2026-06-15",
        ],
        "maturity_date": [
            "2036-06-15",
            "2028-06-15",
            "2031-06-15",
            "2027-06-15",
            "2033-06-15",
            "2041-06-15",
        ],
        "fixed_freq": [
            "6M",
            "1Y",
            "1Y",
            "3M",
            "6M",
            "1Y",
        ],
        "float_freq": [
            "3M",
            "6M",
            "3M",
            "1M",
            "3M",
            "6M",
        ],
        "fixed_daycount": [
            "30_360",
            "30_360",
            "ACT_365F",
            "30_360",
            "30_360",
            "ACT_365F",
        ],
        "float_daycount": [
            "ACT_360",
            "ACT_360",
            "ACT_365F",
            "ACT_360",
            "ACT_360",
            "ACT_360",
        ],
        "forecast_curve": [
            "USD_3M",
            "EUR_6M",
            "GBP_3M",
            "USD_1M",
            "EUR_3M",
            "USD_6M",
        ],
        "discount_curve": [
            "USD_OIS",
            "EUR_OIS",
            "GBP_OIS",
            "USD_OIS",
            "EUR_OIS",
            "USD_OIS",
        ],
        "fixed_holiday_calendar": [None] * 6,
        "float_holiday_calendar": [None] * 6,
        "fixed_business_day_convention": [
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
            "FOLLOWING",
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
        ],
        "float_business_day_convention": [
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
            "FOLLOWING",
            "MODIFIED_FOLLOWING",
            "MODIFIED_FOLLOWING",
        ],
        "date_generation_rule": [
            "BACKWARD",
            "BACKWARD",
            "BACKWARD",
            "BACKWARD",
            "BACKWARD",
            "BACKWARD",
        ],
        "payment_lag": [
            2,
            2,
            0,
            2,
            2,
            2,
        ],
        "fixing_lag": [
            2,
            2,
            0,
            2,
            2,
            2,
        ],
        "netting_set_id": generate_netting_set_ids(
            n=6,
            none_probability=0.0,
        ),
    }


CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA)


_CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical VANILLA_IRS Market-Risk Ingest Data:")
    print_dict(CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
