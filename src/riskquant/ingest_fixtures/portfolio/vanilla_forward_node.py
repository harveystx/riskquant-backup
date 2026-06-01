# src/riskquant/ingest_fixtures/portfolio/vanilla_forward_node.py
"""
Canonical VANILLA_FORWARD_NODE ingest fixture.

VANILLA_FORWARD_NODE = forward contract whose underlying resolves to a
single risk-factor node (or a single curve-point relative name). No
vol. Strike, delivery, settlement, and optional cross-curve discounting.

Coverage:
  - Node underlying, no FX, no discount         -> AAPL_EQ      (uid=100)
  - Node underlying, FX, no discount            -> USDEUR       (uid=101)
  - Node underlying, no FX, settlement date     -> AAPL_EQ      (uid=102)
                                                   discount via USD_SOFR_OIS
  - Node underlying, FX, discount + cross-ccy   -> USDEUR       (uid=103)
                                                   strike in EUR, under in USD;
                                                   under_df=USD_SOFR_OIS,
                                                   strike_df=EUR_ESTR_OIS
  - Curve-point underlying (pin to M+0 tenor)   -> BRENT_CRUDE_FWD[M+0]
                                                   (uid=104, no FX)
  - Curve-point underlying, RF-LEVEL PROXY      -> USDGBP -> USDEUR
                                                   (uid=105, scale 0.85)
"""

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import (
    generate_netting_set_ids,
    validate_column_lengths,
)
from riskquant.portfolio.instruments import Instrument


def _get_data() -> MarketRiskIngestData:
    return {
        "uid": [100, 101, 102, 103, 104, 105],
        "instrument": [Instrument.VANILLA_FORWARD_NODE.name] * 6,
        "quantity": [
            10.0,  # 100. long AAPL fwd
            1_000_000.0,  # 101. long USDEUR fwd
            -5.0,  # 102. short AAPL fwd settle
            500_000.0,  # 103. long USDEUR fwd, EUR strike, discounted
            1_000.0,  # 104. long BRENT[M+0] fwd
            -750_000.0,  # 105. short USDGBP fwd (RF-proxy)
        ],
        "underlying": [
            "AAPL_EQ",
            "USDEUR",
            "AAPL_EQ",
            "USDEUR",
            "BRENT_CRUDE_FWD[M+0]",
            "USDGBP",
        ],
        "settlement_ccy": [
            "USD",
            "EUR",
            "USD",
            "EUR",  # strike_ccy=EUR forces cross-curve discounting below
            "USD",
            "GBP",  # forces FX leg + RF-proxy on USDGBP underlying
        ],
        "strike_price": [180.0, 1.0850, 175.0, 1.0900, 78.50, 1.2750],
        "strike_ccy": [
            "USD",
            "EUR",
            "USD",
            "EUR",
            "USD",
            "GBP",
        ],
        "delivery_date": [
            None,  # 100. no delivery (node fwd, no physical schedule)
            None,  # 101. no delivery
            None,  # 102. no delivery
            None,  # 103. no delivery
            None,  # 104. delivery semantics live on the curve via roll rule
            None,  # 105. no delivery
        ],
        "settlement_date": [
            None,  # 100. spot-style (no settlement schedule needed)
            None,  # 101.
            "2026-12-15",  # 102. EXERCISES SETTLEMENT DATE
            "2026-12-15",  # 103. EXERCISES SETTLEMENT DATE + cross-curve discount
            None,  # 104.
            "2026-09-30",  # 105. EXERCISES SETTLEMENT + RF-proxy
        ],
        "under_discount_curve": [
            None,
            None,
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            None,
            "USD_SOFR_OIS",  # under is USD-quoted FX rate
        ],
        "strike_discount_curve": [
            None,
            None,
            "USD_SOFR_OIS",
            "EUR_ESTR_OIS",  # cross-curve discount
            None,
            "GBP_SONIA_OIS",
        ],
        # FWD has no expiry / is_call / vol.
        "is_call": [None] * 6,
        "expiry_date": [None] * 6,
        # netting linkage: leave empty here; spot fixture carries it.
        "netting_set_id": generate_netting_set_ids(n=6, none_probability=0.0),
    }


CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = (
    _get_data()
)

validate_column_lengths(CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA)


_CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical VANILLA_FORWARD_NODE Market-Risk Ingest Data:")
    print_dict(CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
