# src/riskquant/ingest_fixtures/portfolio/spot.py
"""
Canonical SPOT ingest fixture.

11 rows / 11 unique uids. Exercises every resolution path the SPOT
compiler can take against the canonical static + proxy config universe:

Per-row coverage
----------------
  uid  underlying              kind                       notes
  ---  ------------------      -----------------------    ----------------------
   1   AAPL_EQ                 node, no FX                long
   2   AAPL_EQ                 node, no FX                long  (duplicate row
                                                                of uid 1 — same
                                                                underlying,
                                                                identical
                                                                resolution path)
   3   USDEUR                  node, FX direct hit        long
   4   USDEUR                  node, FX direct hit        short (settle ccy
                                                                forces inversion)
   5   USDGBP                  node, RF-LEVEL PROXY       proxies to USDEUR
                                                          scale 0.85
   6   EURCHF                  node, RF-LEVEL PROXY       proxies to USDEUR
                                                          scale 0.60
   7   BRENT_CRUDE_FWD[M+0]    curve-point, direct        long
   8   BRENT_CRUDE_FWD[M+0]    curve-point, direct        short, EUR settle
                                                          (exercises FX leg)
   9   WTI_CRUDE_FWD[M+0]      curve-point, CONTAINER     proxies to
                               PROXY                      BRENT_CRUDE_FWD[M+0]
                                                          scale 0.95
  10   HENRY_HUB_GAS_FWD[M+1]  curve-point, CONTAINER     proxies to
                               PROXY                      BRENT_CRUDE_FWD[M+1]
                                                          scale 1.40
  11   USD_LIBOR_3M[10Y]       curve-point, RF-LEVEL      proxies to
                               PROXY                      USD_SOFR_OIS[10Y]
                                                          scale 1.00

Netting-set linkage is exercised across uids 1-4 (three sets) so the
credit-exposure pipeline has something to chew on without forcing the
SPOT validator into any new branches.
"""

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import (
    generate_netting_set_ids,
    validate_column_lengths,
)
from riskquant.portfolio.instruments import InstrumentType

_UIDS: list[int] = list(range(1, 12))  # uids 1..11, contiguous and unique.
_NUM_UIDS: int = len(_UIDS)


def _get_data() -> MarketRiskIngestData:
    return {
        "uid": _UIDS,
        "instrument": [InstrumentType.SPOT.name] * _NUM_UIDS,
        "quantity": [
            100.0,  # uid 1.  long AAPL
            100.0,  # uid 2.  long AAPL (duplicate underlying)
            5_000_000.0,  # uid 3.  long USDEUR FX spot
            -5_000_000.0,  # uid 4.  short USDEUR (settle USD forces inversion)
            -3_000_000.0,  # uid 5.  short USDGBP (RF-proxy underlying)
            2_000_000.0,  # uid 6.  long  EURCHF (RF-proxy underlying)
            10_000.0,  # uid 7.  long  BRENT front-month
            -5_000.0,  # uid 8.  short BRENT front-month in EUR settle
            8_000.0,  # uid 9.  long  WTI  front-month (container-proxy)
            -2_500.0,  # uid 10. short HENRY_HUB [M+1] (container-proxy)
            1_000_000.0,  # uid 11. long  USD_LIBOR_3M[10Y] (RF-proxy)
        ],
        "underlying": [
            "AAPL_EQ",  # node, no FX
            "AAPL_EQ",  # node, no FX (dup)
            "USDEUR",  # node, direct FX
            "USDEUR",  # node, inverse direct FX
            "USDGBP",  # node, RF-proxy
            "EURCHF",  # node, RF-proxy
            "BRENT_CRUDE_FWD[M+0]",  # curve-point, direct
            "BRENT_CRUDE_FWD[M+0]",  # curve-point, direct (different settle)
            "WTI_CRUDE_FWD[M+0]",  # curve-point, container-proxy
            "HENRY_HUB_GAS_FWD[M+1]",  # curve-point, container-proxy
            "USD_LIBOR_3M[10Y]",  # curve-point, RF-proxy
        ],
        "settlement_ccy": [
            "USD",
            "EUR",
            "EUR",
            "USD",
            "GBP",
            "CHF",
            "USD",
            "EUR",  # forces FX leg over the BRENT[M+0] underlying
            "USD",
            "USD",
            "USD",
        ],
        # netting_set_id: small set used by exposure pipeline. None -> no
        # netting linkage.
        "netting_set_id": generate_netting_set_ids(n=_NUM_UIDS, none_probability=0.5),
        # SPOT-irrelevant columns: all None.
        "strike_price": [None] * _NUM_UIDS,
        "strike_ccy": [None] * _NUM_UIDS,
        "is_call": [None] * _NUM_UIDS,
        "expiry_date": [None] * _NUM_UIDS,
        "delivery_date": [None] * _NUM_UIDS,
        "settlement_date": [None] * _NUM_UIDS,
        "under_discount_curve": [None] * _NUM_UIDS,
        "strike_discount_curve": [None] * _NUM_UIDS,
        "volatility_container_name": [None] * _NUM_UIDS,
    }


CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA)


_CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical SPOT Market-Risk Ingest Data:")
    print_dict(CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
