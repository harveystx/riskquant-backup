# src/riskquant/ingest_fixtures/portfolio/vanilla_forward_curve.py
"""
Canonical VANILLA_FORWARD_CURVE ingest fixture.

VANILLA_FORWARD_CURVE = forward whose underlying is a *curve container*
(not a single rf). Delivery date selects the curve point at runtime via
the curve's roll rule. Strike + delivery + settlement + discounting are
all live.

Coverage:
  - Curve underlying, no FX, no discount       -> BRENT_CRUDE_FWD
                                                  (uid=200)
  - Curve underlying, no FX, with discount     -> BRENT_CRUDE_FWD
                                                  + USD_SOFR_OIS
                                                  (uid=201)
  - Curve underlying, FX, cross-curve discount -> BRENT_CRUDE_FWD
                                                  EUR settlement,
                                                  USD_SOFR_OIS / EUR_ESTR_OIS
                                                  (uid=202)
  - Curve underlying, CONTAINER PROXY          -> WTI_CRUDE_FWD ->
                                                  BRENT_CRUDE_FWD,
                                                  scale 0.95
                                                  (uid=203)
  - Curve underlying, CONTAINER PROXY + FX     -> HENRY_HUB_GAS_FWD ->
                                                  BRENT_CRUDE_FWD,
                                                  EUR settlement,
                                                  scale 1.40 (uid=204)
  - Curve underlying, IBOR discount-basis path -> BRENT_CRUDE_FWD
                                                  discounted via
                                                  USD_LIBOR_3M instead
                                                  of OIS — exercises the
                                                  non-default discount
                                                  basis (uid=205)

Not covered here (registered proxy that is not exercisable today):
  POWER_CH_BASELOAD_HY -> CO2_EUA_ANNUAL_FWD
    Source curve is OFFSET_HY, proxy is OFFSET_Y. The resolver's
    container-proxy path requires matching x_axis_kinds across required
    and proxy (PrettyKeyError otherwise). Resolving x_axis_kind mismatch
    is a separate work item; until then this proxy is registered but
    not consumed at ingest.
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
        "uid": [200, 201, 202, 203, 204, 205],
        "instrument": [Instrument.VANILLA_FORWARD_CURVE.name] * 6,
        "quantity": [
            5_000.0,  # 200. long BRENT fwd curve
            -2_500.0,  # 201. short BRENT fwd, discounted
            3_000.0,  # 202. long BRENT fwd, EUR settle
            7_500.0,  # 203. long WTI fwd (container-proxy)
            -1_500.0,  # 204. short HENRY_HUB (container-proxy + FX)
            4_000.0,  # 205. long BRENT, LIBOR discount basis
        ],
        "underlying": [
            "BRENT_CRUDE_FWD",
            "BRENT_CRUDE_FWD",
            "BRENT_CRUDE_FWD",
            "WTI_CRUDE_FWD",  # container-proxy
            "HENRY_HUB_GAS_FWD",  # container-proxy
            "BRENT_CRUDE_FWD",
        ],
        "settlement_ccy": [
            "USD",
            "USD",
            "EUR",
            "USD",
            "EUR",
            "USD",
        ],
        "strike_price": [78.0, 79.5, 80.0, 70.0, 3.50, 82.0],
        "strike_ccy": [
            "USD",
            "USD",
            "EUR",
            "USD",
            "EUR",
            "USD",
        ],
        "delivery_date": [
            "2026-06-15",  # 200.
            "2026-09-15",  # 201.
            "2026-12-15",  # 202.
            "2026-06-15",  # 203.
            "2026-09-15",  # 204.
            "2027-03-15",  # 205.
        ],
        "settlement_date": [
            None,
            "2026-09-17",
            "2026-12-17",
            None,
            "2026-09-17",
            "2027-03-17",
        ],
        "under_discount_curve": [
            None,
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            None,
            "USD_SOFR_OIS",
            "USD_LIBOR_3M",  # non-default basis for USD
        ],
        "strike_discount_curve": [
            None,
            "USD_SOFR_OIS",
            "EUR_ESTR_OIS",  # cross-curve
            None,
            "EUR_ESTR_OIS",
            "USD_LIBOR_3M",
        ],
        "is_call": [None] * 6,
        "expiry_date": [None] * 6,
        "netting_set_id": generate_netting_set_ids(n=6, none_probability=0.0),
    }


CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = (
    _get_data()
)

validate_column_lengths(CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA)


_CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical VANILLA_FORWARD_CURVE Market-Risk Ingest Data:")
    print_dict(CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
