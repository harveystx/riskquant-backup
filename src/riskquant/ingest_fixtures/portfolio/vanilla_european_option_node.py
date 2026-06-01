# src/riskquant/ingest_fixtures/portfolio/vanilla_european_option_node.py
"""
Canonical VANILLA_EUROPEAN_OPTION_NODE ingest fixture.

VANILLA_EUROPEAN_OPTION_NODE = European option whose underlying resolves
to a single risk-factor node (or curve-point relative name). Vol is
sourced from a vol container — conceptually a vol-NODE, but the
canonical static config does not currently declare any vol-node
containers; only vol surfaces exist. This fixture therefore points
`volatility_container_name` at the parent vol SURFACE container, which
ingests cleanly (the schema accepts any container name) but exposes the
documented option-node-vol-source architectural gap: the kernel reads
`vol_nodes[v_uix]` and the slot reserved for the surface id will not
have a populated scalar-vol row at runtime.

The new hoisted `if vol_uix >= 0:` sentinel branch in the kernel makes
this SAFE (no out-of-bounds UB) but not CORRECT — see
notes/REFACTORS_NEEDED.md for the three follow-up options.

Coverage (resolution paths, not end-to-end PV):
  - Call on equity node, no FX                 -> AAPL_EQ      (uid=300)
                                                  vol AAPL_VOL_SURF
  - Put  on equity node, no FX                 -> AAPL_EQ      (uid=301)
                                                  vol AAPL_VOL_SURF
  - Call on FX node, settlement in foreign ccy -> USDEUR       (uid=302)
                                                  strike=EUR, settle=EUR
                                                  vol USDEUR_VOL_SURF
                                                  (DELTA_CALL — runtime
                                                  builder rejects today)
  - Put  on FX node, RF-LEVEL PROXY underlying -> USDGBP       (uid=303)
                                                  proxies to USDEUR
                                                  vol USDEUR_VOL_SURF
                                                  (DELTA_CALL — same)
  - Call on curve-point pinned underlying      -> BRENT_CRUDE_FWD[M+0]
                                                  (uid=304)
                                                  vol BRENT_CRUDE_VOL_SURF
  - Put  on curve-point, CONTAINER PROXY       -> WTI_CRUDE_FWD[M+0]
                                                  proxies to BRENT[M+0]
                                                  (uid=305)
                                                  vol BRENT_CRUDE_VOL_SURF
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
        "uid": [300, 301, 302, 303, 304, 305],
        "instrument": [Instrument.VANILLA_EUROPEAN_OPTION_NODE.name] * 6,
        "quantity": [
            100.0,  # 300. long call AAPL
            -100.0,  # 301. short put AAPL
            500_000.0,  # 302. long call USDEUR
            -250_000.0,  # 303. short put USDGBP (RF-proxy)
            1_000.0,  # 304. long call BRENT[M+0]
            -500.0,  # 305. short put WTI[M+0] (container-proxy)
        ],
        "underlying": [
            "AAPL_EQ",
            "AAPL_EQ",
            "USDEUR",
            "USDGBP",
            "BRENT_CRUDE_FWD[M+0]",
            "WTI_CRUDE_FWD[M+0]",
        ],
        "settlement_ccy": [
            "USD",
            "USD",
            "EUR",
            "GBP",
            "USD",
            "USD",
        ],
        "strike_price": [180.0, 175.0, 1.0850, 1.2750, 78.0, 75.0],
        "strike_ccy": [
            "USD",
            "USD",
            "EUR",
            "GBP",
            "USD",
            "USD",
        ],
        # is_call: 1.0 = call, 0.0 = put. Mixed to exercise both legs of
        # the branchless cp_mult encoding.
        "is_call": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "expiry_date": [
            "2026-06-19",
            "2026-09-18",
            "2026-12-18",
            "2027-03-19",
            "2026-06-26",
            "2026-09-25",
        ],
        "settlement_date": [
            "2026-06-22",
            "2026-09-21",
            "2026-12-21",
            "2027-03-22",
            "2026-06-29",
            "2026-09-28",
        ],
        "under_discount_curve": [
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
        ],
        "strike_discount_curve": [
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "EUR_ESTR_OIS",
            "GBP_SONIA_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
        ],
        # OPTION_NODE has no delivery (single rf underlying).
        "delivery_date": [None] * 6,
        # Vol-container pointer. The option-node kernel reads
        # `vol_nodes[uix]` — a separate axis from vol_surfaces — and the
        # canonical static config has no vol-node entries. Until a vol-
        # node universe lands (or the option-node kernel is refactored
        # to source from `vol_surfaces`), this fixture points at the
        # parent vol surface so ingest validation passes; the runtime
        # PV will be zero or garbage at the surface's universe slot.
        # See notes/REFACTORS_NEEDED.md.
        "volatility_container_name": [
            "AAPL_VOL_SURF",
            "AAPL_VOL_SURF",
            "USDEUR_VOL_SURF",
            "USDEUR_VOL_SURF",  # USDGBP proxies to USDEUR
            "BRENT_CRUDE_VOL_SURF",
            "BRENT_CRUDE_VOL_SURF",  # WTI proxies to BRENT
        ],
        "netting_set_id": generate_netting_set_ids(n=6, none_probability=0.0),
    }


CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(
    CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA
)


_CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical VANILLA_EUROPEAN_OPTION_NODE Market-Risk Ingest Data:")
    print_dict(CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
