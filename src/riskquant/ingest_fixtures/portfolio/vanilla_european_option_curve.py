# src/riskquant/ingest_fixtures/portfolio/vanilla_european_option_curve.py
"""
Canonical VANILLA_EUROPEAN_OPTION_CURVE ingest fixture.

VANILLA_EUROPEAN_OPTION_CURVE = European option on a forward curve.
Underlying is a curve container; vol is sourced from a surface container;
delivery + expiry + strike + is_call + vol-container all live.

Coverage:
  - Call on commodity curve, no FX             -> BRENT_CRUDE_FWD
                                                  vol surface
                                                  BRENT_CRUDE_VOL_SURF (uid=400)
  - Put  on commodity curve, no FX             -> BRENT_CRUDE_FWD
                                                  BRENT_CRUDE_VOL_SURF (uid=401)
  - Call on commodity curve, FX settlement     -> BRENT_CRUDE_FWD
                                                  EUR settlement
                                                  BRENT_CRUDE_VOL_SURF (uid=402)
  - Call on commodity curve, CONTAINER PROXY   -> WTI_CRUDE_FWD ->
                                                  BRENT_CRUDE_FWD, scale 0.95;
                                                  WTI vol surface is itself
                                                  resolved via the underlying-
                                                  container proxy on the vol
                                                  side at the runtime builder.
                                                  Here we point at
                                                  BRENT_CRUDE_VOL_SURF
                                                  directly. (uid=403)
  - Put  on FX curve via surface (uid=404)     -> USDEUR; vol surface
                                                  USDEUR_VOL_SURF is
                                                  declared DELTA_CALL — the
                                                  runtime market-data builder
                                                  currently rejects DELTA
                                                  surfaces, so this trade
                                                  compiles cleanly but cannot
                                                  yet be priced end-to-end.
                                                  See notes/REFACTORS_NEEDED.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.ingest_fixtures.rng import get_canonical_rng
from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import (
    generate_netting_set_ids,
    validate_column_lengths,
)
from riskquant.portfolio.instruments import Instrument

_rng: np.random.Generator = get_canonical_rng()


def _get_data() -> MarketRiskIngestData:
    return {
        "uid": [400, 401, 402, 403, 404],
        "instrument": [Instrument.VANILLA_EUROPEAN_OPTION_CURVE.name] * 5,
        "quantity": [
            2_000.0,  # 400. long call on Brent fwd
            -1_500.0,  # 401. short put on Brent fwd
            1_000.0,  # 402. long call on Brent fwd, EUR settle
            3_000.0,  # 403. long call on WTI fwd (container-proxy)
            -500.0,  # 404. short put on USDEUR fwd (vol surface USDEUR_VOL_SURF)
        ],
        "underlying": [
            "BRENT_CRUDE_FWD",
            "BRENT_CRUDE_FWD",
            "BRENT_CRUDE_FWD",
            "WTI_CRUDE_FWD",  # container-proxy
            "USDEUR",
        ],
        "settlement_ccy": [
            "USD",
            "USD",
            "EUR",
            "USD",
            "EUR",
        ],
        "strike_price": [80.0, 75.0, 82.0, 73.0, 1.0900],
        "strike_ccy": [
            "USD",
            "USD",
            "EUR",
            "USD",
            "EUR",
        ],
        "is_call": [1.0, 0.0, 1.0, 1.0, 0.0],
        "expiry_date": [
            "2026-06-25",
            "2026-09-24",
            "2026-12-24",
            "2026-06-25",
            "2026-09-24",
        ],
        "delivery_date": [
            "2026-07-01",
            "2026-10-01",
            "2027-01-01",
            "2026-07-01",
            None,  # FX option has no underlying delivery
        ],
        "settlement_date": [
            "2026-06-29",
            "2026-09-28",
            "2026-12-28",
            "2026-06-29",
            "2026-09-28",
        ],
        "under_discount_curve": [
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
        ],
        "strike_discount_curve": [
            "USD_SOFR_OIS",
            "USD_SOFR_OIS",
            "EUR_ESTR_OIS",  # cross-curve discount
            "USD_SOFR_OIS",
            "EUR_ESTR_OIS",
        ],
        # Vol-container resolution lands at the ingest perimeter now: the
        # compile step maps each name to an int32 container id via
        # `static_config.container_name_to_id` and derives the int8 y-axis
        # type via `surface_meta_dict[id].y_axis_kind`. uid=404 references
        # a DELTA_CALL surface (USDEUR_VOL_SURF) — that ingests fine but
        # the runtime builder rejects DELTA surfaces today (see
        # notes/REFACTORS_NEEDED.md).
        "volatility_container_name": [
            "BRENT_CRUDE_VOL_SURF",
            "BRENT_CRUDE_VOL_SURF",
            "BRENT_CRUDE_VOL_SURF",
            "BRENT_CRUDE_VOL_SURF",  # WTI vol surface not in canonical config
            "USDEUR_VOL_SURF",
        ],
        "netting_set_id": generate_netting_set_ids(n=5, none_probability=0.0),
    }


CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(
    CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA
)


_CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical VANILLA_EUROPEAN_OPTION_CURVE Market-Risk Ingest Data:")
    print_dict(CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA)

    print("As DataFrame:")
    print_df(
        _CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DF,
        index=False,
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
