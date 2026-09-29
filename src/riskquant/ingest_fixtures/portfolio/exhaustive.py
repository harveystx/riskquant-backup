# src/riskquant/ingest_fixtures/portfolio/exhaustive.py
"""
Aggregated canonical portfolio ingest fixture.

Concatenates all per-instrument fixtures into one DataFrame for full
end-to-end ingest-pipeline coverage:

  - SPOT                           (uid 1..11)    -> spot.py
  - VANILLA_FORWARD_NODE           (uid 100..105) -> vanilla_forward_node.py
  - VANILLA_FORWARD_CURVE          (uid 200..205) -> vanilla_forward_curve.py
  - VANILLA_EUROPEAN_OPTION_NODE   (uid 300..305) -> vanilla_european_option_node.py
  - VANILLA_EUROPEAN_OPTION_CURVE  (uid 400..404) -> vanilla_european_option_curve.py

uid partition is contiguous-per-instrument so a downstream test can
filter by instrument without re-reading the source files. Total rows:
11 + 6 + 6 + 6 + 5 = 34.

Coverage of resolution paths (against the canonical static + proxy
config fixtures):

  - Node and curve-point underlyings
  - Underlyings that a market history may not carry (USDGBP, EURCHF,
    USD_LIBOR_3M[10Y], WTI_CRUDE_FWD, HENRY_HUB_GAS_FWD): a scenario run
    needs a world column or an authored Proxy V1 term for each (D088)
  - FX paths            (direct USDEUR; inverse settle currency;
                          USDGBP/EURCHF underlyings)
  - Discount basis      (OIS default vs explicit IBOR override)
  - Cross-curve discount (USD_SOFR_OIS / EUR_ESTR_OIS; USD_SOFR_OIS /
                          GBP_SONIA_OIS)
  - Netting linkage     (spot fixture; sets 101, 102, 103)
  - Option payoff sign  (calls and puts both present)
  - Date semantics      (expiry + delivery + settlement combinations)

"""

from __future__ import annotations

from typing import cast

import pandas as pd

from riskquant.ingest_fixtures.portfolio.spot import (
    CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_european_option_curve import (
    CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_european_option_node import (
    CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_forward_curve import (
    CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_forward_node import (
    CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_irs import (
    CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.schemas import MarketRiskIngestData
from riskquant.ingest_fixtures.utils import validate_column_lengths


def _get_data() -> MarketRiskIngestData:
    # Concat via DataFrame -> dict round-trip so column unions land
    # cleanly (per-fixture columns are a subset of the full schema).
    per_instrument_dfs = [
        pd.DataFrame(CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA),
        pd.DataFrame(CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA),
        pd.DataFrame(CANONICAL_VANILLA_FORWARD_CURVE_MARKET_RISK_PORTFOLIO_DATA),
        pd.DataFrame(CANONICAL_VANILLA_EUROPEAN_OPTION_NODE_MARKET_RISK_PORTFOLIO_DATA),
        pd.DataFrame(
            CANONICAL_VANILLA_EUROPEAN_OPTION_CURVE_MARKET_RISK_PORTFOLIO_DATA
        ),
        pd.DataFrame(CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA),
    ]

    # Drop entirely-NA columns *before* concat so a missing column on
    # one input doesn't bleed object dtype into the others.
    valid_dfs = [
        df.dropna(axis=1, how="all") for df in per_instrument_dfs if not df.empty
    ]

    assert valid_dfs, "No valid per-instrument DataFrames to concat"

    combined_df = (
        pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()
    )

    assert not combined_df.empty, "Combined DataFrame is empty after concat"

    assert combined_df["uid"].is_unique, "Combined DataFrame has duplicate 'uid' values"

    return cast(MarketRiskIngestData, combined_df.to_dict(orient="list"))


CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA: MarketRiskIngestData = _get_data()

validate_column_lengths(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA)


_CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DF = pd.DataFrame(
    CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    n_rows = len(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA["uid"])
    title = f"Canonical Exhaustive Market-Risk Ingest Data ({n_rows} rows)"
    print_dict(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA, title=title)

    title += " (as DataFrame)"
    print_df(
        _CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DF,
        title=title,
        float_format="_.2f",
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
