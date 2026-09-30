# src/riskquant/ingest_fixtures/config/risk_proxy_model/canonical_risk_proxy_model.py
"""
Canonical Proxy V1 authored-term frame (D088 §12.3; D093, PROPOSED).

Small and diagnostic. Against the canonical static configuration it exercises:
  - a multi-term row with signed loadings: EURCHF <- +1 USDCHF - 1 USDEUR
    (the LOG cross identity EURCHF = USDCHF / USDEUR);
  - kappa != tau and a repeated (driver, chart) term that canonicalisation
    merges: AAPL_EQ (LOG) <- 0.3 + 0.2 = 0.5 * Y(USDEUR, DIFFERENCE);
  - a signed-zero term that canonicalisation drops (-0.0 * Y(USDGBP, SIMPLE));
  - a container term expanded per target pillar by EXACT pairing:
    WTI_CRUDE_FWD <- 0.95 * Y(BRENT_CRUDE_FWD pillar, LOG_RETURN).
Every other V1 axis node receives its identity row at compile time.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from riskquant.ingest_fixtures.schemas import RiskProxyModelTermData

CANONICAL_RISK_PROXY_MODEL_DATA: Final[RiskProxyModelTermData] = {
    "term_kind": ["NODE", "NODE", "NODE", "NODE", "NODE", "CONTAINER"],
    "target_name": ["EURCHF", "EURCHF", "AAPL_EQ", "AAPL_EQ", "AAPL_EQ", "WTI_CRUDE_FWD"],
    "driver_name": ["USDCHF", "USDEUR", "USDEUR", "USDEUR", "USDGBP", "BRENT_CRUDE_FWD"],
    "driver_chart": [
        "LOG_RETURN", "LOG_RETURN", "DIFFERENCE", "DIFFERENCE", "SIMPLE_RETURN", "LOG_RETURN",
    ],
    "loading": [1.0, -1.0, 0.3, 0.2, -0.0, 0.95],
}

CANONICAL_RISK_PROXY_MODEL_DF: Final[pl.DataFrame] = pl.DataFrame(
    dict(CANONICAL_RISK_PROXY_MODEL_DATA),
    schema={
        "term_kind": pl.Utf8,
        "target_name": pl.Utf8,
        "driver_name": pl.Utf8,
        "driver_chart": pl.Utf8,
        "loading": pl.Float64,
    },
)
