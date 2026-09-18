# tests/market_data/remediation/md_oracle.py
"""
Independent market-data economic oracle (test-only).

Authority: economic names and quotes only. Imports math/dataclasses/numpy. It
never imports riskquant, never sees an id, a uix, a universe, a pair table, a
fold, or a pricer. Its conventions are the ECONOMIC contracts:

    FX quote  "BASEQUOTE" = X  means 1 BASE buys X QUOTE.
              FX(a -> s) = X if (a, s) quoted, 1/X if (s, a) quoted,
              1 if a == s, otherwise the quote is MISSING and the oracle raises.
    Spot      PV = Q * S * FX(ccy(S) -> settle); ESP = S * FX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MissingQuoteError(KeyError):
    """The economic market has no FX quote connecting two currencies."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class EconomicMarket:
    levels: dict[str, np.ndarray]  # factor name -> (S,) level
    factor_ccy: dict[str, str]  # factor name -> quote currency
    fx_quotes: dict[tuple[str, str], np.ndarray]  # (base, quote) -> (S,) level


def fx(market: EconomicMarket, a: str, s: str, n: int) -> np.ndarray:
    if a == s:
        return np.ones(n)
    if (a, s) in market.fx_quotes:
        return market.fx_quotes[(a, s)]
    if (s, a) in market.fx_quotes:
        return 1.0 / market.fx_quotes[(s, a)]
    raise MissingQuoteError(f"no FX quote between {a} and {s}")


def effective_spot(
    market: EconomicMarket, underlying: str, settle: str, n: int
) -> np.ndarray:
    return market.levels[underlying] * fx(
        market, market.factor_ccy[underlying], settle, n
    )


def spot_pv(
    market: EconomicMarket, *, qty: float, underlying: str, settle: str, n: int
) -> np.ndarray:
    return qty * effective_spot(market, underlying, settle, n)
