# riskquant/portfolio/blocks/forward_block.py
from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True)
class ForwardBlock(Block):
    """
    Compiled forward trades (struct-of-arrays).

    Row order is defined by trade_uids: row i corresponds to trade_uids[i].

    Indices refer to CompiledPortfolio factor universes:
      - price_uix indexes CompiledPortfolio.price_factor_ids
      - fx_*_uix index CompiledPortfolio.fx_factor_ids (or -1 meaning "no FX conversion")

    Notes
    -----
    - settlement_date_ordinal32 is the contractual settlement date expressed as int32
      days since riskquant DATE_EPOCH (1970-01-01).
      It is used for discounting only. If settlement_date_ordinal32 == NO_SETTLEMENT_DATE,
      discounting is disabled and the discount factor is assumed to be 1.0.
    - Economic exposure timing is defined by price_uix (forward curve point), not by
      settlement_date_ordinal32.
    """

    # --- TRADE IDENTITY --- #
    trade_uids: np.ndarray  # (m,) int64

    # --- ECONOMIC TERMS --- #
    qty: np.ndarray  # (m,) float64
    strike: np.ndarray  # (m,) float64

    # --- PAYMENT / DISCOUNTING ANCHOR --- #
    settlement_date_ordinal32: np.ndarray  # (m,) int32 (NO_SETTLEMENT_DATE => no discounting)

    # --- MARKET LINKAGES (UNIVERSE INDICES) --- #
    price_uix: np.ndarray  # (m,) int32  -> CompiledPortfolio.price_factor_ids
    fx_under_to_settle_uix: np.ndarray  # (m,) int32  -> CompiledPortfolio.fx_factor_ids or -1
    fx_strike_to_settle_uix: np.ndarray  # (m,) int32  -> CompiledPortfolio.fx_factor_ids or -1
