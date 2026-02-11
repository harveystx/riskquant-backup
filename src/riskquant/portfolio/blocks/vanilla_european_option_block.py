# riskquant/portfolio/blocks/vanilla_european_option_block.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True)
class VanillaEuropeanOptionBlock(Block):
    """
    Compiled block for plain-vanilla European option trades (struct-of-arrays).

    This is the compiled (array-of-structs -> struct-of-arrays) representation used by
    fast valuation / simulation kernels.

    Row order is defined by trade_uids: row i corresponds to trade_uids[i].

    Indices refer to CompiledPortfolio factor universes:
      - price_uix indexes CompiledPortfolio.price_factor_ids
      - vol_uix indexes CompiledPortfolio.vol_factor_ids (pricing input; shockable only if you choose)
      - fx_*_uix index CompiledPortfolio.fx_factor_ids (or -1 meaning "no FX conversion")

    Notes
    -----
    - expiry_date_ordinal32 is the contractual option expiry date expressed as int32 days
      since riskquant DATE_EPOCH (1970-01-01).
      If expiry_date_ordinal32 == NO_EXPIRY_DATE, the pricer must interpret the expiry
      via another convention (e.g., instrument-specific default) or raise.
    - This block is intentionally model-neutral: it does not embed pricing model choice.
    """

    # --- TRADE IDENTITY --- #
    trade_uids: np.ndarray  # (m_opt,) int64

    # --- ECONOMIC TERMS --- #
    qty: np.ndarray  # (m_opt,) float64
    strike: np.ndarray  # (m_opt,) float64
    is_call: np.ndarray  # (m_opt,) int8  (1=call, 0=put)

    # --- MARKET LINKAGES (UNIVERSE INDICES) --- #
    price_uix: np.ndarray  # (m_opt,) int32  -> CompiledPortfolio.price_factor_ids
    vol_uix: np.ndarray  # (m_opt,) int32    -> CompiledPortfolio.vol_factor_ids

    # --- EXPIRY DATE --- #
    expiry_date_ordinal32: np.ndarray  # (m_opt,) int32 (NO_EXPIRY_DATE => pricer-defined)

    # --- OPTIONAL FX LINKAGES (UNIVERSE INDICES) --- #
    fx_under_to_settle_uix: np.ndarray  # (m_opt,) int32  -> CompiledPortfolio.fx_factor_ids or -1
    fx_strike_to_settle_uix: np.ndarray  # (m_opt,) int32  -> CompiledPortfolio.fx_factor_ids or -1
