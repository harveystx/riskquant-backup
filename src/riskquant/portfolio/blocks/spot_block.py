# riskquant/portfolio/blocks/spot_block.py
from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True)
class SpotBlock(Block):
    """
    Compiled spot trades (struct-of-arrays).

    Row order is defined by trade_uids: row i corresponds to trade_uids[i].

    Indices refer to CompiledPortfolio factor universes:
      - price_uix indexes CompiledPortfolio.price_factor_ids
      - fx_uix indexes CompiledPortfolio.fx_factor_ids (or -1 meaning "no FX conversion")
    """

    trade_uids: np.ndarray  # (m,) int64
    qty: np.ndarray  # (m,) float64
    price_uix: np.ndarray  # (m,) int32
    fx_uix: np.ndarray  # (m,) int32 (-1 => FX_NONE)
