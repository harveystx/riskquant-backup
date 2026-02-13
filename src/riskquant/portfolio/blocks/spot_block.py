"""
Compiled, memory-efficient container for spot positions used by the compiled
portfolio runtime.

This module provides `SpotBlock`, a struct-of-arrays representation of a set
of spot trades. The block is intentionally compact (NumPy arrays, primitive
numeric dtypes) so it can be held efficiently in memory and passed between
processes in high-throughput valuation workflows.

Design notes:
  - Row i of every array corresponds to the trade with UID `trade_uids[i]`.
  - Index fields (e.g. `price_uix`, `fx_uix`) are integer indices into the
    `CompiledPortfolio` factor universes. These are not risk-factor IDs but
    zero-based indices used for fast array-based lookup.
  - FX linkage uses a sentinel index (e.g. `-1`) to indicate "no FX conversion",
    keeping the representation numeric and avoiding object arrays.
"""

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True)
class SpotBlock(Block):
    """
    Struct-of-arrays for compiled spot trades.

    Each attribute is a NumPy array with matching length `m_spot` (number of spot trades). The block is frozen and uses slots for memory and performance benefits.

    Row ordering:
      - row i corresponds to trade with UID `trade_uids[i]`.

    Index semantics:
      - `price_uix` is an int32 index into `CompiledPortfolio.price_factor_ids`.
      - `fx_uix` is an int32 index into `CompiledPortfolio.fx_factor_ids` or
        the sentinel value (e.g. -1) indicating no FX conversion is required.
    """

    # ========= TRADE IDENTITY ==========

    # Unique trade identifiers, one per row. dtype: int64 (np.int64)
    trade_uids: np.ndarray

    # ========== ECONOMIC TERMS ==========

    # Signed quantities / notionals for the spot positions. dtype: float64 (np.float64)
    qty: np.ndarray

    # ========== MARKET LINKAGES (UNIVERSE INDICES) ==========

    # Index into the compiled price factor universe for this row. dtype: int32 (np.int32)
    price_uix: np.ndarray

    # Index into the compiled FX factor universe, or sentinel (-1) when no FX needed. dtype: int32 (np.int32)
    fx_uix: np.ndarray
