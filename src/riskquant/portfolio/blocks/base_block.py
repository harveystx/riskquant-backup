# riskquant/portfolio/blocks/base_block.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Block:
    """
    Marker base class for compiled instrument blocks.

    Blocks are the compiled (struct-of-arrays) representation produced by the
    portfolio compiler and consumed by valuation / simulation kernels.

    A Block contains only compact numeric arrays (no Python objects) and
    references CompiledPortfolio factor universes by integer indices.
    """

    pass
