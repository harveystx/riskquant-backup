"""
Base types for compiled instrument blocks.

This module provides the `Block` marker base class used to type the
struct-of-arrays containers produced by the portfolio compiler and consumed
by valuation and simulation kernels. Blocks are intentionally minimal and
carry only compact numeric NumPy arrays and integer indices into compiled
factor universes.

The class exists primarily for typing and to provide a shared superclass
for all compiled block types (spot, forward, option, etc.).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Block:
    """
    Marker base class for compiled instrument blocks.

    Implementations (concrete blocks) should expose only NumPy arrays and
    primitive numeric dtypes to remain efficient in memory and serialisation.
    The dataclass is frozen and uses slots to reduce per-instance overhead.
    """

    # Intentionally empty; subclasses define typed NumPy array fields.
    pass
