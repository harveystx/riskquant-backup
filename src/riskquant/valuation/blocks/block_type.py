# src/riskquant/valuation/blocks/block_type.py
"""
Compiled valuation primitives (Block Types).

These are the executable, structure-of-arrays (SoA) blocks consumed
by the valuation engine.  They are **not** economic instruments.

Examples
--------
VANILLA_IRS
    ├── FIXED_CASHFLOW
    └── FLOATING_CASHFLOW

FIXED_RATE_BOND
    ├── FIXED_CASHFLOW
    └── PRINCIPAL_CASHFLOW

FRN
    ├── FLOATING_CASHFLOW
    └── PRINCIPAL_CASHFLOW
"""

from enum import IntEnum
from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# Sentinel value for an explicitly unset / missing block type ID.
# ---------------------------------------------------------------------------
UNSET_BLOCK_TYPE_ID_SENTINEL: Final[int] = -999


class BlockType(IntEnum):
    """
    Executable SoA block primitives.

    Each member carries a stable integer ID used for serialisation,
    dispatch tables, and AoS ↔ SoA conversion.
    """

    # ======================================================================
    # MARKET RISK PRIMITIVES
    # ======================================================================
    SPOT = 0

    VANILLA_FORWARD = 1

    VANILLA_EUROPEAN_OPTION = 2

    # ======================================================================
    # INTEREST-RATE CASHFLOW PRIMITIVES
    # ======================================================================
    FIXED_CASHFLOW = 3
    FLOATING_CASHFLOW = 4
    PRINCIPAL_CASHFLOW = 5

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_id(cls, id_: int) -> "BlockType":
        """Return the member whose ``value`` equals *id_*."""
        if id_ not in BLOCK_TYPE_ID_TO_NAME:  # pragma: no cover
            raise ValueError(f"Invalid BlockType id: {id_}")
        # IntEnum allows instantiation directly from the integer value.
        return cls(id_)

    @classmethod
    def from_name(cls, name: str) -> "BlockType":
        """Return the member whose ``name`` equals *name* (case-sensitive)."""
        if name not in BLOCK_TYPE_NAME_TO_ID:  # pragma: no cover
            raise ValueError(f"Invalid BlockType name: {name!r}")
        return cls[name]


# ==========================================================================
# CONVENIENCE ID CONSTANTS FOR DISPATCH / LOOKUP
# ==========================================================================

SPOT_BLOCK_TYPE_ID: int = BlockType.SPOT.value
VANILLA_FORWARD_BLOCK_TYPE_ID: int = BlockType.VANILLA_FORWARD.value
VANILLA_EUROPEAN_OPTION_BLOCK_TYPE_ID: int = BlockType.VANILLA_EUROPEAN_OPTION.value
FIXED_CASHFLOW_BLOCK_TYPE_ID: int = BlockType.FIXED_CASHFLOW.value
FLOATING_CASHFLOW_BLOCK_TYPE_ID: int = BlockType.FLOATING_CASHFLOW.value
PRINCIPAL_CASHFLOW_BLOCK_TYPE_ID: int = BlockType.PRINCIPAL_CASHFLOW.value

# ---------------------------------------------------------------------------
# Immutable lookup tables (single source of truth).
# ---------------------------------------------------------------------------

NUM_BLOCK_TYPES: Final[int] = len(BlockType)

BLOCK_TYPE_NAME_TO_ID: Final[MappingProxyType[str, int]] = MappingProxyType(
    {member.name: member.value for member in BlockType}
)

BLOCK_TYPE_ID_TO_NAME: Final[MappingProxyType[int, str]] = MappingProxyType(
    {member.value: member.name for member in BlockType}
)

# Frozen sets for fast membership tests / validation.
VALID_BLOCK_TYPE_IDS: Final[frozenset[int]] = frozenset(BLOCK_TYPE_ID_TO_NAME.keys())
VALID_BLOCK_TYPE_NAMES: Final[frozenset[str]] = frozenset(BLOCK_TYPE_NAME_TO_ID.keys())

# Guard: the sentinel must never collide with a real block type.
assert UNSET_BLOCK_TYPE_ID_SENTINEL not in VALID_BLOCK_TYPE_IDS, (
    "Sentinel collides with a valid block-type ID"
)


# ---------------------------------------------------------------------------
# Light-weight demo / introspection (kept intentionally simple).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- BlockType Enumeration ---\n")
    for block_type in BlockType:
        print(f"{block_type.name:40s}  id = {block_type.value:>2d}")

    print("\nID → Name mapping:")
    print(BLOCK_TYPE_ID_TO_NAME)

    print("\nName → ID mapping:")
    print(BLOCK_TYPE_NAME_TO_ID)

    print("\nValid BlockType IDs:")
    print(VALID_BLOCK_TYPE_IDS)

    print("\nValid BlockType Names:")
    print(VALID_BLOCK_TYPE_NAMES)

    print("\nSentinel value for unset block type ID:")
    print(UNSET_BLOCK_TYPE_ID_SENTINEL)

    assert UNSET_BLOCK_TYPE_ID_SENTINEL not in VALID_BLOCK_TYPE_IDS, (
        "Sentinel collides with a valid block-type ID"
    )

    print("\nAll checks passed: sentinel does not collide with valid block type IDs.")
    print("\n----------------------------\n")
