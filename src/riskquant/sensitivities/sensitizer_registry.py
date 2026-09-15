# src/riskquant/sensitivities/sensitizer_registry.py
"""
Registry of block sensitizers for the sensitivity engine.

Deliberate mirror of `valuation.pricer_registry`: a `Sensitizer` Protocol, a
flat table indexed by `Block.BLOCK_TYPE_ID`, frozen after module load, and
resolved by a single array index -- no `type()`, no isinstance ladder, no
string-keyed numerical dispatch.

Divergence from `pricer_registry`, and why
------------------------------------------
`pricer_registry` hard-fails AT IMPORT if any `BlockType` lacks a pricer. That
is correct there: every block type has a real, authored pricer.

Sensitivity coverage is being built incrementally, and only `SPOT` has an
authored, validated implementation today. Reproducing the import-time guard
would force stub sensitizers for forwards, options, and cashflows -- and a
stub that returns empty or zero sensitivities is far worse than no
sensitivity at all, because it silently reports a risk-free position for a
trade that carries real risk.

So the guard moves from import time to CALL time, per block: an unregistered
block type raises the moment something actually tries to differentiate it.
Coverage stays honest, nothing is silently zero, and the import-time
completeness check returns for free once the remaining sensitizers are
authored (see `UNREGISTERED_BLOCK_TYPE_NAMES` below, which names the gap
explicitly rather than hiding it).
"""

from __future__ import annotations

from typing import Any, Final, Protocol, TypeVar, runtime_checkable

from riskquant.market_data.state import MarketDataMatrix
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizers.spot import sensitize_spot_block
from riskquant.universe import UixMapper
from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    NUM_BLOCK_TYPES,
    SPOT_BLOCK_TYPE_ID,
    BlockType,
)

# ==============================================================================
# SENSITIZER PROTOCOL
# ==============================================================================

B_contra = TypeVar("B_contra", bound=Block, contravariant=True)


@runtime_checkable
class Sensitizer(Protocol[B_contra]):
    """Structural contract for all block sensitizers."""

    def __call__(
        self,
        *,
        block: B_contra,
        uix_mapper: UixMapper,
        market_data: MarketDataMatrix,
        scenario: int = ...,
    ) -> BlockSensitivities: ...


# ==============================================================================
# DISPATCH TABLE
# ==============================================================================

_BLOCK_SENSITIZERS_MUT: list[Sensitizer[Any] | None] = [None] * NUM_BLOCK_TYPES


def _register_block_sensitizer(block_type_id: int, sensitizer: Sensitizer[Any]) -> None:
    if not isinstance(sensitizer, Sensitizer):
        raise TypeError(
            f"Sensitizer for block_type_id={block_type_id} does not satisfy "
            f"the Sensitizer protocol: {sensitizer!r}"
        )
    _BLOCK_SENSITIZERS_MUT[block_type_id] = sensitizer


# --- REGISTER ALL AUTHORED BLOCK SENSITIZERS HERE ---
# Only block types with a real, validated implementation appear here. Do NOT
# add a stub to silence the coverage gap; see the module docstring.

_register_block_sensitizer(
    int(SPOT_BLOCK_TYPE_ID),
    sensitize_spot_block,
)

# Freeze. No mutations after module load.
BLOCK_SENSITIZERS: Final[tuple[Sensitizer[Any] | None, ...]] = tuple(
    _BLOCK_SENSITIZERS_MUT
)

# The coverage gap, named explicitly at import so it is discoverable and
# assertable rather than implicit.
UNREGISTERED_BLOCK_TYPE_NAMES: Final[tuple[str, ...]] = tuple(
    BlockType(i).name for i, s in enumerate(BLOCK_SENSITIZERS) if s is None
)


def resolve_sensitizer(block: Block) -> Sensitizer[Any]:
    """
    Resolve the sensitizer for a compiled block via its `BLOCK_TYPE_ID`.

    Single-array index. Raises if the block type has no authored sensitizer,
    so an uncovered block type can never be silently skipped or reported as
    carrying zero risk.
    """
    sensitizer = BLOCK_SENSITIZERS[block.BLOCK_TYPE_ID]
    if sensitizer is None:
        raise NotImplementedError(
            f"No sensitizer registered for BLOCK_TYPE_ID="
            f"{block.BLOCK_TYPE_ID} ({BlockType(block.BLOCK_TYPE_ID).name}). "
            f"Authored sensitizers cover: "
            f"{[BlockType(i).name for i, s in enumerate(BLOCK_SENSITIZERS) if s is not None]}. "
            f"Register it in sensitizer_registry.py."
        )
    return sensitizer
