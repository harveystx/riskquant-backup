# src/riskquant/portfolio/compilers/registry.py
"""
Per-instrument block compiler dispatch table.

Each compiler is a `(batch, start, end, uix_mapper) -> Block | None`
callable. The engine's `compile_portfolio_from_ingest_batch` sorts the
batch by `instrument_id`, partitions it into contiguous runs, and resolves
the compiler for each run by direct integer indexing into `COMPILERS`.

The mapper is passed positionally as the fourth argument so the engine
loop stays branch-light and the call sites are uniform.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, TypeAlias

from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionCurveBlock,
    VanillaEuropeanOptionNodeBlock,
)
from riskquant.portfolio.blocks.vanilla_forward_block import (
    VanillaForwardCurveBlock,
    VanillaForwardNodeBlock,
)
from riskquant.portfolio.compilers.block_compilers.spot import compile_spot
from riskquant.portfolio.compilers.block_compilers.vanilla_european_option import (
    compile_vanilla_euro_option_curve,
    compile_vanilla_euro_option_node,
)
from riskquant.portfolio.compilers.block_compilers.vanilla_forward import (
    compile_vanilla_forward_curve,
    compile_vanilla_forward_node,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import (
    NUM_INSTRUMENTS,
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
)

if TYPE_CHECKING:
    from riskquant.portfolio.engine import UixMapper


# ==============================================================================
# Type aliases
# ==============================================================================

BlockType: TypeAlias = (
    SpotBlock
    | VanillaForwardNodeBlock
    | VanillaForwardCurveBlock
    | VanillaEuropeanOptionNodeBlock
    | VanillaEuropeanOptionCurveBlock
)

CompilerFn: TypeAlias = Callable[
    [IngestBatch, int, int, "UixMapper"],
    BlockType | None,
]


# ==============================================================================
# Dispatch tables
# ==============================================================================

_MAX_INSTRUMENT_ID: Final = NUM_INSTRUMENTS

COMPILERS: Final[list[CompilerFn | None]] = [None] * _MAX_INSTRUMENT_ID
FIELD_NAMES: Final[list[str | None]] = [None] * _MAX_INSTRUMENT_ID


def _register(instrument_id: int, field_name: str, fn: CompilerFn) -> None:
    COMPILERS[instrument_id] = fn
    FIELD_NAMES[instrument_id] = field_name


# ==============================================================================
# Registration (executed once at import)
# ==============================================================================

_register(int(SPOT_ID), "spot", compile_spot)

_register(
    int(VANILLA_FORWARD_NODE_ID),
    "vanilla_forward_node",
    compile_vanilla_forward_node,
)
_register(
    int(VANILLA_FORWARD_CURVE_ID),
    "vanilla_forward_curve",
    compile_vanilla_forward_curve,
)

_register(
    int(VANILLA_EUROPEAN_OPTION_NODE_ID),
    "vanilla_european_option_node",
    compile_vanilla_euro_option_node,
)
_register(
    int(VANILLA_EUROPEAN_OPTION_CURVE_ID),
    "vanilla_european_option_curve",
    compile_vanilla_euro_option_curve,
)


# Fail loud on missing registrations at import time.
_unregistered = [i for i, fn in enumerate(COMPILERS) if fn is None]
if _unregistered:
    raise RuntimeError(
        f"No compiler registered for instrument IDs: {_unregistered}. "
        f"Register them in compilers.registry."
    )
