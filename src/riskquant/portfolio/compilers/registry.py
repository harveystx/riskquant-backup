# src/riskquant/portfolio/compilers/registry.py
"""
Per-instrument block compiler dispatch table.

Each compiler is a `(batch, start, end, uix_mapper) -> Block | None`
callable. The engine's `compile_portfolio_from_ingest_batch` sorts the
batch by `instrument_id`, partitions it into contiguous runs, and resolves
the compiler for each run by direct integer indexing into `COMPILERS`.

The mapper is passed positionally as the fourth argument so the engine
loop stays branch-light and the call sites are uniform.

Instrument vs block taxonomy
----------------------------
`InstrumentType` keeps the node/curve distinction (a real ingest-spec input:
node level vs interpolated curve point). The *block* taxonomy is flat -- both
forward instruments compile to one `VanillaForwardBlock`, both option
instruments to one `VanillaEuropeanOptionBlock`. Node-vs-curve becomes a
compile-time forward-source tag inside the effective-forward universe, never a
block type.

Registration is partial during buildout: only instruments whose flat compiler
exists are registered here. Unregistered instruments are recorded in
`UNREGISTERED_INSTRUMENT_IDS` for diagnostics; the engine fails loud if a batch
actually contains a run of one of them (it never silently drops trades).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, TypeAlias

from riskquant.portfolio.compilers.block_compilers.spot import (
    compile_spot_block,
)
from riskquant.portfolio.compilers.block_compilers.vanilla_european_option import (
    compile_vanilla_european_option_block,
)
from riskquant.portfolio.compilers.block_compilers.vanilla_forward import (
    compile_vanilla_forward_block,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import (
    NUM_INSTRUMENTS,
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
)
from riskquant.valuation.blocks._base_block import Block

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


CompilerFn: TypeAlias = Callable[
    [IngestBatch, int, int, "UixMapper"],
    Block | None,
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

# --- SPOT ---
_register(
    int(SPOT_ID),
    "spot",
    compile_spot_block,
)

# --- VANILLA FORWARD (NODE-sourced) ---
_register(
    int(VANILLA_FORWARD_NODE_ID),
    "vanilla_forward",
    compile_vanilla_forward_block,
)

# --- VANILLA EUROPEAN OPTION (NODE-sourced) ---
_register(
    int(VANILLA_EUROPEAN_OPTION_NODE_ID),
    "vanilla_european_option",
    compile_vanilla_european_option_block,
)

# --- PENDING (flat compilers not yet authored) ---------------------------------
# VANILLA_FORWARD_CURVE_ID                -> field "vanilla_forward"
#   Reads `price_curve_grid`, not `forward_nodes`; needs a source-kind column in
#   the effective-forward universe.
# VANILLA_EUROPEAN_OPTION_CURVE_ID        -> field "vanilla_european_option"
#   Reads `vol_surfaces`; needs surface interpolation the flat `vols` universe
#   does not model.
# VANILLA_IRS_ID                          -> decomposes into cashflow blocks
#   `CompiledPortfolio` has no cashflow slot yet.
# All stay unregistered so the engine fails loud on encounter rather than
# pricing off the wrong array.

# Diagnostics only -- no import-time hard fail (partial registration is the
# intended buildout state).
UNREGISTERED_INSTRUMENT_IDS: Final[tuple[int, ...]] = tuple(
    i for i, fn in enumerate(COMPILERS) if fn is None
)
