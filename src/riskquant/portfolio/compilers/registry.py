from __future__ import annotations

from typing import Callable, Final, Optional, TypeAlias

from riskquant.portfolio.ingest.batch import IngestBatch

from riskquant.portfolio.compilers.instruments.spot import compile_spot
from riskquant.portfolio.compilers.instruments.forward import (
    compile_forward_curve,
    compile_forward_node,
)
from riskquant.portfolio.compilers.instruments.vanilla_european_option import (
    compile_vanilla_euro_option_curve,
    compile_vanilla_euro_option_node,
)

from riskquant.portfolio.instruments import (
    NUM_INSTRUMENTS,
    SPOT_ID,
    FORWARD_NODE_ID,
    FORWARD_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
)

from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.blocks.forward_block import (
    ForwardNodeBlock,
    ForwardCurveBlock,
)
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
    VanillaEuropeanOptionCurveBlock,
)

# ============================================================
# Type aliases
# ============================================================

BlockType: TypeAlias = (
    SpotBlock
    | ForwardNodeBlock
    | ForwardCurveBlock
    | VanillaEuropeanOptionNodeBlock
    | VanillaEuropeanOptionCurveBlock
)

CompilerFn: TypeAlias = Callable[
    [IngestBatch, int, int],
    Optional[BlockType],
]

# ============================================================
# Dispatch tables (GOD-TIER)
# ============================================================

# If instrument IDs are small integers → this is fastest possible dispatch

_MAX_INSTRUMENT_ID: Final = NUM_INSTRUMENTS

COMPILERS: Final[list[Optional[CompilerFn]]] = [None] * _MAX_INSTRUMENT_ID
FIELD_NAMES: Final[list[Optional[str]]] = [None] * _MAX_INSTRUMENT_ID


def _register(instrument_id: int, field_name: str, fn: CompilerFn) -> None:
    COMPILERS[instrument_id] = fn
    FIELD_NAMES[instrument_id] = field_name


# ============================================================
# Registration (executed once at import)
# ============================================================

# ---> SPOT
_register(
    int(SPOT_ID),
    "spot",
    compile_spot,
)

# ---> FORWARDS
_register(
    int(FORWARD_NODE_ID),
    "forward_node",
    compile_forward_node,
)
_register(
    int(FORWARD_CURVE_ID),
    "forward_curve",
    compile_forward_curve,
)

# ---> VANILLA EUROPEAN OPTIONS
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

# ---> REGISTER MORE INSTRUMENTS HERE... (if instrument_id < _MAX_INSTRUMENT_ID)
