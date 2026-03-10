# src/riskquant/portfolio/compilers/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Optional, TypeAlias

from riskquant.portfolio.compilers.instruments.spot import (
    compile_spot,
)
from riskquant.portfolio.compilers.instruments.forward import (
    compile_forward_curve,
    compile_forward_node,
)
from riskquant.portfolio.compilers.instruments.vanilla_european_option import (
    compile_vanilla_euro_option_curve,
    compile_vanilla_euro_option_node,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import (
    SPOT_ID,
    FORWARD_NODE_ID,
    FORWARD_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
)

from riskquant.portfolio.blocks.spot_block import (
    SpotBlock,
)
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

# Updated to take the entire batch natively
CompilerFn: TypeAlias = Callable[[IngestBatch], Optional[BlockType]]


# ============================================================
# Registry
# ============================================================


@dataclass(frozen=True, slots=True)
class _CompilerSpec:
    field_name: str
    compiler: CompilerFn


# Removed the lambda wrappers to pass the function reference directly,
# avoiding unnecessary call overhead in the dispatch loop.
COMPILER_REGISTRY: Final[dict[int, _CompilerSpec]] = {
    int(SPOT_ID): _CompilerSpec(
        field_name="spot",
        compiler=compile_spot,
    ),
    int(FORWARD_NODE_ID): _CompilerSpec(
        field_name="forward_node",
        compiler=compile_forward_node,
    ),
    int(FORWARD_CURVE_ID): _CompilerSpec(
        field_name="forward_curve",
        compiler=compile_forward_curve,
    ),
    int(VANILLA_EUROPEAN_OPTION_NODE_ID): _CompilerSpec(
        field_name="vanilla_european_option_node",
        compiler=compile_vanilla_euro_option_node,
    ),
    int(VANILLA_EUROPEAN_OPTION_CURVE_ID): _CompilerSpec(
        field_name="vanilla_european_option_curve",
        compiler=compile_vanilla_euro_option_curve,
    ),
    # Add others as required
}
