# src/riskquant/valuation/pricer_registry.py

"""
Registry of block pricers for the valuation engine.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

# VAULATION STATE
from riskquant.market_data.state import ValuationState

# BLOCK TYPE DATA
from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    FIXED_CASHFLOW_BLOCK_TYPE_ID,
    FLOATING_CASHFLOW_BLOCK_TYPE_ID,
    NUM_BLOCK_TYPES,
    PRINCIPAL_CASHFLOW_BLOCK_TYPE_ID,
    SPOT_BLOCK_TYPE_ID,
    VANILLA_EUROPEAN_OPTION_BLOCK_TYPE_ID,
    VANILLA_FORWARD_BLOCK_TYPE_ID,
    BlockType,
)

# BLOCK PRICERS
from riskquant.valuation.pricers.fixed_cashflow import (
    price_fixed_cashflow_block,
)
from riskquant.valuation.pricers.floating_cashflow import (
    price_floating_cashflow_block,
)
from riskquant.valuation.pricers.principal_cashflow import (
    price_principal_cashflow_block,
)
from riskquant.valuation.pricers.spot import (
    price_spot_block,
)
from riskquant.valuation.pricers.vanilla_european_option import (
    price_vanilla_european_option_block,
)
from riskquant.valuation.pricers.vanilla_forward import (
    price_vanilla_forward_block,
)

# ==============================================================================
# PRICER PROTOCOL
# ==============================================================================

# 1. Define a contravariant TypeVar bound to Block
B_contra = TypeVar("B_contra", bound=Block, contravariant=True)


@runtime_checkable
class Pricer(Protocol[B_contra]):
    """
    Structural contract for all block pricers.
    """

    def __call__(
        self,
        *,
        block: B_contra,  # 2. Use the TypeVar instead of base Block
        valuation_state: ValuationState,
        out: npt.NDArray[np.float64],
    ) -> None: ...


# ==============================================================================
# DISPATCH TABLE
# ==============================================================================

# 3. Explicitly type the list elements as Pricer[Any] | None
_BLOCK_PRICERS_MUT: list[Pricer[Any] | None] = [None] * NUM_BLOCK_TYPES


def _register_block_pricer(block_type_id: int, pricer: Pricer[Any]) -> None:
    # runtime_checkable ignores TypeVars, so this check still works natively
    if not isinstance(pricer, Pricer):
        raise TypeError(
            f"Pricer for block_type_id={block_type_id} does not satisfy the "
            f"Pricer protocol: {pricer!r}"
        )
    _BLOCK_PRICERS_MUT[block_type_id] = pricer


# --- REGISTER ALL BLOCK PRICERS HERE ---

_register_block_pricer(
    int(SPOT_BLOCK_TYPE_ID),
    price_spot_block,
)
_register_block_pricer(
    int(VANILLA_FORWARD_BLOCK_TYPE_ID),
    price_vanilla_forward_block,
)
_register_block_pricer(
    int(VANILLA_EUROPEAN_OPTION_BLOCK_TYPE_ID),
    price_vanilla_european_option_block,
)
_register_block_pricer(
    int(PRINCIPAL_CASHFLOW_BLOCK_TYPE_ID),
    price_principal_cashflow_block,
)
_register_block_pricer(
    int(FIXED_CASHFLOW_BLOCK_TYPE_ID),
    price_fixed_cashflow_block,
)
_register_block_pricer(
    int(FLOATING_CASHFLOW_BLOCK_TYPE_ID),
    price_floating_cashflow_block,
)

# Freeze. No mutations after module load.
# This tells the type checker: "Trust me, the runtime guard above handled the Nones."
BLOCK_PRICERS: Final[tuple[Pricer[Any], ...]] = cast(
    tuple[Pricer[Any], ...], tuple(_BLOCK_PRICERS_MUT)
)

# Hard fail at import if any block has no pricer: catches the case where
# a new Block variant is added but not wired up here.
_unregistered_blocks = [
    BlockType(i).name for i, ev in enumerate(BLOCK_PRICERS) if ev is None
]
if _unregistered_blocks:
    raise RuntimeError(
        f"No pricer registered for blocks: {_unregistered_blocks}. "
        f"Register them in registry.py."
    )


def resolve_pricer(block: Block) -> Pricer:
    """
    Resolve the pricer for a compiled block via its `BLOCK_TYPE_ID`.

    Single-array index — no `type()`, no isinstance ladder, no string keys.
    """
    pricer = BLOCK_PRICERS[block.BLOCK_TYPE_ID]
    if pricer is None:  # unreachable given the import-time guard above
        raise RuntimeError(
            f"No pricer for BLOCK_TYPE_ID={block.BLOCK_TYPE_ID} "
            f"({type(block).__name__})."
        )
    return pricer
