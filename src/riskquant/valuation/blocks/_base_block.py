# src/riskquant/portfolio/blocks/base_block.py

"""
Base class for all Struct-of-Arrays (SoA) trade blocks.

`Block` carries only what is universal to every compiled block: the flat int64
`trade_uids` array and the `BLOCK_TYPE_ID` dispatch tag. All economic SoA arrays
live on the concrete subclasses.

`BLOCK_TYPE_ID` is the single integer the valuation engine dispatches on: it
indexes directly into `BLOCK_EVALUATORS`, so no `type()` calls, no isinstance
ladders, no string keys. The base value is the `UNSET_BLOCK_TYPE_ID_SENTINEL`
sentinel; every concrete block MUST override it with its `BlockType` enum id.
The override is enforced at class-creation time by `__init_subclass__`, which
fails loudly the moment a new block type is added without wiring its id — a
compile-time guard rather than a runtime surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks.block_type import (
    UNSET_BLOCK_TYPE_ID_SENTINEL,
    VALID_BLOCK_TYPE_IDS,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    """
    Universal base for compiled SoA valuation blocks.

    Every array element of every field corresponds to the same row index.
    Concrete subclasses define the economic SoA fields required by their
    valuation kernel.

    `trade_uids[i]` identifies the originating trade that produced row `i` of the block.
    """

    # Block type identifier for dispatching to the right pricer kernel. Every concrete
    # subclass MUST override this with its `BlockType` enum id, which is
    # guaranteed to be unique.
    BLOCK_TYPE_ID: ClassVar[int] = UNSET_BLOCK_TYPE_ID_SENTINEL

    trade_uids: npt.NDArray[np.int64]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls.__name__.startswith("_"):
            return

        if cls.BLOCK_TYPE_ID == UNSET_BLOCK_TYPE_ID_SENTINEL:
            raise TypeError(f"{cls.__name__} must define BLOCK_TYPE_ID")

        if cls.BLOCK_TYPE_ID not in VALID_BLOCK_TYPE_IDS:
            raise ValueError(
                f"{cls.__name__} has invalid BLOCK_TYPE_ID {cls.BLOCK_TYPE_ID}"
            )

    @property
    def size(self) -> int:
        """
        Number of elements in a block, i.e. number of cashflows, legs, or
        trades it represents.
        """
        return self.trade_uids.shape[0]

    def print_block(
        self,
        *,
        max_items: int | None = 10,
        float_precision: int = 4,
        sep: str = ", ",
        pad: int = 15,  # Increased default pad slightly to account for the new '_' characters
        thousand_sep: str = "_",
    ) -> None:
        """Print every SoA field with shape, dtype, and a value preview."""

        print(f"{type(self).__name__}:")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                # Replace lambdas with typed inner functions to satisfy Pyright/MyPy
                def fmt_float(x: Any) -> str:
                    return f"{x:>{pad}{thousand_sep}.{float_precision}f}"

                def fmt_int(x: Any) -> str:
                    # Use 'd' (decimal integer) instead of a hacked float format
                    return f"{x:>{pad}{thousand_sep}d}"

                # Format the array preview cleanly
                val_preview = np.array2string(
                    v[:max_items],
                    separator=sep,
                    max_line_width=100_000,
                    formatter={
                        "float_kind": fmt_float,
                        "int_kind": fmt_int,
                    },
                )

                print(
                    f"  {f.name:<33} | shape: {v.shape!s:>10} | "
                    f"dtype: {v.dtype!s:>10} | values: {val_preview}"
                )
            else:
                print(f"  {f.name:<33} {v}")
        print()
