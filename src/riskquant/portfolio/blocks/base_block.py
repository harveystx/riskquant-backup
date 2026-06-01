# src/riskquant/portfolio/blocks/base_block.py

"""
Base class for all Struct-of-Arrays (SoA) trade blocks.

`Block` carries only what is universal to every compiled block: the flat int64
`trade_uids` array and the `INSTRUMENT_ID` dispatch tag. All economic SoA arrays
live on the concrete subclasses.

`INSTRUMENT_ID` is the single integer the valuation engine dispatches on: it
indexes directly into `BLOCK_EVALUATORS`, so no `type()` calls, no isinstance
ladders, no string keys. The base value is the `_UNSET` sentinel; every
concrete block MUST override it with its `Instrument` enum id. The override is
enforced at class-creation time by `__init_subclass__`, which fails loudly the
moment a new block type is added without wiring its id — a compile-time guard
rather than a runtime surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

# Sentinel for "concrete subclass forgot to set INSTRUMENT_ID". Negative so it
# can never collide with a real Instrument enum id or index a dispatch table.
_UNSET_INSTRUMENT_ID: int = -999


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    """Universal base for compiled SoA trade blocks."""

    # Dispatch tag. ClassVar => not a dataclass field, zero per-instance cost.
    INSTRUMENT_ID: ClassVar[int] = _UNSET_INSTRUMENT_ID

    trade_uids: npt.NDArray[np.int64]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """
        Enforce the dispatch contract for *concrete* block subclasses.

        Intermediate abstract bases (names starting with '_') are exempt: they
        legitimately leave `INSTRUMENT_ID` unset for their own subclasses to
        define. Any concrete block that ships without an id is a wiring bug and
        is rejected here, at import time.
        """
        # super().__init_subclass__(**kwargs)
        is_abstract_base = cls.__name__.startswith("_")
        if not is_abstract_base and cls.INSTRUMENT_ID == _UNSET_INSTRUMENT_ID:
            raise TypeError(
                f"Block subclass '{cls.__name__}' must set INSTRUMENT_ID "
                f"(its riskquant.portfolio.instruments id) for engine dispatch."
            )

    @property
    def num_trades(self) -> int:
        """Number of trades packed into this block."""
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
