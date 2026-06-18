# src/riskquant/credit/jtd/blocks/base.py
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
import numpy.typing as npt

_UNSET_JTD_MODEL_ID: int = -999


@dataclass(frozen=True, slots=True, kw_only=True)
class _JtdBlock:
    """Model-agnostic base for compiled obligor-keyed JTD blocks."""

    # ===== Class contract =====

    JTD_MODEL_ID: ClassVar[int] = _UNSET_JTD_MODEL_ID
    _ABSTRACT: ClassVar[bool] = True  # opt-in per class (not inherited)

    # ===== Core data (assumed valid) =====

    obligor_uids: npt.NDArray[np.int64]
    lgd: npt.NDArray[np.float64]

    # ===== Class construction guard =====

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Explicit super() bypasses the @dataclass(slots=True) closure cell bug
        super(_JtdBlock, cls).__init_subclass__(**kwargs)

        # Only consider attributes defined *on this class*
        is_abstract = cls.__dict__.get("_ABSTRACT", False) or cls.__name__.startswith(
            "_"
        )

        if is_abstract:
            return

        if cls.JTD_MODEL_ID == _UNSET_JTD_MODEL_ID:
            raise TypeError(f"JTD block '{cls.__name__}' must set JTD_MODEL_ID.")

    # ===== Lightweight API =====

    @property
    def num_obligors(self) -> int:
        return self.obligor_uids.shape[0]

    def print_block(self, max_items: int | None = 10) -> None:
        print(f"{type(self).__name__}:")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                print(
                    f"  {f.name:<36} | shape: {v.shape!s:>10} | "
                    f"dtype: {v.dtype!s:>8} | values: {v[:max_items]}"
                )
            else:
                print(f"  {f.name:<36} {v}")
        print()
