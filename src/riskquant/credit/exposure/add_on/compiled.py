# src/riskquant/credit/exposure/add_on/compiled.py
"""
Supervisory add-on factors, keyed by asset-subclass UIX.

`CompiledAddOnFactors` is the SoA form of the supervisory add-on table: one factor
per asset subclass (energy, power, metals, environmental, ...), the CEM analogue
of SA-CCR's supervisory-factor table. It is the sole home of the add-on
percentages — no magic numbers in the resolve or the kernel. `subclass_uix` is
ascending-sorted and unique: the structural invariant the `searchsorted` lookup in
`resolve_supervisory_factors` relies on.
"""

from __future__ import annotations

__all__ = [
    "CompiledAddOnFactors",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledAddOnFactors:
    """Per-subclass supervisory add-on factors with their UIX join key."""

    subclass_uix: npt.NDArray[np.int32]  # (K,) ascending sorted, unique — join key
    factor: npt.NDArray[np.float64]  # (K,) add-on factor; aligned to subclass_uix

    def __post_init__(self) -> None:
        if self.subclass_uix.ndim != 1 or self.factor.ndim != 1:
            raise ValueError(
                "CompiledAddOnFactors: subclass_uix and factor must be 1-D."
            )
        if self.subclass_uix.shape[0] != self.factor.shape[0]:
            raise ValueError(
                f"CompiledAddOnFactors: factor count {self.factor.shape[0]} != "
                f"subclass count {self.subclass_uix.shape[0]}."
            )
        if self.subclass_uix.dtype != np.int32:
            raise TypeError("CompiledAddOnFactors: subclass_uix must be int32.")
        if self.factor.dtype != np.float64:
            raise TypeError("CompiledAddOnFactors: factor must be float64.")
        if self.subclass_uix.shape[0] > 1 and not bool(
            (np.diff(self.subclass_uix) > 0).all()
        ):
            raise ValueError(
                "CompiledAddOnFactors: subclass_uix must be ascending-sorted and "
                "unique (the searchsorted join key)."
            )

    @property
    def num_subclasses(self) -> int:
        return self.subclass_uix.shape[0]
