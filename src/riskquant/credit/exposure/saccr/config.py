# src/riskquant/credit/exposure/saccr/config.py
"""
SA-CCR configuration: hedging-set taxonomy + correlations + alpha + floor.

Everything here is HOUSE-DEFINED — the asset subclasses, how they roll into
hedging sets, the per-hedging-set supervisory correlation, and the scalars
(alpha, multiplier floor). SA-CCR's *structure* is fixed; its *parameters* are
ours. Supervisory factors stay in `CompiledAddOnFactors` (shared with CEM); this
config carries only the SA-CCR-specific bits. Join keys (`subclass_uix`,
`hedging_set_uix`) are ascending-sorted and unique for the searchsorted lookups.
"""

from __future__ import annotations

__all__ = ["CompiledSaccrConfig"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSaccrConfig:
    subclass_uix: npt.NDArray[np.int32]  # (K,) ascending, unique — commodity types
    subclass_hedging_set: npt.NDArray[np.int32]  # (K,) hedging set per subclass
    hedging_set_uix: npt.NDArray[np.int32]  # (H,) ascending, unique
    hedging_set_rho: npt.NDArray[np.float64]  # (H,) supervisory correlation, [0, 1]
    alpha: float  # EAD = alpha * (RC + PFE)
    multiplier_floor: float  # SA-CCR PFE-multiplier floor, (0, 1)

    def __post_init__(self) -> None:
        if (
            self.subclass_uix.dtype != np.int32
            or self.hedging_set_uix.dtype != np.int32
        ):
            raise TypeError("CompiledSaccrConfig: uix arrays must be int32.")
        if self.subclass_hedging_set.dtype != np.int32:
            raise TypeError("CompiledSaccrConfig: subclass_hedging_set must be int32.")
        if self.hedging_set_rho.dtype != np.float64:
            raise TypeError("CompiledSaccrConfig: hedging_set_rho must be float64.")
        if self.subclass_uix.shape != self.subclass_hedging_set.shape:
            raise ValueError("CompiledSaccrConfig: subclass arrays misaligned.")
        if self.hedging_set_uix.shape != self.hedging_set_rho.shape:
            raise ValueError("CompiledSaccrConfig: hedging-set arrays misaligned.")
        for name, key in (
            ("subclass_uix", self.subclass_uix),
            ("hedging_set_uix", self.hedging_set_uix),
        ):
            if key.shape[0] > 1 and not bool((np.diff(key) > 0).all()):
                raise ValueError(
                    f"CompiledSaccrConfig: {name} must be ascending, unique."
                )
        if not np.isin(self.subclass_hedging_set, self.hedging_set_uix).all():
            raise ValueError(
                "CompiledSaccrConfig: a subclass maps to an unknown hedging set."
            )
        if not ((self.hedging_set_rho >= 0.0) & (self.hedging_set_rho <= 1.0)).all():
            raise ValueError("CompiledSaccrConfig: hedging_set_rho must lie in [0, 1].")
        if not (self.alpha > 0.0):
            raise ValueError("CompiledSaccrConfig: alpha must be > 0.")
        if not (0.0 < self.multiplier_floor < 1.0):
            raise ValueError("CompiledSaccrConfig: multiplier_floor must be in (0, 1).")
