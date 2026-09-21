# src/riskquant/credit/exposure/obligor_ead/compiled.py
"""
Obligor-grain exposure-at-default, keyed by uid.

`CompiledObligorEad` is the obligor-level analogue of the exposure engine's
netting-set EAD: exposure that already lives at obligor grain — a vendor feed, or
a desk's direct input — carrying its own uid key. That key is what lets it be
reindexed onto a JTD portfolio's band order by uid (a `searchsorted` join) rather
than by the positional convention a bare ``(N, S)`` array would force on the
caller. `uid` is ascending-sorted and unique: the structural invariant the join
relies on.
"""

from __future__ import annotations

__all__ = [
    "CompiledObligorEad",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CompiledObligorEad:
    """
    Per-obligor EAD with its uid join key.

    `ead` is either per-obligor scalar ``(N,)`` (scenario-independent, e.g. a desk
    popping in one exposure per name) or scenario-varying ``(N, S)`` (e.g. a
    vendor exposure feed); its rows align to `uid`. `uid` is ascending-sorted and
    unique. Validation of dtypes and values lives in `compile_obligor_ead`; this
    container enforces only the structural invariants the uid-join depends on.
    """

    uid: npt.NDArray[np.int64]  # (N,) ascending sorted, unique — the join key
    ead: npt.NDArray[np.float64]  # (N,) or (N, S); rows aligned to uid

    def __post_init__(self) -> None:
        if self.uid.ndim != 1:
            raise ValueError(
                f"CompiledObligorEad: uid must be 1-D, got {self.uid.ndim}-D."
            )
        if self.ead.ndim not in (1, 2):
            raise ValueError(
                f"CompiledObligorEad: ead must be 1-D (N,) or 2-D (N, S), got "
                f"{self.ead.ndim}-D."
            )
        if self.ead.shape[0] != self.uid.shape[0]:
            raise ValueError(
                f"CompiledObligorEad: ead rows {self.ead.shape[0]} != uid count "
                f"{self.uid.shape[0]}."
            )
        if self.uid.shape[0] > 1 and not bool((np.diff(self.uid) > 0).all()):
            raise ValueError(
                "CompiledObligorEad: uid must be ascending-sorted and unique "
                "(the searchsorted join key)."
            )

    @property
    def num_obligors(self) -> int:
        return self.uid.shape[0]

    @property
    def is_scenario_varying(self) -> bool:
        return self.ead.ndim == 2
