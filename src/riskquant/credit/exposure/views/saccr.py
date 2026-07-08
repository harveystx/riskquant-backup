# src/riskquant/credit/exposure/views/saccr.py

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.models import SACCR_ID
from riskquant.credit.exposure.views.base import _NettingSetView


@dataclass(frozen=True, slots=True, kw_only=True)
class SaccrNettingSetView(_NettingSetView):
    """SA-CCR view: RC CSR grouping + collateral + the scenario-independent add-on.

    ``add_on`` is the hedging-set-aggregated PFE add-on per netting set (computed
    at compile via `compute_saccr_add_on`); the kernel applies the supervisory
    multiplier and alpha per scenario. ``alpha`` and ``multiplier_floor`` are
    carried on the view so the kernel takes no config object across the JIT wall.
    """

    EXPOSURE_MODEL_ID: ClassVar[int] = SACCR_ID

    collateral: npt.NDArray[np.float64]  # (N,) net VM per netting set
    add_on: npt.NDArray[np.float64]  # (N,) hedging-set-aggregated add-on, scenario-free
    alpha: float
    multiplier_floor: float

    def print(self, *, max_items: int | None = 10, precision: int = 2) -> None:
        max_items = max_items if max_items is not None else 10
        n_sets = self.num_netting_sets
        print(
            f"\nSaccrNettingSetView: {n_sets} netting sets, {self.num_trades} trades "
            f"(alpha={self.alpha}, floor={self.multiplier_floor})"
        )
        header = f"{'Netting Set UID':>16} | {'Collateral':>14} | {'AddOn':>16} | {'Band':>16}"
        print("\n" + header)
        print("-" * len(header))
        with np.printoptions(precision=precision, suppress=True):
            for i in range(min(n_sets, max_items)):
                start = int(self.netting_set_offsets[i])
                end = int(self.netting_set_offsets[i + 1])
                print(
                    f"{self.netting_set_uids[i]:>16_.0f} | "
                    f"{self.collateral[i]:>14_.{precision}f} | "
                    f"{self.add_on[i]:>16_.{precision}f} | [{start:>6}, {end:>6})"
                )
            if n_sets > max_items:
                print(f"... ({n_sets - max_items} more netting sets)")
