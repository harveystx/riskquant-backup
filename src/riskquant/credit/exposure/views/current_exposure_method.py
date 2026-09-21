# src/riskquant/credit/exposure/views/current_exposure_method.py

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.models import CURRENT_EXPOSURE_METHOD_ID
from riskquant.credit.exposure.views.base import _NettingSetView


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CemNettingSetView(_NettingSetView):
    """Current Exposure Method view: RC CSR grouping + net collateral + the two
    per-trade add-on scalars.

    The trade-level gross add-on is ``effective_notional * supervisory_factor``,
    kept as two arrays rather than one product so the SA-CCR upgrade is a scalar
    swap, not a reshape: ``effective_notional`` gains supervisory delta and the
    maturity factor (both 1 under CEM), and ``supervisory_factor`` moves to a
    per-hedging-set application. Both are stored in netting-set-contiguous (CSR)
    order — the ``mtm_row_perm`` axis the offsets index — so the evaluator forms
    the product over a band with no gather. They are scenario-independent (current
    marks), so their contribution is reduced once, outside the scenario loop.
    """

    EXPOSURE_MODEL_ID: ClassVar[int] = CURRENT_EXPOSURE_METHOD_ID

    collateral: npt.NDArray[np.float64]  # (N,)     net VM per netting set
    effective_notional: npt.NDArray[
        np.float64
    ]  # (M_view,) adjusted notional, CSR order

    supervisory_factor: npt.NDArray[np.float64]  # (M_view,) add-on factor, CSR order
    pfe_floor: float  # NGR compression: net_pfe = gross * (floor + coeff * NGR)
    pfe_coefficient: float

    def print(self, *, max_items: int | None = 10, precision: int = 2) -> None:
        """Console overview: per-set collateral, band, and summed gross add-on."""
        max_items = max_items if max_items is not None else 10
        n_sets = self.num_netting_sets
        n_trades = self.num_trades
        print(f"\nCemNettingSetView: {n_sets} netting sets, {n_trades} trades")

        header = (
            f"{'Netting Set UID':>16} | {'Collateral':>14} | "
            f"{'Gross Add-On (set)':>18} | {'Band':>16}"
        )
        print("\n" + header)
        print("-" * len(header))

        display_rows = min(n_sets, max_items)
        with np.printoptions(precision=precision, suppress=True):
            for i in range(display_rows):
                ns_offset_i = self.netting_set_offsets[i].item()
                ns_offset_ip1 = self.netting_set_offsets[i + 1].item()
                start = int(ns_offset_i)
                end = int(ns_offset_ip1)
                add_on = float(
                    (
                        self.effective_notional[start:end]
                        * self.supervisory_factor[start:end]
                    ).sum()
                )
                print(
                    f"{self.netting_set_uids[i]:>16_.0f} | "
                    f"{self.collateral[i]:>14_.{precision}f} | "
                    f"{add_on:>18_.{precision}f} | [{start:>6}, {end:>6})"
                )
            if display_rows < n_sets:
                print(f"... ({n_sets - display_rows} more netting sets)")
