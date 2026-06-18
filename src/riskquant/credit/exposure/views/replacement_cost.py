# src/riskquant/credit/exposure/views/replacement_cost.py

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.models import REPLACEMENT_COST_ID
from riskquant.credit.exposure.views.base import _NettingSetView


@dataclass(frozen=True, slots=True, kw_only=True)
class RcNettingSetView(_NettingSetView):
    """Replacement Cost netting-set view: CSR grouping + net collateral."""

    EXPOSURE_MODEL_ID: ClassVar[int] = REPLACEMENT_COST_ID

    collateral: npt.NDArray[np.float64]

    def print(
        self,
        *,
        max_items: int | None = 10,
        precision: int = 2,
        show_collateral: bool = True,
        add_space_before_table: bool = True,
    ) -> None:
        """
        Pretty-print the RcNettingSetView.

        Parameters
        ----------
        max_rows
            Maximum number of netting sets to display.
        precision
            Number of decimal places for floats.
        show_collateral
            Whether to display the collateral column.
        add_space_before_table
            Insert a blank line before the table header.
        """
        max_items = max_items if max_items is not None else 10
        n_sets = self.num_netting_sets
        n_trades = self.num_trades

        if add_space_before_table:
            print()

        print(f"RcNettingSetView: {n_sets} netting sets, {n_trades} trades")

        display_rows = min(n_sets, max_items)

        header = f"{'Netting Set UID':>16}"
        if show_collateral:
            header += f" | {'Collateral':>12}"
        header += f" | {'Netting Set Offsets':>20}"
        print()
        print(header)
        print("-" * 80)

        with np.printoptions(precision=precision, suppress=True):
            for i in range(display_rows):
                uid_str = f"{self.netting_set_uids[i]:>16_.0f}"

                collateral_str = ""
                if show_collateral:
                    collateral_str = f" | {self.collateral[i]:>12_.{precision}f}"

                offsets = self.netting_set_offsets[i : i + 2]
                offsets_str = f" | [{offsets[0]:>6}, {offsets[1]:>6})"

                print(f"{uid_str}{collateral_str}{offsets_str}")

            if display_rows < n_sets:
                print(f"... ({n_sets - display_rows} more netting sets)")

            print(f"\nSummary statistics (across {n_sets} netting sets):")
            print(f"  Total collateral: {self.collateral.sum():>16_.{precision}f}")
            print(f"  Mean collateral : {self.collateral.mean():>16_.{precision}f}")
            print(f"  Std collateral  : {self.collateral.std():>16_.{precision}f}")
            print(f"  Min collateral  : {self.collateral.min():>16_.{precision}f}")
            print(f"  Max collateral  : {self.collateral.max():>16_.{precision}f}")

            print(f"\nPermutation (first {min(10, n_trades)} rows):")
            print(f"  mtm_row_perm[:10] = {self.mtm_row_perm[:10]}")
            if n_trades > 10:
                print(f"  ... ({n_trades - 10} more)")
