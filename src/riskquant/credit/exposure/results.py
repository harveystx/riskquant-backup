# src/riskquant/credit/exposure/results.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class EadResult:
    """
    Exposure-at-default, resolved per effective netting set.

    Row i of every array describes the netting set identified by ``uid[i]``.
    ``uid`` is the join key into the downstream credit layer (the ``uid``
    column consumed by the deterministic / copula JTD ingest cases).
    """

    uid: npt.NDArray[np.int64]  # (N,)     effective-netting-set identifier
    ead: npt.NDArray[np.float64]  # (N, S)   EAD per netting set, per scenario

    def print(
        self,
        *,
        max_rows: int = 10,
        precision: int = 0,
        summary: bool = True,
    ) -> None:
        """
        Pretty-print the EAD result.

        Parameters
        ----------
        max_rows
            Maximum number of rows to display.
        precision
            Number of decimal places for floats.
        summary
            Whether to print summary statistics.
        """
        n_rows, n_scenarios = self.ead.shape

        print(f"EadResult: {n_rows} netting sets × {n_scenarios} scenarios")

        display_rows = min(n_rows, max_rows)

        print(f"\n{'UID':>12} | EAD (per scenario)")
        print("-" * 50)

        # Local print-options scope: never mutate numpy globals.
        with np.printoptions(precision=precision, suppress=True):
            for i in range(display_rows):
                uid_str = f"{self.uid[i]:>12_.0f}"
                ead_str = np.array2string(
                    self.ead[i],
                    separator=", ",
                    formatter={"float_kind": lambda x: f"{x:>18_.{precision}f}"},
                )
                print(f"{uid_str} | {ead_str}")

        if display_rows < n_rows:
            print(f"... ({n_rows - display_rows} more rows)")

        if summary:
            print("\nSummary statistics (across all netting sets & scenarios):")
            print(f"  Mean : {self.ead.mean():>18_.{precision}f}")
            print(f"  Std  : {self.ead.std():>18_.{precision}f}")
            print(f"  Min  : {self.ead.min():>18_.{precision}f}")
            print(f"  Max  : {self.ead.max():>18_.{precision}f}")
