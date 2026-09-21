# src/riskquant/credit/exposure/results.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class NettingSetExposurePaths:
    """
    NettingSetExposurePaths holds the exposure paths for a set of netting sets across
    multiple scenarios.

    NettingSetExposurePaths.exposure --> (n_netting_sets, n_scenarios) array of exposure values.

    Row i of every array describes the netting set identified by ``uid[i]``.
    ``uid`` is the join key into the downstream credit layer (the ``uid``
    column consumed by the deterministic / copula JTD ingest cases).
    """

    uid: npt.NDArray[np.int64]  # (N,)     effective-netting-set identifier
    exposure: npt.NDArray[np.float64]  # (N, S)   Exposure per netting set, per scenario

    def print(
        self,
        *,
        max_rows: int = 10,
        precision: int = 0,
        summary: bool = True,
    ) -> None:
        """
        Pretty-print the simulated exposure result.

        Parameters
        ----------
        max_rows
            Maximum number of rows to display.
        precision
            Number of decimal places for floats.
        summary
            Whether to print global matrix summary statistics.
        """
        n_rows, n_scenarios = self.exposure.shape

        print(
            f"NettingSetExposurePaths: {n_rows} netting sets × {n_scenarios} scenarios"
        )

        display_rows = min(n_rows, max_rows)

        # Adjusted headers to accommodate the row-wise Expected Exposure
        print(f"\n{'UID':>12} | {'Expected Exposure (EE)':>22} | Simulated Paths")
        print("-" * 91)

        # Local print-options scope: never mutate numpy globals.
        with np.printoptions(precision=precision, suppress=True):
            for i in range(display_rows):
                uid_str = f"{self.uid[i]:>12_.0f}"

                # Expected Exposure: The mean of this specific netting set across all scenarios
                ee_str = f"{self.exposure[i].mean():>22_.{precision}f}"

                # The raw stochastic distribution paths
                exposure_str = np.array2string(
                    self.exposure[i],
                    separator=", ",
                    formatter={"float_kind": lambda x: f"{x:>18_.{precision}f}"},
                )

                print(f"{uid_str} | {ee_str} | {exposure_str}")

        if display_rows < n_rows:
            print(f"... ({n_rows - display_rows} more rows)")

        if summary:
            # Renamed to clarify these are global matrix stats, not counterparty metrics
            print("\nGlobal matrix statistics (across all netting sets & scenarios):")
            print(f"  Mean : {self.exposure.mean():>18_.{precision}f}")
            print(f"  Std  : {self.exposure.std():>18_.{precision}f}")
            print(f"  Min  : {self.exposure.min():>18_.{precision}f}")
            print(f"  Max  : {self.exposure.max():>18_.{precision}f}")
