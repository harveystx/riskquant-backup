# src/riskquant/solutions/sensitivity_pnl/handoff.py
"""
Writing a sensitivity block into a grid the caller already owns.

`run_sensitivity_pnl` allocates its own `(M, S)` block, which is right when the
sensitivity book is the whole portfolio. It is wrong when the book is one half of
a mixed portfolio priced alongside a full-reval block: taking two finished
matrices and concatenating them allocates a third while both sources are still
live, which is the 11.2 GB path `riskquant.sensitivity_pnl.grid` exists to kill.

So this is the co-allocation entry point. The caller allocates one grid across
all blocks and hands over the slot to fill:

    grid = allocate_pnl_grid(
        block_position_uids=[full_reval_uids, sensitivity_uids],
        num_scenarios=num_scenarios,
    )
    ...                                          # full-reval engine fills block 0
    write_sensitivity_pnl(request, grid=grid, block_index=1)
    matrix = grid.to_pnl_matrix()                # zero-copy

The declared row set matters more here than anywhere else. Block boundaries are
fixed at allocation, so the compiled book must produce exactly the rows the
caller registered — hence `position_uids` is required rather than optional, and
is checked against the block before a single scenario is priced. A book one row
short would leave the `np.empty` garbage the grid was allocated with sitting in
the output, indistinguishable from a PnL.

A caller holding a `SensitivityPnlCalculator` should call
`calculator.run_into_grid(...)` directly; this function is the one-shot form,
which compiles the book and then does exactly that.
"""

from __future__ import annotations

__all__ = [
    "write_sensitivity_pnl",
]

from riskquant.sensitivity_pnl.grid import PnlGrid
from riskquant.solutions.sensitivity_pnl.calculator import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.errors import DefectLog
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityPnlProvenance,
    SensitivityPnlRequest,
)


def write_sensitivity_pnl(
    request: SensitivityPnlRequest,
    *,
    grid: PnlGrid,
    block_index: int,
) -> SensitivityPnlProvenance:
    """
    Compile the book and price it directly into `grid`'s block `block_index`.

    Returns the provenance stamp; the PnL lands in the caller's grid, so there is
    no matrix to hand back.

    Raises
    ------
    SensitivityPnlInputError
        If the request is malformed, if `position_uids` is absent, if the
        compiled row set does not match the registered block exactly, or if the
        market's scenario count disagrees with the grid's.
    """
    log = DefectLog()
    if request.position_uids is None:
        log.add(
            source="position_uids",
            problem="absent, but required when writing into a caller-owned grid.",
            remedy=(
                "the block's row count is fixed at allocation. Declare the same "
                "ascending UID array registered with allocate_pnl_grid for this "
                "block."
            ),
        )
        log.raise_if_any()

    calculator = SensitivityPnlCalculator.compile(request.book_spec())
    return calculator.run_into_grid(
        request.market, grid=grid, block_index=block_index, book_reused=False
    )
