# src/riskquant/sensitivity_pnl/grid.py
"""
Shared PnL grid: one allocation, disjoint row-slice views.

Replaces the previous `attach_pnl` concatenate. Concatenating a full-reval block
and a sensitivity block allocates a third `(M_total, S)` array while both
sources are still live; at `M_fr = M_sb = 50k, S = 10k` that path peaked at
11.2 GB against an achievable 3.9 GB.

Here the grid is allocated ONCE and each engine is handed a row-slice view to
write into. A row slice of a C-contiguous 2-D array is itself C-contiguous, so
the views satisfy the pinned kernel signatures with no copy and no re-layout.

    grid = allocate_pnl_grid(block_position_uids=[fr_uids, sb_uids],
                             num_scenarios=S)
    full_reval_view = grid.block_view(0)     # (M_fr, S) f32 C-contig
    sensitivity_view = grid.block_view(1)    # (M_sb, S) f32 C-contig
    ...                                      # both engines write in place
    pnl_matrix = grid.to_pnl_matrix()        # zero-copy

Disjointness is enforced at ALLOCATION time, not after the run. A position
priced by the full-reval engine and also carried in the sensitivity book would
be double-counted in every tail statistic; discovering that after a two-hour
streaming loop is a waste the check has no reason to impose.
"""

from __future__ import annotations

__all__ = [
    "PnlGrid",
    "allocate_pnl_grid",
]

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.pnl.results import PnlMatrix


@dataclass(frozen=True, slots=True, kw_only=True)
class PnlGrid:
    """
    One `(M_total, S)` float32 C-contiguous grid partitioned into row blocks.

    `pnl` is the sole owner of the memory. `block_offsets` are row pointers, so
    `block_view(i)` is a view, never a copy.
    """

    trade_uids: npt.NDArray[np.int64]
    pnl: npt.NDArray[np.float32]
    block_offsets: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        if self.pnl.dtype != np.float32:
            raise ValueError("PnlGrid: pnl must be float32")
        if not self.pnl.flags.c_contiguous:
            raise ValueError("PnlGrid: pnl must be C-contiguous")
        if self.trade_uids.shape[0] != self.pnl.shape[0]:
            raise ValueError("PnlGrid: trade_uids / pnl row mismatch")
        if int(self.block_offsets[-1]) != self.pnl.shape[0]:
            raise ValueError("PnlGrid: block_offsets do not terminate at M_total")

    @property
    def num_blocks(self) -> int:
        return int(self.block_offsets.shape[0]) - 1

    @property
    def num_scenarios(self) -> int:
        return int(self.pnl.shape[1])

    def block_view(self, block_index: int) -> npt.NDArray[np.float32]:
        """C-contiguous `(M_block, S)` view of block `block_index`."""
        if not 0 <= block_index < self.num_blocks:
            raise IndexError(
                f"block_index {block_index} outside [0, {self.num_blocks})"
            )
        start = int(self.block_offsets[block_index])
        stop = int(self.block_offsets[block_index + 1])
        return self.pnl[start:stop]

    def block_uids(self, block_index: int) -> npt.NDArray[np.int64]:
        """Row-aligned UID view for block `block_index`."""
        if not 0 <= block_index < self.num_blocks:
            raise IndexError(
                f"block_index {block_index} outside [0, {self.num_blocks})"
            )
        start = int(self.block_offsets[block_index])
        stop = int(self.block_offsets[block_index + 1])
        return self.trade_uids[start:stop]

    def to_pnl_matrix(self) -> PnlMatrix:
        """Zero-copy handoff to the canonical result container."""
        return PnlMatrix(trade_uids=self.trade_uids, pnl=self.pnl)


def allocate_pnl_grid(
    *,
    block_position_uids: Sequence[npt.NDArray[np.int64]],
    num_scenarios: int,
) -> PnlGrid:
    """
    Allocate the shared grid and validate block disjointness up front.

    Row order within a block is the caller's UID order, preserved verbatim —
    each engine must emit rows in the order of the UID array it registered here.
    `CompiledSensitivityBook.position_uids` is ascending, so passing it directly
    is correct.

    Raises
    ------
    ValueError
        On an empty block list, a non-int64 UID array, duplicate UIDs within a
        block, or any UID shared across blocks.
    """
    if not block_position_uids:
        raise ValueError("allocate_pnl_grid requires at least one block")
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    offsets = np.empty(len(block_position_uids) + 1, dtype=np.int32)
    offsets[0] = 0
    for index, uids in enumerate(block_position_uids):
        if uids.dtype != np.int64 or uids.ndim != 1:
            raise ValueError(f"block {index}: position UIDs must be 1-D int64")
        offsets[index + 1] = offsets[index] + uids.shape[0]

    trade_uids = np.ascontiguousarray(np.concatenate(block_position_uids))
    unique_uids, counts = np.unique(trade_uids, return_counts=True)
    if unique_uids.shape[0] != trade_uids.shape[0]:
        collisions = unique_uids[counts > 1]
        raise ValueError(
            f"{collisions.shape[0]} position UID(s) are duplicated across or "
            f"within PnL blocks and would be double-counted. "
            f"First 10: {collisions[:10].tolist()}"
        )

    return PnlGrid(
        trade_uids=trade_uids,
        pnl=np.empty((trade_uids.shape[0], num_scenarios), dtype=np.float32, order="C"),
        block_offsets=offsets,
    )
