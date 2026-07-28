# src/riskquant/sensitivity_pnl/scheduling.py
"""
Work-balanced chunk scheduling for the Taylor kernel's `prange`.

The problem
-----------
Numba's parallel backend partitions a `prange` into CONTIGUOUS, EQUAL-COUNT
chunks. Verified empirically:

    T=4, n=48  ->  000000000000111111111111222222222222333333333333

Equal COUNT is not equal WORK. A CSR row's cost is proportional to its entry
count, and real sensitivity books are heavy-tailed: a handful of exotics carry
tens of factors while the long tail carries three or four. Whether the engine
scales therefore depends on where those heavy rows happen to sit in the row
order — which is an artefact of the upstream feed, not a property anyone
controls. Measured on a 12-core host, the same code and the same book gave
0.88x and 2.13x scaling on different runs.

Why sorting by nnz is the WRONG fix
-----------------------------------
It is the obvious move and it is catastrophic. Because chunks are contiguous,
sorting descending puts every heavy row in chunk 0 and starves the rest.
Predicted scaling ceiling (mean thread load / max thread load) on the SKEWED
shape at 12 threads:

    natural order        0.17     (heavy rows contiguous)
    sorted descending    0.17
    natural order        0.93     (heavy rows shuffled)
    sorted descending    0.17     <-- actively WORSE than doing nothing
    balanced chunks      1.00

A low-discrepancy (bit-reversal) permutation does work — 0.99 — but it is
statistical, it degrades on heavy tails (0.93 on a power-law book against 1.00
for chunking), and it destroys the ascending `position_uids` ordering that
`PnlGrid` and every downstream `searchsorted` join depend on. Rejected.

The fix
-------
Leave the row order alone and change what `prange` iterates over. Cut the row
axis into contiguous ranges of ~equal COST rather than ~equal count, and range
over those. Each chunk carries `total_cost / num_chunks` by construction, so
any contiguous group of chunks — which is exactly what numba hands a thread —
carries a proportional share. The balance is structural, not probabilistic.

`num_chunks` oversubscribes the thread count (default 8x) so that lowering
threads at runtime via `set_num_threads` still lands whole chunks on every
thread.

Cost model
----------
    cost(m) = nnz(m) + 2 * nnz_cross(m) + PER_ROW_SCATTER_COST

A cross entry dereferences two displacement rows, hence the 2. Every row pays
one additional `batch_len` pass regardless of its entries — the fused
narrow-cast scatter at the tail of the kernel body — hence the constant, which
also guarantees `cost > 0` for every row and keeps the partition well defined
for a book of entirely empty positions.

Known limit
-----------
A row is the atom of work; it cannot be split without cross-thread reduction on
the output row. A single position holding 30% of the book's entries caps
scaling at ~0.33 at 12 threads no matter how the chunks are cut. Chunking does
not fix that and does not claim to; splitting hot rows across threads would
require an atomic accumulate and is a separate design.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CHUNKS_PER_THREAD",
    "PER_ROW_SCATTER_COST",
    "balanced_chunk_offsets",
    "default_num_chunks",
    "row_costs",
]

import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook

#: Fixed per-row cost in units of "one entry pass": the narrow-cast scatter runs
#: one `batch_len` loop per row whatever the row contains.
PER_ROW_SCATTER_COST: float = 1.0

#: Chunks per thread. Oversubscription absorbs the granularity loss from rows
#: being unsplittable and keeps the partition valid when `set_num_threads`
#: lowers the thread count below the ceiling the chunks were cut for.
DEFAULT_CHUNKS_PER_THREAD: int = 8


def row_costs(book: CompiledSensitivityBook) -> npt.NDArray[np.float64]:
    """Per-position kernel cost in units of one entry-scenario pass."""
    diagonal = np.diff(book.position_offsets).astype(np.float64)
    cross = np.diff(book.cross_offsets).astype(np.float64)
    return diagonal + 2.0 * cross + PER_ROW_SCATTER_COST


def default_num_chunks(
    *,
    num_positions: int,
    thread_ceiling: int,
    chunks_per_thread: int = DEFAULT_CHUNKS_PER_THREAD,
) -> int:
    """Chunk count for a machine, capped at one chunk per position."""
    if chunks_per_thread <= 0:
        raise ValueError("chunks_per_thread must be positive")
    if num_positions <= 0:
        return 0
    return int(min(num_positions, max(1, thread_ceiling) * chunks_per_thread))


def balanced_chunk_offsets(
    *, book: CompiledSensitivityBook, num_chunks: int
) -> npt.NDArray[np.int32]:
    """
    Contiguous row ranges of approximately equal cost.

    Returns `(C+1,)` int32 row pointers, strictly increasing, starting at 0 and
    terminating at M. `C <= num_chunks`: cuts that collide on a single very
    heavy row are merged rather than emitted as empty chunks, since an empty
    chunk would consume a `prange` slot for no work.

    The partition is `searchsorted` against the cost prefix sum — O(M + C log M),
    computed once per engine, never on the hot path.
    """
    num_positions = book.num_positions
    if num_positions == 0:
        return np.zeros(1, dtype=np.int32)
    if num_chunks <= 1:
        return np.array([0, num_positions], dtype=np.int32)

    cost = row_costs(book)
    cumulative = np.empty(num_positions + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(cost, out=cumulative[1:])

    targets = cumulative[-1] * np.arange(1, num_chunks, dtype=np.float64) / num_chunks
    cuts = np.searchsorted(cumulative, targets, side="left")

    bounds = np.unique(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                cuts.astype(np.int64, copy=False),
                np.full(1, num_positions, dtype=np.int64),
            )
        )
    )
    return np.ascontiguousarray(bounds, dtype=np.int32)
