# src/riskquant/sensitivity_pnl/engine.py
"""
Sensitivity PnL orchestrator.

The engine OWNS NO OUTPUT. It writes into a `(M, S)` float32 C-contiguous view
injected by the caller — normally a `PnlGrid.block_view(i)` sitting alongside
the full-reval block in one allocation. Its own scratch is bounded by
`batch_size`, never by `num_scenarios`:

    displacement   (Ka, B)          f32   -- 4 MB at Ka=2000, B=500
    accumulator    (n_threads, Bp)  f64   -- 64 KB at 16 threads, B=500

`Bp` is `batch_size` padded so that consecutive threads' rows cannot share a
cache line. See `_ACCUMULATOR_ROW_PAD_ELEMENTS`.

Both are allocated once in `__post_init__` and only ever overwritten. Nothing on
the hot path allocates: `absorb_*` performs shape checks against pre-computed
scalars and then makes two kernel calls.

Two input contracts
-------------------
`absorb_levels`       — INTEGRATED path. Consumes the same `(N_req, B)` f64
                        shocked-levels buffer the full-reval runner fills, and
                        derives displacement per factor from the declared
                        convention. Both PnL blocks are then driven by one
                        market state; if the sensitivity block saw different
                        draws, the joint tail would be a fiction.

`absorb_displacement` — STANDALONE path. Consumes `(K, S)` f32 displacements
                        directly, for a caller who holds a shock matrix already
                        expressed in the book's declared space and no level data
                        at all. f32 matches `ScenarioSetBatch.shocks`, so there
                        is no conversion at the boundary. Nothing in this path
                        touches `riskquant.simlab`.

Both feed the identical Taylor kernel; they differ only in how the `(Ka, B)`
displacement scratch is filled.

Bounds discipline
-----------------
The kernels run under `boundscheck=False` and add no guards. Index safety is
split cleanly: `CompiledSensitivityBook.__post_init__` owns everything internal
to the book (CSR monotonicity, factor indices against Ka), and this class owns
everything that depends on caller-supplied buffers (levels-matrix height,
batch and scenario offsets, dtype, contiguity). Between them every dereference
in every kernel is covered.
"""

from __future__ import annotations

__all__ = [
    "SensitivityPnlEngine",
    "accumulator_row_width",
    "compute_sensitivity_pnl_from_displacement",
    "compute_sensitivity_pnl_from_levels",
    "numba_thread_ceiling",
]

from dataclasses import dataclass, field
from typing import Any

import numba as nb  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import RATIO_DISPLACEMENT_IDS
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.sensitivity_pnl.kernels.displacement import kernel_build_displacement
from riskquant.sensitivity_pnl.kernels.gather import kernel_gather_displacement
from riskquant.sensitivity_pnl.kernels.taylor import kernel_taylor_pnl
from riskquant.sensitivity_pnl.scheduling import (
    DEFAULT_CHUNKS_PER_THREAD,
    balanced_chunk_offsets,
    default_num_chunks,
)

#: Dead float64 slots appended to every accumulator row so that thread t's last
#: written byte and thread t+1's first written byte are at least one cache line
#: apart. Without it the rows are packed end to end and 11 of 12 thread
#: boundaries land inside a shared line, which the innermost loop then writes on
#: every accumulator pass — the textbook false-sharing pattern, invisible on one
#: core and a coherence storm on twelve.
#:
#: Rounding the row STRIDE to a 64-byte multiple is NOT sufficient: it only
#: works when the array's base address is itself 64-byte aligned, and numpy
#: guarantees no such thing. Measured across base offsets 0..56 the
#: stride-rounded layout still shared 11 of 11 boundaries at offsets 40, 48 and
#: 56, whereas an explicit gap shared none at any offset. Worse, the base
#: alignment numpy hands back varies between allocations, so the unpadded
#: layout is a RUN-TO-RUN source of variance, not a constant tax.
_ACCUMULATOR_ROW_PAD_ELEMENTS: int = 8  # 8 * 8 B = one 64-byte cache line


def accumulator_row_width(batch_size: int) -> int:
    """Padded accumulator row width in float64 elements."""
    aligned = (
        (batch_size + _ACCUMULATOR_ROW_PAD_ELEMENTS - 1)
        // _ACCUMULATOR_ROW_PAD_ELEMENTS
    ) * _ACCUMULATOR_ROW_PAD_ELEMENTS
    return aligned + _ACCUMULATOR_ROW_PAD_ELEMENTS


def numba_thread_ceiling() -> int:
    """
    Hard upper bound on `nb.get_thread_id()`; `set_num_threads` cannot exceed it.

    Wrapped because `numba.config` carries no type stubs, so the attribute read
    is the one place that needs a suppression rather than every call site.
    """
    return int(nb.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True, kw_only=True)
class SensitivityPnlEngine:
    """Immutable compute topology over a bounded, pre-allocated scratch arena."""

    book: CompiledSensitivityBook
    out: npt.NDArray[np.float32]
    batch_size: int = 500

    #: `prange` chunks per thread. Higher oversubscription tightens the balance
    #: on heavy-tailed books at the cost of a slightly longer chunk_offsets
    #: array; 8 measures at a 1.00 predicted ceiling on every realistic shape.
    chunks_per_thread: int = DEFAULT_CHUNKS_PER_THREAD

    #: Explicit chunk boundaries, overriding the balanced partition. For tests
    #: and for callers pinning a schedule across machines; leave unset.
    chunk_offsets: npt.NDArray[np.int32] | None = None

    #: Required by `absorb_levels` only; the standalone displacement path never
    #: reads it. Defaults to empty so a caller with no level data is not forced
    #: to invent one.
    base_levels: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    _chunk_offsets: npt.NDArray[np.int32] = field(init=False, repr=False)
    _displacement: npt.NDArray[np.float32] = field(init=False, repr=False)
    _accumulator: npt.NDArray[np.float64] = field(init=False, repr=False)
    _min_source_rows: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        out = self.out
        if out.dtype != np.float32 or out.ndim != 2:
            raise ValueError("out must be a 2-D float32 view")
        if not out.flags.c_contiguous:
            raise ValueError(
                "out must be C-contiguous. Row slices of a C-contiguous grid "
                "are C-contiguous; column slices are not."
            )
        if out.shape[0] != self.book.num_positions:
            raise ValueError(
                f"out has {out.shape[0]} rows; book has "
                f"{self.book.num_positions} positions"
            )

        # Highest levels/displacement row any kernel will dereference. Derived
        # from the active-factor axis, NOT from nnz: a cross-only book has
        # nnz == 0 and a fully populated active-factor axis.
        object.__setattr__(self, "_min_source_rows", self.book.max_level_uix + 1)

        if self.base_levels.size:
            self._validate_base_levels()

        object.__setattr__(self, "_chunk_offsets", self._resolve_chunk_offsets())

        object.__setattr__(
            self,
            "_displacement",
            np.empty(
                (self.book.num_active_factors, self.batch_size),
                dtype=np.float32,
                order="C",
            ),
        )
        # Rows sized at the ceiling `set_num_threads` cannot exceed, so
        # `get_thread_id()` can never over-index it mid-run; row WIDTH padded so
        # that no two threads write into the same cache line.
        object.__setattr__(
            self,
            "_accumulator",
            np.empty(
                (numba_thread_ceiling(), accumulator_row_width(self.batch_size)),
                dtype=np.float64,
                order="C",
            ),
        )

    @property
    def displacement_scratch(self) -> npt.NDArray[np.float32]:
        """
        Read-only view of the `(Ka, batch_size)` displacement scratch.

        Holds the LAST batch absorbed, so it is a sample, not a record of the
        run. Exposed for diagnostics — Taylor-strain reporting, tail
        attribution against factor moves — rather than for callers to write
        into; mutating it corrupts the next batch.
        """
        view = self._displacement.view()
        view.flags.writeable = False
        return view

    @property
    def num_chunks(self) -> int:
        """`prange` slots the Taylor kernel will range over. Diagnostic only."""
        return int(self._chunk_offsets.shape[0]) - 1

    def _resolve_chunk_offsets(self) -> npt.NDArray[np.int32]:
        """
        Work-balanced `prange` partition, computed once per engine.

        Deliberately an ENGINE concern, not a compile-time one. The partition
        depends on `NUMBA_NUM_THREADS`, which is a property of the host, not of
        the book; baking it into `CompiledSensitivityBook` would mean a book
        compiled on a 12-core box carried the wrong schedule when loaded on a
        64-core one. The engine already owns the other host-shaped allocation
        (the per-thread accumulator), so the seam belongs here.
        """
        if self.chunk_offsets is None:
            return balanced_chunk_offsets(
                book=self.book,
                num_chunks=default_num_chunks(
                    num_positions=self.book.num_positions,
                    thread_ceiling=numba_thread_ceiling(),
                    chunks_per_thread=self.chunks_per_thread,
                ),
            )

        offsets = self.chunk_offsets
        if offsets.dtype != np.int32 or offsets.ndim != 1 or offsets.shape[0] < 1:
            raise ValueError("chunk_offsets must be a non-empty 1-D int32 array")
        if not offsets.flags.c_contiguous:
            raise ValueError("chunk_offsets must be C-contiguous")
        if int(offsets[0]) != 0:
            raise ValueError("chunk_offsets must start at 0")
        if offsets.shape[0] > 1 and not bool(np.all(np.diff(offsets) > 0)):
            raise ValueError(
                "chunk_offsets must be strictly increasing — an empty chunk "
                "consumes a prange slot for no work"
            )
        if int(offsets[-1]) != self.book.num_positions:
            raise ValueError(
                f"chunk_offsets terminate at {int(offsets[-1])}, expected "
                f"M={self.book.num_positions}; rows outside the partition would "
                "never be written and would leak np.empty garbage into the grid"
            )
        return offsets

    def _validate_base_levels(self) -> None:
        levels = self.base_levels
        if levels.dtype != np.float64 or levels.ndim != 1:
            raise ValueError("base_levels must be 1-D float64")
        if not levels.flags.c_contiguous:
            raise ValueError("base_levels must be C-contiguous")
        if levels.shape[0] < self._min_source_rows:
            raise ValueError(
                f"base_levels has {levels.shape[0]} rows; the book dereferences "
                f"row {self._min_source_rows - 1}"
            )

        # RELATIVE / LOG displacements divide by the base level.
        # `error_model='numpy'` emits inf rather than raising, so the gate is here.
        ratio_mask = np.isin(
            self.book.displacement_convention_ids,
            np.asarray(RATIO_DISPLACEMENT_IDS, dtype=np.int8),
        )
        if not ratio_mask.any():
            return
        bases = levels[self.book.active_level_uix[ratio_mask]]
        valid = np.isfinite(bases) & (bases > 0.0)
        if not bool(valid.all()):
            bad = self.book.active_factor_ids[ratio_mask][~valid]
            raise ValueError(
                "RELATIVE / LOG displacement requires a finite, strictly "
                f"positive base level; violated by factor_id(s) {bad[:10].tolist()}"
            )

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def absorb_levels(
        self,
        *,
        shocked_levels: npt.NDArray[np.float64],
        batch_len: int,
        scenario_offset: int,
        levels_offset: int = 0,
    ) -> None:
        """
        Integrated path: one scenario batch of shocked levels into `out`.

        `levels_offset` lets a caller pass a full-width `(N_req, S)` levels
        matrix instead of a per-batch buffer; a column slice would not be
        C-contiguous and would break the pinned kernel signature.
        """
        if self.base_levels.size == 0:
            raise ValueError(
                "absorb_levels requires base_levels; construct the engine with "
                "it, or use absorb_displacement for the standalone path."
            )
        self._check_source(
            source=shocked_levels,
            dtype=np.float64,
            source_offset=levels_offset,
            batch_len=batch_len,
            scenario_offset=scenario_offset,
            name="shocked_levels",
        )
        if shocked_levels.shape[0] != self.base_levels.shape[0]:
            raise ValueError(
                f"shocked_levels has {shocked_levels.shape[0]} factor rows; "
                f"base_levels has {self.base_levels.shape[0]}"
            )
        kernel_build_displacement(
            self.base_levels,
            shocked_levels,
            self.book.active_level_uix,
            self.book.displacement_convention_ids,
            batch_len,
            levels_offset,
            self._displacement,
        )
        self._run_taylor(batch_len=batch_len, scenario_offset=scenario_offset)

    def absorb_displacement(
        self,
        *,
        displacement: npt.NDArray[np.float32],
        batch_len: int,
        scenario_offset: int,
        source_offset: int = 0,
    ) -> None:
        """
        Standalone path: one scenario batch of pre-computed displacements.

        `displacement` is `(K, S)` C-contiguous float32 over the SAME factor axis
        the book was compiled against, already expressed in each factor's
        declared `DisplacementConvention`. No base levels are read or required.
        """
        self._check_source(
            source=displacement,
            dtype=np.float32,
            source_offset=source_offset,
            batch_len=batch_len,
            scenario_offset=scenario_offset,
            name="displacement",
        )
        kernel_gather_displacement(
            displacement,
            self.book.active_level_uix,
            batch_len,
            source_offset,
            self._displacement,
        )
        self._run_taylor(batch_len=batch_len, scenario_offset=scenario_offset)

    def _run_taylor(self, *, batch_len: int, scenario_offset: int) -> None:
        kernel_taylor_pnl(
            self._chunk_offsets,
            self.book.position_offsets,
            self.book.factor_uix,
            self.book.first_order,
            self.book.second_order,
            self.book.cross_offsets,
            self.book.cross_factor_uix_i,
            self.book.cross_factor_uix_j,
            self.book.cross_second_order,
            self._displacement,
            self._accumulator,
            batch_len,
            scenario_offset,
            self.out,
        )

    def _check_source(
        self,
        *,
        source: npt.NDArray[np.floating[Any]],
        dtype: type[np.floating[Any]],
        source_offset: int,
        batch_len: int,
        scenario_offset: int,
        name: str,
    ) -> None:
        """
        The complete bounds gate for caller-supplied buffers. Everything the
        kernels index into a caller array is bounded here; there are no checks
        downstream.
        """
        if source.ndim != 2:
            raise ValueError(f"{name} must be 2-D")
        if source.dtype != dtype:
            raise ValueError(f"{name} must be {np.dtype(dtype).name}")
        if not source.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if source.shape[0] < self._min_source_rows:
            raise ValueError(
                f"{name} has {source.shape[0]} factor rows; the book dereferences "
                f"row {self._min_source_rows - 1}. Pass the matrix over the SAME "
                "factor axis the book was compiled against."
            )
        if not 0 < batch_len <= self.batch_size:
            raise ValueError(f"batch_len must lie in (0, {self.batch_size}]")
        if source_offset < 0 or source_offset + batch_len > source.shape[1]:
            raise ValueError(f"batch overruns the {name} scenario axis")
        if scenario_offset < 0 or scenario_offset + batch_len > self.out.shape[1]:
            raise ValueError("batch overruns the out scenario axis")


# =========================================================================
# ONE-SHOT ENTRY POINTS
# =========================================================================


def compute_sensitivity_pnl_from_levels(
    *,
    book: CompiledSensitivityBook,
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    batch_size: int = 500,
) -> npt.NDArray[np.float32]:
    """Integrated one-shot: `(N_req, S)` f64 shocked levels -> `(M, S)` f32 PnL."""
    num_scenarios = shocked_levels.shape[1]
    grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=num_scenarios
    )
    engine = SensitivityPnlEngine(
        book=book,
        out=grid.block_view(0),
        base_levels=base_levels,
        batch_size=min(batch_size, num_scenarios),
    )
    for offset, length in _batches(num_scenarios, engine.batch_size):
        engine.absorb_levels(
            shocked_levels=shocked_levels,
            batch_len=length,
            scenario_offset=offset,
            levels_offset=offset,
        )
    return grid.pnl


def compute_sensitivity_pnl_from_displacement(
    *,
    book: CompiledSensitivityBook,
    displacement: npt.NDArray[np.float32],
    batch_size: int = 500,
) -> npt.NDArray[np.float32]:
    """Standalone one-shot: `(K, S)` f32 displacements -> `(M, S)` f32 PnL."""
    num_scenarios = displacement.shape[1]
    grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=num_scenarios
    )
    engine = SensitivityPnlEngine(
        book=book,
        out=grid.block_view(0),
        batch_size=min(batch_size, num_scenarios),
    )
    for offset, length in _batches(num_scenarios, engine.batch_size):
        engine.absorb_displacement(
            displacement=displacement,
            batch_len=length,
            scenario_offset=offset,
            source_offset=offset,
        )
    return grid.pnl


def _batches(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Edge-side batch plan; not hot-path."""
    return [
        (offset, min(batch_size, total - offset))
        for offset in range(0, total, batch_size)
    ]
