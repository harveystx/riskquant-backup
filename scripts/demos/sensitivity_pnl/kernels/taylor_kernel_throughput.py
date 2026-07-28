# scripts/demos/sensitivity_pnl/kernels/taylor_kernel_throughput.py
"""
Throughput, work-balance, and peak-memory benchmark for the sensitivity PnL engine.

Reports four things:

  1. Entry-scenario throughput of `kernel_taylor_pnl` across book shapes, and
     its parallel scaling.

  2. prange work balance. Numba slices a prange into contiguous, equal-COUNT
     chunks, so on a heavy-tailed book the scaling ceiling is decided by where
     the heavy rows happen to sit. Measured on a 12-core host BEFORE chunk
     scheduling, identical code and book gave 0.88x (41.83 ms) and 2.13x
     (14.10 ms) on different runs. `report_balance` prints the predicted
     ceiling with and without balanced chunks; the balanced column must NOT
     move between runs, because the partition is deterministic rather than
     sampled from a race. If it moves, the scheduling seam is broken.

  3. Peak resident bytes of the streaming path against the theoretical floor
     (`M_total * S * 4` for the shared grid plus the bounded scratch). The
     ratio must stay at ~1.0; the pre-grid design ran at 2.9x.

  4. Phase attribution. `absorb_displacement` runs TWO kernels: a gather that
     compacts the caller's factor axis into the scratch, and the Taylor kernel
     proper. The gather's cost is fixed at `Ka * B` elements regardless of nnz
     and is pure memory traffic, so it scales differently. `report_phases`
     times them separately and prints the implied Amdahl serial fraction, which
     is the number that says whether a scaling shortfall is a kernel problem or
     a fixed-cost problem. Balanced chunks removed load imbalance as a
     candidate; this identifies what replaced it.

  5. Loop-order regression guard. A register-blocked scenario-tiled variant of
     the Taylor kernel was measured at 0.21-0.30x the entry-major kernel's
     throughput, and a split linear/quadratic CSR layout at 0.58-0.90x. If
     anyone reproposes either, run this first.

Read (1) and (2) together. `bench_shape` reports what the machine actually did
on this run; `report_balance` reports what the schedule permits on any run. A
SKEWED scaling number well below its balanced ceiling means something other
than load imbalance is the limit.
"""

from __future__ import annotations

import time
import tracemalloc

import numba as nb
import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import DISPLACEMENT_ABSOLUTE_ID
from riskquant.sensitivity_pnl.engine import (
    SensitivityPnlEngine,
    numba_thread_ceiling,
)
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.sensitivity_pnl.kernels.gather import kernel_gather_displacement
from riskquant.sensitivity_pnl.kernels.taylor import kernel_taylor_pnl
from riskquant.sensitivity_pnl.scheduling import (
    balanced_chunk_offsets,
    default_num_chunks,
    row_costs,
)

BATCH_SIZE = 500


def build_book(
    *, num_positions: int, num_factors: int, entries: np.ndarray, seed: int
) -> CompiledSensitivityBook:
    rng = np.random.default_rng(seed)
    factor_ids = np.arange(num_factors, dtype=np.int32)
    position_ids = np.repeat(np.arange(1, num_positions + 1, dtype=np.int64), entries)
    picked = np.concatenate(
        [rng.choice(num_factors, n, replace=False) for n in entries]
    ).astype(np.int32)
    sensitivities = pl.DataFrame(
        {
            "position_id": position_ids,
            "factor_id": picked,
            "first_order": rng.normal(0.0, 1e4, position_ids.size),
            "second_order": rng.normal(0.0, 1e6, position_ids.size),
        }
    )
    factor_meta = pl.DataFrame(
        {
            "factor_id": factor_ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    return compile_sensitivity_book(
        sensitivities=sensitivities,
        factor_meta=factor_meta,
        level_universe_factor_ids=factor_ids,
    )


def bench_shape(
    *, label: str, num_positions: int, num_factors: int, entries, seed=3
) -> CompiledSensitivityBook:
    """Time one book shape and hand the compiled book back for `report_balance`."""
    book = build_book(
        num_positions=num_positions,
        num_factors=num_factors,
        entries=entries,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    displacement = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(num_factors, BATCH_SIZE)).astype(np.float32)
    )
    grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=BATCH_SIZE
    )
    engine = SensitivityPnlEngine(
        book=book, out=grid.block_view(0), batch_size=BATCH_SIZE
    )
    engine.absorb_displacement(
        displacement=displacement, batch_len=BATCH_SIZE, scenario_offset=0
    )  # warm JIT

    work = book.nnz + 2 * book.nnz_cross
    max_threads = numba_thread_ceiling()
    original = nb.get_num_threads()
    print(
        f"\n{label}: M={book.num_positions:,}  nnz={book.nnz:,}  "
        f"Ka={book.num_active_factors:,}  chunks={engine.num_chunks:,}"
    )
    baseline = None
    try:
        for threads in sorted({1, max(1, max_threads // 2), max_threads}):
            nb.set_num_threads(threads)
            reps = 6
            start = time.perf_counter()
            for _ in range(reps):
                engine.absorb_displacement(
                    displacement=displacement,
                    batch_len=BATCH_SIZE,
                    scenario_offset=0,
                )
            elapsed = (time.perf_counter() - start) / reps
            rate = work * BATCH_SIZE / elapsed / 1e9
            baseline = baseline or elapsed
            print(
                f"  threads={threads:>3}  {elapsed * 1e3:8.2f} ms/batch  "
                f"{rate:6.3f} G entry-scenario/s  scaling={baseline / elapsed:5.2f}x"
            )
    finally:
        nb.set_num_threads(original)

    return book


def simulated_thread_loads(cost: np.ndarray, num_threads: int) -> np.ndarray:
    """
    Replicate numba's contiguous equal-count prange partition. Verified:

        T=4, n=48  ->  000000000000111111111111222222222222333333333333

    Simulating it makes the balance report a property of the SCHEDULE rather
    than of this run, so it reproduces on a single-core CI box.
    """
    base, remainder = divmod(cost.shape[0], num_threads)
    loads, start = [], 0
    for t in range(num_threads):
        length = base + (1 if t < remainder else 0)
        loads.append(cost[start : start + length].sum())
        start += length
    return np.array(loads)


def report_balance(
    *,
    label: str,
    book: CompiledSensitivityBook,
    thread_counts=(2, 4, 8, 12, 24),
) -> None:
    """
    Predicted scaling ceiling = mean thread load / max thread load.

    `attainable` is the ceiling the schedule COULD reach: a row is the atom of
    work, so a thread can overshoot the mean by at most one row's cost. When
    `balanced` tracks `attainable`, the partition is doing everything available
    to it and the residual gap is book granularity, not a scheduling defect.
    """
    cost = row_costs(book)
    print(f"\n{label}: work balance  (1.00 = perfect; deterministic, not sampled)")
    print(
        f"  {'threads':>8} | {'chunks':>7} | {'unbalanced':>11} | "
        f"{'balanced':>9} | {'attainable':>11}"
    )
    for threads in thread_counts:
        unbalanced = simulated_thread_loads(cost, threads)
        offsets = balanced_chunk_offsets(
            book=book,
            num_chunks=default_num_chunks(
                num_positions=book.num_positions, thread_ceiling=threads
            ),
        )
        chunk_cost = np.array(
            [
                cost[offsets[i] : offsets[i + 1]].sum()
                for i in range(offsets.shape[0] - 1)
            ]
        )
        balanced = simulated_thread_loads(chunk_cost, min(threads, chunk_cost.shape[0]))
        mean_thread_load = cost.sum() / threads
        attainable = min(1.0, mean_thread_load / (mean_thread_load + cost.max()))
        print(
            f"  {threads:>8} | {offsets.shape[0] - 1:>7} | "
            f"{unbalanced.mean() / unbalanced.max():>11.2f} | "
            f"{balanced.mean() / balanced.max():>9.2f} | {attainable:>11.2f}"
        )


def report_phases(*, label, book, num_factors, thread_counts):
    """
    Split `absorb_displacement` into its two kernels and scale each separately.

    `serial fraction` is Amdahl solved for f from the observed speedup S at N
    threads: f = (N/S - 1) / (N - 1). Read it against the phase split — if the
    gather is a large fixed share, f is telling you about the gather, not about
    the Taylor kernel.
    """
    rng = np.random.default_rng(101)
    displacement = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(num_factors, BATCH_SIZE)).astype(np.float32)
    )
    grid = allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=BATCH_SIZE
    )
    engine = SensitivityPnlEngine(
        book=book, out=grid.block_view(0), batch_size=BATCH_SIZE
    )

    def run_gather():
        kernel_gather_displacement(
            displacement, book.active_level_uix, BATCH_SIZE, 0, engine._displacement
        )

    def run_taylor():
        kernel_taylor_pnl(
            engine._chunk_offsets,
            book.position_offsets,
            book.factor_uix,
            book.first_order,
            book.second_order,
            book.cross_offsets,
            book.cross_factor_uix_i,
            book.cross_factor_uix_j,
            book.cross_second_order,
            engine._displacement,
            engine._accumulator,
            BATCH_SIZE,
            0,
            engine.out,
        )

    def time_it(fn, reps=6):
        fn()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - start) / reps

    displacement_bytes = book.nnz * BATCH_SIZE * 4
    original = nb.get_num_threads()
    print(f"\n{label}: phase attribution")
    print(
        f"  {'threads':>8} | {'gather':>9} | {'taylor':>9} | {'total':>9} | "
        f"{'gather %':>9} | {'disp read':>11} | {'serial f':>9}"
    )
    base_total = None
    try:
        for threads in thread_counts:
            nb.set_num_threads(threads)
            gather = time_it(run_gather)
            taylor = time_it(run_taylor)
            total = gather + taylor
            base_total = base_total or total
            speedup = base_total / total
            serial = (threads / speedup - 1.0) / (threads - 1.0) if threads > 1 else 0.0
            print(
                f"  {threads:>8} | {gather * 1e3:>7.2f}ms | {taylor * 1e3:>7.2f}ms | "
                f"{total * 1e3:>7.2f}ms | {100 * gather / total:>8.1f}% | "
                f"{displacement_bytes / taylor / 1e9:>8.1f} GB/s | {serial:>9.3f}"
            )
    finally:
        nb.set_num_threads(original)


def bench_peak_memory(*, num_positions=20_000, num_factors=2_000, num_scenarios=4_000):
    """Streaming peak vs the floor. Ratio must stay near 1.0."""
    book = build_book(
        num_positions=num_positions,
        num_factors=num_factors,
        entries=np.full(num_positions, 12),
        seed=11,
    )
    rng = np.random.default_rng(12)
    displacement = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(num_factors, BATCH_SIZE)).astype(np.float32)
    )
    full_reval_uids = np.arange(10**9, 10**9 + num_positions, dtype=np.int64)

    tracemalloc.start()
    grid = allocate_pnl_grid(
        block_position_uids=[full_reval_uids, book.position_uids],
        num_scenarios=num_scenarios,
    )
    engine = SensitivityPnlEngine(
        book=book, out=grid.block_view(1), batch_size=BATCH_SIZE
    )
    for offset in range(0, num_scenarios, BATCH_SIZE):
        engine.absorb_displacement(
            displacement=displacement,
            batch_len=min(BATCH_SIZE, num_scenarios - offset),
            scenario_offset=offset,
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    m_total = 2 * num_positions
    floor = (
        m_total * num_scenarios * 4
        + book.num_active_factors * BATCH_SIZE * 4
        + numba_thread_ceiling() * BATCH_SIZE * 8
        + (engine.num_chunks + 1) * 4
    )
    mb = 1024**2
    print(
        f"\nPeak memory  M_total={m_total:,} S={num_scenarios:,}\n"
        f"  measured peak : {peak / mb:8.1f} MB\n"
        f"  theoretical   : {floor / mb:8.1f} MB\n"
        f"  ratio         : {peak / floor:8.2f}x"
    )


def run_demo() -> None:
    print("\n" + "=" * 108)
    print(f"{'SENSITIVITY PnL — TAYLOR KERNEL THROUGHPUT':^108}")
    print("=" * 108 + "\n")
    print(f"NUMBA_NUM_THREADS={numba_thread_ceiling()}  batch_size={BATCH_SIZE}")

    uniform_book = bench_shape(
        label="UNIFORM  (12 entries/position)",
        num_positions=20_000,
        num_factors=2_000,
        entries=np.full(20_000, 12),
    )
    bench_shape(
        label="WIDE     (40 entries/position)",
        num_positions=20_000,
        num_factors=2_000,
        entries=np.full(20_000, 40),
    )
    bench_shape(
        label="SPARSE   (4 entries/position)",
        num_positions=200_000,
        num_factors=2_000,
        entries=np.full(200_000, 4),
    )
    skew = np.full(20_000, 4)
    skew[: 20_000 // 20] = 80  # 5% of positions carry ~50% of the entries
    skewed_book = bench_shape(
        label="SKEWED   (5% carry half the nnz — prange load-imbalance probe)",
        num_positions=20_000,
        num_factors=2_000,
        entries=skew,
    )

    report_balance(label="SKEWED", book=skewed_book)
    report_balance(label="UNIFORM", book=uniform_book)

    max_threads = numba_thread_ceiling()
    thread_counts = sorted(
        {t for t in (1, 2, max_threads // 2, max_threads) if 1 <= t <= max_threads}
    )
    report_phases(
        label="UNIFORM",
        book=uniform_book,
        num_factors=2_000,
        thread_counts=thread_counts,
    )
    report_phases(
        label="SKEWED",
        book=skewed_book,
        num_factors=2_000,
        thread_counts=thread_counts,
    )

    bench_peak_memory()
    print("\n" + "=" * 108 + "\n")


if __name__ == "__main__":
    run_demo()
