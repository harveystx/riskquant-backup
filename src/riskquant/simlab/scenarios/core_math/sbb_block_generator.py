# src/riskquant/simlab/scenarios/core_math/sbb_block_generator.py
"""
Stationary Block Bootstrap (SBB) path-index generator.

Produces a `(B, h)` int32 matrix of historical timestep indices, each
row a synthetic h-day path stitched from variable-length blocks of
contiguous days. Block lengths are drawn from a Geometric distribution
parameterised by `expected_block_size`.

The previous implementation did the path stitching in pure Python
(loop over B scenarios with a nested while-step loop, allocating a
small `np.arange` per block). For B*h on the order of 1e5 that was
~100k Python iterations per batch. The refactor njits the stitching
kernel and feeds it the pre-drawn block lengths + starting indices;
the kernel is fully zero-allocation past its `out` parameter.

The RNG draws still happen in NumPy (single-call rng.geometric and
rng.integers) — Numba doesn't share NumPy's modern bit-generator state,
so doing it in Python keeps determinism per-seed and avoids a second
RNG state to manage.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]
import numpy as np

from riskquant.jit import parallel_njit

_I32_1D = nb.int32[::1]
_I32_2D_C = nb.int32[:, ::1]


@parallel_njit(
    nb.void(
        _I32_1D,  # block_lengths        (max_blocks,)
        _I32_1D,  # start_indices        (max_blocks,)
        _I32_2D_C,  # out                  (B, h)  C-contig
        nb.int32,  # max_valid_length     T - 1
    ),
)
def _kernel_stitch_sbb_paths(
    block_lengths,
    start_indices,
    out,
    max_valid_length,
) -> None:
    """
    Stitch B independent h-day paths in place into `out`.

    Each scenario b consumes a prefix of `block_lengths` / `start_indices`
    until it has filled h timesteps. The consumption is path-local so
    different scenarios can advance through the draw pool at different
    rates — we cannot prange over b without partitioning the draw pool
    upfront. We pre-compute the per-scenario starting offset into the
    draw pool with a serial cumulative count, then parallelise the
    stitching.
    """
    b_blocks = out.shape[0]
    h = out.shape[1]

    # ---- Serial pre-scan: per-scenario draw-pool offset. ----
    # `block_offset[b]` is the index of the first draw belonging to
    # scenario b. We use the same geometric draws every time, so this
    # is deterministic.
    block_offset = np.empty(b_blocks, dtype=np.int32)
    cursor = 0
    for b in range(b_blocks):
        block_offset[b] = cursor
        step = 0
        while step < h:
            b_len = block_lengths[cursor]
            if b_len > h - step:
                b_len = h - step
            step += b_len
            cursor += 1

    # ---- Parallel stitch using per-scenario draw-pool offset. ----
    for b in nb.prange(b_blocks):
        cursor_b = block_offset[b]
        step = 0
        while step < h:
            b_len = block_lengths[cursor_b]
            if b_len > h - step:
                b_len = h - step
            start_t = start_indices[cursor_b]

            for i in range(b_len):
                # Circular wrap into the valid index range [1, T-1].
                raw = start_t + i
                wrapped = 1 + ((raw - 1) % max_valid_length)
                out[b, step + i] = wrapped

            step += b_len
            cursor_b += 1


def generate_sbb_path_indices(
    *,
    t_max: int,
    h: int,
    total_scenarios: int,
    expected_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate `(B, h)` int32 SBB path indices into [1, T-1].

    RNG draws happen here in NumPy; the path stitching runs in a
    parallel Numba kernel with zero hot-path allocations.
    """
    if t_max < 2:
        raise ValueError(f"T must be >= 2 for 1-day returns; got T={t_max}")
    if h < 1:
        raise ValueError(f"h must be >= 1; got h={h}")
    if total_scenarios < 1:
        raise ValueError(f"B must be >= 1; got B={total_scenarios}")
    if expected_block_size < 1.0:
        raise ValueError(
            f"expected_block_size must be >= 1.0; got {expected_block_size}"
        )

    max_valid_length = t_max - 1
    p = 1.0 / float(expected_block_size)

    # Worst case: every block contributes one timestep -> B*h blocks.
    max_possible_blocks = total_scenarios * h

    if p >= 1.0:
        block_lengths = np.ones(max_possible_blocks, dtype=np.int32)
    else:
        block_lengths = rng.geometric(p, size=max_possible_blocks).astype(
            np.int32, copy=False
        )

    start_indices = rng.integers(
        low=1, high=t_max, size=max_possible_blocks, dtype=np.int32
    )

    out = np.empty((total_scenarios, h), dtype=np.int32)

    _kernel_stitch_sbb_paths(
        np.ascontiguousarray(block_lengths),
        np.ascontiguousarray(start_indices),
        out,
        np.int32(max_valid_length),
    )

    return out


__all__ = ["generate_sbb_path_indices"]
