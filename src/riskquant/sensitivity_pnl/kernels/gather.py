# src/riskquant/sensitivity_pnl/kernels/gather.py
"""
Zero-allocation displacement gather.

The standalone entry point receives displacements directly — the caller's shock
matrix already lives in the book's declared displacement space — so there is no
level arithmetic to do, only a compaction from the caller's `(K, S)` factor axis
onto the book's `(Ka, B)` active-factor axis.

Input is f32, matching `ScenarioSetBatch.shocks` and the `riskquant.constants`
rule that scenario shocks are float32. A standalone caller therefore hands over
its shock matrix with no dtype conversion at the boundary.

Separate kernel rather than an IDENTITY member of `DisplacementConvention`: an
identity convention would be a taxonomy entry describing the caller's plumbing
rather than the mathematics of a sensitivity, and every downstream dispatch on
the enum would carry a meaningless arm.

Both source and destination rows are contiguous runs, so this is a pure
contiguous-to-contiguous copy — the cheapest possible path into the kernel.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]
from numba.core import types as nb_types

# See displacement.py for why these come from riskquant.jit rather than a
# local `nb.float32[:, ::1]`-style redefinition.
from riskquant.jit import _F32_2D_C, _I32_1D_C, parallel_njit


@parallel_njit(
    nb_types.void(
        _F32_2D_C,  # source_displacement (K, S)           f32 C-contig
        _I32_1D_C,  # active_level_uix      (Ka,)
        nb_types.int64,  # batch_len
        nb_types.int64,  # source_offset
        _F32_2D_C,  # out                 (Ka, B_capacity) f32 C-contig
    ),
)
def kernel_gather_displacement(
    source_displacement,
    active_level_uix,
    batch_len,
    source_offset,
    out,
) -> None:
    """Compact the caller's factor axis onto the book's active-factor axis."""
    for k in nb.prange(active_level_uix.shape[0]):
        src = source_displacement[active_level_uix[k]]
        dst = out[k]
        for s in range(batch_len):
            dst[s] = src[source_offset + s]
