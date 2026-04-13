# src/riskquant/simlab/apply/shock_kernel.py
from __future__ import annotations

import numpy as np
import numba as nb
from typing import Optional

from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE,
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
)

# ============================================================
# ZERO-ALLOCATION NUMBA APPLY KERNEL
# ============================================================


@nb.njit(parallel=True, fastmath=True, cache=True)
def _apply_shocks_numba_core(
    base_levels: np.ndarray,  # (N,) float64
    shocks: np.ndarray,  # (B, K) float64 (Fortran-contiguous)
    req_to_uix: np.ndarray,  # (N,) int32
    shock_scale: np.ndarray,  # (N,) float64
    shift_types: np.ndarray,  # (N,) int8
    out: np.ndarray,  # (N, B) float64 (C-contiguous)
):
    """
    The inner JIT-compiled loop.
    Parallelized across the N required exposures.
    """
    N = base_levels.shape[0]
    B = shocks.shape[0]

    # Parallelize over the portfolio exposures (N)
    for n in nb.prange(N):
        # 1. Fetch factor-specific metadata once per row
        base = base_levels[n]
        k = req_to_uix[n]
        scale = shock_scale[n]
        st = shift_types[n]

        # Pull the shock column once per row. This is the only way to avoid
        # redundant memory reads of the huge shocks array inside the B-loop.
        shock_col = shocks[:, k]

        # 2. Branching is hoisted OUTSIDE the B-loop.
        # The CPU executes the inner loop with zero branch prediction misses.
        if st == SHIFT_LOG_RETURN:
            for b in range(B):
                # out is (N, B) C-contiguous.
                # Looping over b writes sequentially into memory. Cache-perfect.
                out[n, b] = base * np.exp(shock_col[b] * scale)

        elif st == SHIFT_SIMPLE_RETURN:
            for b in range(B):
                out[n, b] = base * (1.0 + (shock_col[b] * scale))

        elif st == SHIFT_DIFFERENCE:
            for b in range(B):
                out[n, b] = base + (shock_col[b] * scale)


# ============================================================
# PYTHON DISPATCH WRAPPER
# ============================================================


def apply_shocks_to_base_levels(
    *,
    base_levels: np.ndarray,
    batch: ScenarioSetBatch,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unpacks the Python ScenarioSetBatch dataclass and feeds the raw
    NumPy arrays into the Numba kernel.
    """
    N = base_levels.shape[0]
    B = batch.shocks.shape[0]

    # Force shocks to be Fortran-contiguous for optimal access patterns in the Numba kernel.
    if not batch.shocks.flags["F_CONTIGUOUS"]:
        shocks = np.asfortranarray(batch.shocks)
    else:
        shocks = batch.shocks

    # Pre-allocate if not provided. In the hot VaR loop,
    # you should always be passing a slice of the global array here.
    if out is None:
        out = np.empty((N, B), dtype=np.float64, order="C")

    _apply_shocks_numba_core(
        base_levels=base_levels,
        shocks=shocks,
        req_to_uix=batch.required_to_shock_uix,
        shock_scale=batch.shock_scale,
        shift_types=batch.required_shift_type_id,
        out=out,
    )

    return out
