# src/riskquant/simlab/shocks/linear_term_kernel.py
"""
Zero-allocation kernel: sparse linear shock aggregation, then reconstruction.

For row ``n`` and scenario ``b`` (D088 §13.5; Design Specification §26)::

    X[n, b]   = sum_{t = off[n]}^{off[n+1] - 1} loadings[t] * float64(shocks[b, cols[t]])
    out[n, b] = T^-1_{kappa[n]}(base[n], X[n, b])

The kernel knows only numbers: offsets, column indices, loadings, target
shift-type ids, base levels and a shock matrix. It knows no factor, name,
chart semantics beyond the inverse formulas, horizon, world, model or run.
Every input contract is enforced by the wrapper
(`riskquant.simlab.shocks.linear_term_apply`) before this kernel is called.

Numerical contract
------------------
- Order: for every (n, b) the products are added in ascending term index,
  starting from 0.0: ``((0.0 + p_0) + p_1) + ... + p_last``. The kernel never
  sorts, reorders or splits the terms of a row.
- Precision: each float32 shock is converted to float64 before the
  multiplication; the loading, the product, the accumulator and the output
  are float64.
- Reassociation: compiled with `stable_math_parallel_njit`, which omits
  ``"reassoc"``, so the compiler cannot reorder the term reduction.
  ``"contract"`` (FMA) stays enabled, as in every canonical decorator; the
  results are therefore bitwise reproducible within one execution
  environment, not across platforms (D088 §13.6).
- Parallelism: ``prange`` over rows only. One thread computes one row; the
  term reduction of a row is sequential.
- Reconstruction is applied after the complete sum (never per term).

Memory layout
-------------
- ``shocks`` is ``(B, K)`` F-contiguous: ``shocks[:, c]`` is one contiguous
  run down the scenario axis.
- ``out`` is ``(N, B)`` C-contiguous: ``out[n]`` is one contiguous row.

Loop orientation: for each row, the terms are visited in ascending order and
each term updates the whole scenario row ``out[n, :]``. For every scenario
this performs exactly the operation sequence above; the scenarios are
independent lanes, so the interleaving across ``b`` does not change any
result. The accumulator lives in ``out[n, b]`` (float64), so no scratch
buffer is allocated.

NO_SHIFT rows (D088 §13.4) carry zero terms and write ``base`` for every
scenario. ABSOLUTE_BASE rows apply ``base + X * |base|`` (D025).
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]
import numpy as np

from riskquant.jit import (
    _F32_2D_C_RO,
    _F32_2D_F_RO,
    _F64_1D_C_RO,
    _F64_2D_C,
    _I8_1D_C_RO,
    _I32_1D_C_RO,
    _NB_VOID,
    stable_math_parallel_njit,
)
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)


# Every input is declared read-only: the wrapper passes zero-copy read-only
# views, so frozen arrays (for example those of a compiled proxy model) and
# writable caller arrays share one signature, and the kernel cannot write to
# any input. Two `shocks` layouts are registered for the same reason as in
# `shock_kernel`: an array with one scenario or one column is both C- and
# F-contiguous, and Numba resolves that case to 'C'.
@stable_math_parallel_njit(
    [
        _NB_VOID(
            _F64_1D_C_RO,  # base_levels         (N,)
            _I32_1D_C_RO,  # row_term_offsets    (N + 1,)
            _I32_1D_C_RO,  # term_column_uix     (T,)
            _F64_1D_C_RO,  # term_loadings       (T,)
            _I8_1D_C_RO,  # target_shift_types  (N,)
            _F32_2D_F_RO,  # shocks              (B, K)  F-contig
            _F64_2D_C,  # out                 (N, B)  C-contig
        ),
        _NB_VOID(
            _F64_1D_C_RO,  # base_levels         (N,)
            _I32_1D_C_RO,  # row_term_offsets    (N + 1,)
            _I32_1D_C_RO,  # term_column_uix     (T,)
            _F64_1D_C_RO,  # term_loadings       (T,)
            _I8_1D_C_RO,  # target_shift_types  (N,)
            _F32_2D_C_RO,  # shocks              (B, K)  single scenario or column
            _F64_2D_C,  # out                 (N, B)  C-contig
        ),
    ],
)
def _kernel_apply_linear_terms(
    base_levels,
    row_term_offsets,
    term_column_uix,
    term_loadings,
    target_shift_types,
    shocks,
    out,
) -> None:
    """Row-parallel term aggregation followed by target reconstruction."""
    n_rows = base_levels.shape[0]
    n_scn = shocks.shape[0]

    for n in nb.prange(n_rows):
        base = base_levels[n]
        st = target_shift_types[n]
        out_row = out[n]

        if st == SHIFT_NO_SHIFT_ID:
            # T^-1_NO_SHIFT(base, X) = base; the row has no terms.
            for b in range(n_scn):
                out_row[b] = base
        else:
            for b in range(n_scn):
                out_row[b] = 0.0
            # Ascending canonical term order; sequential within the row.
            for t in range(row_term_offsets[n], row_term_offsets[n + 1]):
                beta = term_loadings[t]
                col = shocks[:, term_column_uix[t]]
                for b in range(n_scn):
                    out_row[b] += beta * np.float64(col[b])

            if st == SHIFT_LOG_RETURN_ID:
                for b in range(n_scn):
                    out_row[b] = base * math.exp(out_row[b])
            elif st == SHIFT_SIMPLE_RETURN_ID:
                for b in range(n_scn):
                    out_row[b] = base * (1.0 + out_row[b])
            elif st == SHIFT_DIFFERENCE_ID:
                for b in range(n_scn):
                    out_row[b] = base + out_row[b]
            elif st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
                abs_base = abs(base)
                for b in range(n_scn):
                    out_row[b] = base + out_row[b] * abs_base
            else:
                # Unreachable: the wrapper rejects every other id. A loud NaN
                # is written rather than leaving the unreconstructed sum.
                for b in range(n_scn):
                    out_row[b] = math.nan
