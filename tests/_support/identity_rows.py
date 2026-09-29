# tests/_support/identity_rows.py
"""
Proxy V1 identity rows for tests that apply world columns directly.

D088 §2: a V1 node without an authored model has the identity row
J_A = {(A, kappa_A)}, beta = 1. When a test requests one world column per row
in row order (driver = the row's factor, chart = the row's kappa), row k reads
column k with loading 1 and is reconstructed with its own chart. This helper
builds exactly that operator and applies it with the production Slice 2
wrapper; it adds no semantics of its own.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels


def apply_identity_rows(
    *,
    base_levels: npt.NDArray[np.float64],
    chart_ids: npt.NDArray[np.int8],
    shocks: npt.NDArray[np.float32],
    out: npt.NDArray[np.float64],
) -> None:
    k = base_levels.shape[0]
    apply_linear_terms_to_base_levels(
        base_levels=base_levels,
        row_term_offsets=np.arange(k + 1, dtype=np.int32),
        term_column_uix=np.arange(k, dtype=np.int32),
        term_loadings=np.ones(k, dtype=np.float64),
        target_shift_types=np.ascontiguousarray(chart_ids, dtype=np.int8),
        shocks=shocks,
        out=out,
    )
