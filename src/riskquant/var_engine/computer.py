# src/riskquant/var_engine/computer.py
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def calculate_historical_var_and_es(pnl_matrix, confidence_level):
    G, B = pnl_matrix.shape
    tail_count = int(np.floor(B * (1.0 - confidence_level)))

    var_out = np.empty(G, dtype=np.float64)
    es_out = np.empty(G, dtype=np.float64)

    for g in nb.prange(G):
        row = pnl_matrix[g, :].copy()

        if tail_count == 0:
            min_val = np.min(row)
            var_out[g] = min_val
            es_out[g] = min_val
            continue

        row.partition(tail_count)

        var_out[g] = row[tail_count]
        es_out[g] = row[:tail_count].sum() / tail_count

    return var_out, es_out
