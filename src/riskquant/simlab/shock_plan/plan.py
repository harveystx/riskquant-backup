# src/riskquant/simlab/shocks/plan.py

from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass(slots=True)
class ShockComputePlan:
    cols_sorted: np.ndarray
    shift_types_sorted: np.ndarray
    inverse_permutation: np.ndarray


def build_shock_compute_plan(
    *,
    shock_source_ids: np.ndarray,
    shock_shift_type_ids: np.ndarray,
    id_to_col: Dict[int, int],
) -> ShockComputePlan:

    cols = np.array(
        [id_to_col[int(sid)] for sid in shock_source_ids],
        dtype=np.int32,
    )

    shift_types = shock_shift_type_ids.astype(np.int8, copy=False)

    order = np.argsort(cols)

    cols_sorted = cols[order]
    shift_sorted = shift_types[order]

    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0])

    return ShockComputePlan(
        cols_sorted=cols_sorted,
        shift_types_sorted=shift_sorted,
        inverse_permutation=inv,
    )
