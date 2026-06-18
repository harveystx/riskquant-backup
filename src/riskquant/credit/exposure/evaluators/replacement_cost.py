# src/riskquant/credit/exposure/evaluators/replacement_cost.py
"""
Replacement Cost (RC) exposure model: evaluator.

    V  = sum of MtM over the netting set's member trades
    RC = max(V - C, 0)

`_kernel_replacement_cost` computes RC for all netting sets and scenarios
in one parallel pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.evaluators.kernels.replacement_cost import (
    _kernel_replacement_cost,
)
from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView


def evaluate_replacement_cost(
    *,
    view: RcNettingSetView,
    mtm: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Unpack the view SoA fields and dispatch to the JIT kernel."""
    _kernel_replacement_cost(
        out=out,
        mtm=mtm,
        mtm_row_perm=view.mtm_row_perm,
        ns_offsets=view.netting_set_offsets,
        collateral=view.collateral,
    )
