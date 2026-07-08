# src/riskquant/credit/exposure/evaluators/saccr.py
"""SA-CCR exposure model: evaluator. Unpacks the view and dispatches the kernel."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.evaluators.kernels.saccr import _kernel_saccr
from riskquant.credit.exposure.views.saccr import SaccrNettingSetView


def evaluate_saccr(
    *,
    view: SaccrNettingSetView,
    mtm: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    _kernel_saccr(
        out,
        mtm,
        view.mtm_row_perm,
        view.netting_set_offsets,
        view.collateral,
        view.add_on,
        view.alpha,
        view.multiplier_floor,
    )
