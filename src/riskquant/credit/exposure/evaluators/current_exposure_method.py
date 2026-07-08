# src/riskquant/credit/exposure/evaluators/current_exposure_method.py
"""
Current Exposure Method (CEM) exposure model: evaluator.

    EAD = NRC + sum(gross_PFE) * (0.4 + 0.6 * NGR)

Forms the per-trade gross add-on ``effective_notional * supervisory_factor`` once
at the JIT boundary (the two are kept separable on the view for the SA-CCR path),
then dispatches the SoA fields to the JIT kernel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.evaluators.kernels.current_exposure_method import (
    _kernel_current_exposure_method,
)
from riskquant.credit.exposure.views.current_exposure_method import CemNettingSetView


def evaluate_current_exposure_method(
    *,
    view: CemNettingSetView,
    mtm: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Materialize the gross add-on and dispatch to the JIT kernel."""
    # (M_view,) formed once at the edge, not in the kernel
    gross_pfe = np.multiply(view.effective_notional, view.supervisory_factor)

    _kernel_current_exposure_method(
        out,
        mtm,
        view.mtm_row_perm,
        view.netting_set_offsets,
        view.collateral,
        gross_pfe,
        view.pfe_floor,
        view.pfe_coefficient,
    )
