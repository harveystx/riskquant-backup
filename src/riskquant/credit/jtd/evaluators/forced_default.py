# src/riskquant/credit/jtd/evaluators/forced_default.py
"""JIT-boundary wrapper for the forced-default JTD kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.kernels.forced_default import kernel_forced_default_jtd

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix
    from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock


def evaluate_forced_default_jtd(
    *,
    block: ForcedDefaultJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,  # ignored: model is scenario-independent
    out: npt.NDArray[np.float64],
) -> None:
    kernel_forced_default_jtd(out=out, ead=ead, lgd=block.lgd)
