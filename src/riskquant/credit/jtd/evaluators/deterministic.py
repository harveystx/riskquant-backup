# src/riskquant/credit/jtd/evaluators/deterministic.py
"""JIT-boundary wrapper for the deterministic JTD kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.kernels.deterministic import kernel_deterministic_jtd

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix
    from riskquant.credit.jtd.blocks.deterministic import DeterministicJtdBlock


def evaluate_deterministic_jtd(
    *,
    block: DeterministicJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,  # ignored: model is scenario-independent
    out: npt.NDArray[np.float64],
) -> None:
    kernel_deterministic_jtd(out=out, ead=ead, lgd=block.lgd)
