# src/riskquant/credit/jtd/evaluators/mf_copula_students_t.py
"""JIT-boundary wrapper for the multi-factor Student-t copula JTD kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.kernels.mf_copula_students_t import (
    kernel_mf_students_t_copula_jtd,
)

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix
    from riskquant.credit.jtd.blocks.mf_copula_students_t import (
        MfStudentsTCopulaJtdBlock,
    )


def evaluate_mf_students_t_copula_jtd(
    *,
    block: MfStudentsTCopulaJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_mf_students_t_copula_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        loading_indptr=block.loading_indptr,
        loading_values=block.loading_values,
        systemic_uix=block.systemic_uix,
        idiosyncratic_uix=block.idiosyncratic_uix,
        idio_weight=block.idio_weight,
        thresholds=block.thresholds,
        sqrt_nu=block.sqrt_nu,
        chi_squared_uix=block.chi_squared_uix,
    )
