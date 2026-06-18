# src/riskquant/credit/jtd/evaluators/sf_copula_gaussian.py
"""JIT-boundary wrappers for the copula JTD kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.kernels.sf_copula_gaussian import (
    kernel_sf_gaussian_copula_jtd,
)

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix


def evaluate_sf_gaussian_copula_jtd(
    *,
    block: SfGaussianCopulaJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_sf_gaussian_copula_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        factor_loading=block.factor_loading,
        thresholds=block.thresholds,
        sys_uix=block.systemic_uix,
        idio_uix=block.idiosyncratic_uix,
    )
