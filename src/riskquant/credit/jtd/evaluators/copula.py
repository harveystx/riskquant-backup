# src/riskquant/credit/jtd/evaluators/copula.py
"""JIT-boundary wrappers for the copula JTD kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.kernels.copula import (
    kernel_sf_gaussian_copula_jtd,
    kernel_sf_t_copula_jtd,
)

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix
    from riskquant.credit.jtd.blocks.copula import (
        GaussianCopulaJtdBlock,
        TCopulaJtdBlock,
    )


def evaluate_gaussian_copula_jtd(
    *,
    block: GaussianCopulaJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_sf_gaussian_copula_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        beta=block.beta,
        thresholds=block.thresholds,
        sys_uix=block.systemic_uix,
        idio_uix=block.idiosyncratic_uix,
    )


def evaluate_t_copula_jtd(
    *,
    block: TCopulaJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_sf_t_copula_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        beta=block.beta,
        thresholds=block.thresholds,
        nu=block.nu,
        sys_uix=block.systemic_uix,
        idio_uix=block.idiosyncratic_uix,
        chi_uix=block.chi_squared_uix,
    )
