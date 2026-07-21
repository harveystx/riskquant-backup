# src/riskquant/credit/jtd/evaluators/beta_binomial.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.beta_binomial import (
    BetaBinomialJtdBlock,
)
from riskquant.credit.jtd.kernels.beta_binomial import (
    kernel_beta_binomial_jtd,
)


def evaluate_beta_binomial_jtd(
    *,
    block: BetaBinomialJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_beta_binomial_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        uniform_uix=block.uniform_uix,
        beta_uix=block.beta_uix,
    )
