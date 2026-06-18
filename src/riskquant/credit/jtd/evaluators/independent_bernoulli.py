# src/riskquant/credit/jtd/evaluators/independent_bernoulli.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.kernels.independent_bernoulli import (
    kernel_independent_bernoulli_jtd,
)


def evaluate_independent_bernoulli_jtd(
    *,
    block: IndependentBernoulliJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_independent_bernoulli_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        pd=block.pd,
        uniform_uix=block.uniform_uix,
    )
