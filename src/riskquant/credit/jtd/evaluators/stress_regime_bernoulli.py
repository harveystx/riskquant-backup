# src/riskquant/credit/jtd/evaluators/stress_regime_bernoulli.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)
from riskquant.credit.jtd.kernels.stress_regime_bernoulli import (
    kernel_stress_regime_bernoulli_jtd,
)


def evaluate_stress_regime_bernoulli_jtd(
    *,
    block: StressRegimeBernoulliJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:

    # Extract the stress regime indicator (1 for stressed, 0 for not stressed) for
    # each scenario from the credit factors using the provided index from the block.
    stress_regime_indicator = credit_factors.latent_factors[
        block.stress_regime_indicator_uix
    ]

    kernel_stress_regime_bernoulli_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        stress_regime_indicator=stress_regime_indicator,
        ead=ead,
        lgd=block.lgd,
        pd=block.pd,
        uniform_uix=block.uniform_uix,
        stress_pd_multiplier=block.stress_pd_multiplier,
    )
