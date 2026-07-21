# src/riskquant/credit/jtd/evaluators/marshall_olkin.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.marshall_olkin import (
    MarshallOlkinJtdBlock,
)
from riskquant.credit.jtd.kernels.marshall_olkin import (
    kernel_marshall_olkin_jtd,
)


def evaluate_marshall_olkin_jtd(
    *,
    block: MarshallOlkinJtdBlock,
    ead: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    kernel_marshall_olkin_jtd(
        out=out,
        latent_factors=credit_factors.latent_factors,
        ead=ead,
        lgd=block.lgd,
        pd_idio=block.pd_idio,
        pd_sys=block.pd_sys,
        uniform_uix=block.uniform_uix,
        systemic_uix=block.systemic_uix,
    )
