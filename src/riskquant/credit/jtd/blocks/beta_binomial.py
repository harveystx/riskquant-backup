# src/riskquant/credit/jtd/blocks/beta_binomial.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import BETA_BINOMIAL_ID


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BetaBinomialJtdBlock(_JtdBlock):
    """
    Beta-binomial default model.

    In this model, the default probability is stochastic and common
    across obligors.

    For scenario s:

        P_s ~ Beta(alpha, beta)

    For obligor i:

        D_i,s ~ Bernoulli(P_s)

    Loss:

        L_i,s = EAD_i,s * LGD_i * D_i,s
    """

    JTD_MODEL_ID: ClassVar[int] = BETA_BINOMIAL_ID

    uniform_uix: npt.NDArray[np.int32]  # (N,)

    beta_uix: npt.NDArray[np.int32]  # (N,)
