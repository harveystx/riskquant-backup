# src/riskquant/credit/jtd/blocks/independent_bernoulli.py

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import INDEPENDENT_BERNOULLI_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class IndependentBernoulliJtdBlock(_JtdBlock):
    """
    Independent-default credit model.

    U_i,s ~ Uniform(0,1)

    default_i,s = 1(U_i,s < PD_i)

    loss_i,s = ead_i,s * lgd_i * default_i,s
    """

    JTD_MODEL_ID: ClassVar[int] = INDEPENDENT_BERNOULLI_ID

    pd: npt.NDArray[np.float64]

    uniform_uix: npt.NDArray[np.int32]
