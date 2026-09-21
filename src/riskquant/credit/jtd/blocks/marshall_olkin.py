# src/riskquant/credit/jtd/blocks/marshall_olkin.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import MARSHALL_OLKIN_ID


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MarshallOlkinJtdBlock(_JtdBlock):
    """
    Marshall-Olkin (Common Shock) default model.

    Each obligor is exposed to two independent sources of default:
      1. An idiosyncratic shock unique to the obligor;
      2. A systemic common shock shared by all obligors with the same systemic_uix.

    Note! Critically the kernel already satisfies:

        PD_i = 1 - (1 - p_idio_i) * (1 - p_sys_i)

    meaning that p_idio_i and p_sys_i are decoupled from the marginal PD_i and
    p_idio_i is not the original marginal PD of obligor i.

    An obligor defaults if either shock occurs:

        D_i = I(U_i < p_idio_i) OR I(S_g < p_sys_i)

    where:
        U_i ~ U(0, 1)
        S_g ~ U(0, 1)

    both are mutually independent.

    Loss:
        L_i,s = EAD_i,s * LGD_i * D_i,s
    """

    JTD_MODEL_ID: ClassVar[int] = MARSHALL_OLKIN_ID

    pd_idio: npt.NDArray[np.float64]  # (N,) - decoupled idiosyncratic PD
    pd_sys: npt.NDArray[np.float64]  # (N,) - probability of the common shock

    uniform_uix: npt.NDArray[np.int32]  # (N,) - obligor-specific uniform index
    systemic_uix: npt.NDArray[np.int32]  # (N,) - shared systemic uniform index
