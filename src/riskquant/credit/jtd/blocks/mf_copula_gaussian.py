# src/riskquant/credit/jtd/blocks/mf_copula_gaussian.py
from dataclasses import dataclass
from typing import ClassVar

from riskquant.credit.jtd.blocks.mf_copula import _MultiFactorCopulaBlock
from riskquant.credit.jtd.models import COPULA_MF_GAUSSIAN_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class MfGaussianCopulaJtdBlock(_MultiFactorCopulaBlock):
    """A_i = sum_k beta_ik * Z_k + sqrt(1 - sum beta^2) * eps_i ; default if A_i < t_i."""

    JTD_MODEL_ID: ClassVar[int] = COPULA_MF_GAUSSIAN_ID
