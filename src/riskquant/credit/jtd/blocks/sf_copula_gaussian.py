# src/riskquant/credit/jtd/blocks/sf_copula_gaussian.py
from dataclasses import dataclass
from typing import ClassVar

from riskquant.credit.jtd.blocks.sf_copula import _SingleFactorCopulaBlock
from riskquant.credit.jtd.models import COPULA_SF_GAUSSIAN_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class SfGaussianCopulaJtdBlock(_SingleFactorCopulaBlock):
    """A_i = beta_i * Z + sqrt(1 - beta_i^2) * eps_i ; default if A_i < t_i."""

    JTD_MODEL_ID: ClassVar[int] = COPULA_SF_GAUSSIAN_ID
