# src/riskquant/credit/jtd/blocks/sf_copula_students_t.py
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.sf_copula import _SingleFactorCopulaBlock
from riskquant.credit.jtd.models import COPULA_SF_T_ID


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SfStudentsTCopulaJtdBlock(_SingleFactorCopulaBlock):
    """
    A_i = sqrt(nu_i / W) * (beta_i * Z + sqrt(1 - beta_i^2) * eps_i).

    `nu` is a real per-obligor SoA field, not a hidden default argument. The
    chi-square mixing variable W is read from the latent matrix via
    `chi_squared_uix`; the kernel — not the generator — owns the t-transform.
    """

    JTD_MODEL_ID: ClassVar[int] = COPULA_SF_T_ID

    chi_squared_uix: npt.NDArray[np.int32]  # (N,) index into mixing variable W
    sqrt_nu: npt.NDArray[np.float64]  # (N,) precomputed sqrt(degrees of freedom)
