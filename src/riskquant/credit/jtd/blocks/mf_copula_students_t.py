# src/riskquant/credit/jtd/blocks/mf_copula_students_t.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.mf_copula import _MultiFactorCopulaBlock
from riskquant.credit.jtd.models import COPULA_MF_T_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class MfStudentsTCopulaJtdBlock(_MultiFactorCopulaBlock):
    """
    A_i = sqrt(nu_i / W) * (sum_k beta_ik * Z_k + sqrt(1 - sum beta^2) * eps_i) ;
    default if A_i < t_i.

    The multi-factor Gaussian latent (CSR loadings + idiosyncratic, from the
    shared `_MultiFactorCopulaBlock` base) is t-scaled by the shared chi-square
    mixing variable W — read from the latent matrix via `chi_squared_uix` — so a
    heavy market draw drags every obligor's latent down together (tail
    dependence). `thresholds` are Student-t quantiles of PD (with nu); the kernel,
    not the generator, owns the t-transform.
    """

    JTD_MODEL_ID: ClassVar[int] = COPULA_MF_T_ID

    chi_squared_uix: npt.NDArray[np.int32]  # (N,) index into mixing variable W
    sqrt_nu: npt.NDArray[np.float64]  # (N,) precomputed sqrt(degrees of freedom)
