# src/riskquant/credit/jtd/blocks/copula.py
"""
Copula JTD blocks — single-factor (Gaussian, Student-t) and multi-factor.

Single-factor blocks carry a scalar `beta` per obligor and one systemic / one
idiosyncratic `_uix`. The multi-factor block stores its loadings in CSR layout
(`factor_offsets`, flat `betas`, flat `systemic_uix`) — NEVER a padded
`(N, F_max)` array — so a 3-factor obligor next to a 30-factor obligor wastes
no cache lines and the kernel walks `factor_offsets[i]:factor_offsets[i+1]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import (
    COPULA_SF_GAUSSIAN_ID,
    COPULA_SF_T_ID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SingleFactorCopulaBlock(_JtdBlock):
    """Single-factor copula family base: one systemic factor, scalar loading."""

    _ABSTRACT: ClassVar[bool] = True

    # Threshold per obligor: A_i < t_i implies default. float64
    thresholds: npt.NDArray[np.float64]

    beta: npt.NDArray[np.float64]  # (N,) factor loading
    systemic_uix: npt.NDArray[np.int32]  # (N,) index into CreditFactorMatrix
    idiosyncratic_uix: npt.NDArray[np.int32]  # (N,)


@dataclass(frozen=True, slots=True, kw_only=True)
class GaussianCopulaJtdBlock(_SingleFactorCopulaBlock):
    """A_i = beta_i * Z + sqrt(1 - beta_i^2) * eps_i ; default if A_i < t_i."""

    JTD_MODEL_ID: ClassVar[int] = COPULA_SF_GAUSSIAN_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class TCopulaJtdBlock(_SingleFactorCopulaBlock):
    """
    A_i = sqrt(nu_i / W) * (beta_i * Z + sqrt(1 - beta_i^2) * eps_i).

    `nu` is a real per-obligor SoA field, not a hidden default argument. The
    chi-square mixing variable W is read from the latent matrix via
    `chi_squared_uix`; the kernel — not the generator — owns the t-transform.
    """

    JTD_MODEL_ID: ClassVar[int] = COPULA_SF_T_ID

    chi_squared_uix: npt.NDArray[np.int32]  # (N,) index into mixing variable W
    nu: npt.NDArray[np.float64]  # (N,) degrees of freedom
