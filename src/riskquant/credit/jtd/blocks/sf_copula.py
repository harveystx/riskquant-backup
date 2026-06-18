# src/riskquant/credit/jtd/blocks/sf_copula.py

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock


@dataclass(frozen=True, slots=True, kw_only=True)
class _SingleFactorCopulaBlock(_JtdBlock):
    """Single-factor copula family base: one systemic factor, scalar loading."""

    _ABSTRACT: ClassVar[bool] = True

    # Threshold per obligor: A_i < t_i implies default. float64
    thresholds: npt.NDArray[np.float64]

    factor_loading: npt.NDArray[np.float64]  # (N,) factor loading
    systemic_uix: npt.NDArray[np.int32]  # (N,) index into CreditFactorMatrix
    idiosyncratic_uix: npt.NDArray[np.int32]  # (N,)
