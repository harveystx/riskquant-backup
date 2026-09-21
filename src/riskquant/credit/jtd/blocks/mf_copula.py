# src/riskquant/credit/jtd/blocks/mf_copula.py

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _MultiFactorCopulaBlock(_JtdBlock):
    """
    Multi-factor copula family base: K systemic loadings per obligor, CSR-packed.

    Each obligor i loads on a variable-length set of (independent, standard
    normal) systemic factors. The loadings are stored in compressed sparse row
    (CSR) form so obligors may reference different factor subsets with zero
    padding:

        loading_indptr[i] : loading_indptr[i + 1]

    slices the flat `loading_values` / `systemic_uix` arrays for obligor i.

    The latent variable is

        A_i = sum_j loading_values[j] * Z[systemic_uix[j]]  +  idio_weight[i] * eps_i

    over j in the obligor's CSR slice, with `idio_weight[i] = sqrt(1 - sum beta^2)`
    precomputed at ingest (the systemic factors are independent, so the systemic
    variance is the sum of squared loadings). Default iff A_i < thresholds[i].
    """

    _ABSTRACT: ClassVar[bool] = True

    # Threshold per obligor: A_i < t_i implies default. float64
    thresholds: npt.NDArray[np.float64]  # (N,)

    # CSR loading structure (ragged systemic loadings per obligor).
    loading_indptr: npt.NDArray[np.int64]  # (N + 1,)
    loading_values: npt.NDArray[np.float64]  # (nnz,)
    systemic_uix: npt.NDArray[np.int32]  # (nnz,) index into CreditFactorMatrix

    idiosyncratic_uix: npt.NDArray[np.int32]  # (N,)
    idio_weight: npt.NDArray[np.float64]  # (N,) sqrt(1 - sum beta^2)
