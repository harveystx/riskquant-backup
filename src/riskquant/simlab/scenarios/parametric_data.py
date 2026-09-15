# src/riskquant/simlab/scenarios/parametric_data.py
"""
Construction-data contract for the parametric scenario generator.

SimLab owns this contract: its shape is dictated by ``ParametricGenerator``'s
needs (the generator resolves Cholesky columns / vols / GMM multipliers by
``searchsorted`` on ``factor_ids``, which therefore MUST be sorted-ascending and
unique), and it is a typed-array data object of the same kind SimLab already
owns (``ScenarioSetBatch``). It is a plain frozen dataclass over NumPy — no
Pydantic, no strings — so it stays inside the NumPy/Numba core tier.

The object owns DATA invariants only (dtype, layout, dimensions, finiteness,
positivity, canonical ordering). Model semantics — total_scenarios, seed,
distribution selection and distribution parameters — remain owned by
``ParametricGenerator``. Inputs are validated, never silently cast: a caller
must supply canonical arrays or fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ParametricRiskData:
    """
    Canonical numerical inputs to construct a ``ParametricGenerator``.

    Contract (all rejected, never coerced):
      - ``factor_ids``      : int32, 1-D, C-contiguous, non-empty, sorted
                              ascending and unique (length K).
      - ``cholesky_matrix`` : float64, 2-D, (K, K) square, C-contiguous, finite.
      - ``volatilities``    : float64, 1-D, length K, C-contiguous, finite,
                              strictly positive.
    """

    factor_ids: np.ndarray
    cholesky_matrix: np.ndarray
    volatilities: np.ndarray

    def __post_init__(self) -> None:
        f = self.factor_ids
        if f.ndim != 1 or f.dtype != np.int32:
            raise TypeError(f"factor_ids must be 1-D int32, got {f.ndim}-D {f.dtype}.")
        if not f.flags.c_contiguous:
            raise ValueError("factor_ids must be C-contiguous.")
        if f.shape[0] == 0:
            raise ValueError("factor_ids must be non-empty.")
        if f.shape[0] > 1 and np.any(np.diff(f) <= 0):
            raise ValueError("factor_ids must be sorted ascending and unique.")
        k = f.shape[0]

        c = self.cholesky_matrix
        if c.ndim != 2 or c.dtype != np.float64:
            raise TypeError(
                f"cholesky_matrix must be 2-D float64, got {c.ndim}-D {c.dtype}."
            )
        if c.shape != (k, k):
            raise ValueError(
                f"cholesky_matrix must be square ({k}, {k}), got {c.shape}."
            )
        if not c.flags.c_contiguous:
            raise ValueError("cholesky_matrix must be C-contiguous.")
        with np.errstate(all="ignore"):
            if not np.isfinite(c).all():
                raise ValueError("cholesky_matrix must contain only finite values.")

        v = self.volatilities
        if v.ndim != 1 or v.dtype != np.float64:
            raise TypeError(
                f"volatilities must be 1-D float64, got {v.ndim}-D {v.dtype}."
            )
        if v.shape[0] != k:
            raise ValueError(f"volatilities length must be {k}, got {v.shape[0]}.")
        if not v.flags.c_contiguous:
            raise ValueError("volatilities must be C-contiguous.")
        with np.errstate(all="ignore"):
            if not np.isfinite(v).all():
                raise ValueError("volatilities must contain only finite values.")
            if np.any(v <= 0.0):
                raise ValueError("volatilities must be strictly positive.")
