# src/riskquant/credit/factors/simulate/parametric_copula_t.py
"""Student-t-copula latent-factor generator (zero-allocation, streaming).

Emits the RAW latent universe: standard-normal columns for systemic / idio
factors and chi-square columns for the mixing variables W. The Student-t
transform ``sqrt(nu_i / W)`` is applied ONCE, per obligor, inside
``kernel_sf_t_copula_jtd`` — never pre-applied here.

`nu_by_chi` is a per-chi-factor SoA array, not a hidden scalar default.
Chi-square draws use ``standard_gamma`` (which supports ``out=``);
``chisquare`` does not. There is no Cholesky here — systemic correlation
belongs to multi-factor loadings, and single-factor idiosyncratic draws
are independent by definition.

CONTRACT: yielded `latent_factors` are views into a reused buffer.

LAYOUT
------
- Main buffer is C-order ``(batch_size, n_factors)`` so the partial-batch
  slice ``buffer[:bs, :]`` is C-contiguous — required by
  ``rng.standard_normal(out=...)``.
- A separate 1D ``chi_scratch`` buffer of length ``batch_size`` is reused
  for each chi-square fill. The scratch is contiguous; the strided write
  ``buffer[:bs, chi_cols[c]] = chi_scratch[:bs]`` happens after the gamma
  draw, NOT during it. Filling chi columns directly in the main C-order
  buffer would require a non-contiguous ``out=``, which numpy rejects.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.simulate.base import (
    CreditFactorBatch,
    CreditFactorGenerator,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TCopulaGenerator(CreditFactorGenerator):
    """
    Zero-allocation Student-t-copula latent-factor generator.

    `chi_factor_columns` lists the factor-axis positions that carry a
    chi-square mixing variable W; `nu_by_chi` is the degrees-of-freedom
    array aligned to it. Every other column is a standard-normal latent
    factor.
    """

    available_factor_ids: npt.NDArray[np.int32]
    total_scenarios: int
    chi_factor_columns: npt.NDArray[np.int32]  # (n_chi,) positions in the F axis
    nu_by_chi: npt.NDArray[np.float64]  # (n_chi,) dof, aligned
    seed: int | None = field(default=None)

    def iter_batches(self, *, batch_size: int) -> Iterator[CreditFactorBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chi_factor_columns.shape != self.nu_by_chi.shape:
            raise ValueError("chi_factor_columns and nu_by_chi must be aligned")
        if self.nu_by_chi.size and not np.all(self.nu_by_chi > 0.0):
            raise ValueError("nu_by_chi must be strictly positive")

        n_factors = int(self.available_factor_ids.shape[0])
        chi_cols = self.chi_factor_columns
        chi_shape_halved = self.nu_by_chi * 0.5  # chi2(nu) = 2 * Gamma(nu/2)
        n_chi = int(chi_cols.shape[0])

        rng = np.random.default_rng(self.seed)

        # Main buffer: C-order so [:bs, :] is C-contiguous for any bs.
        buffer = np.empty((batch_size, n_factors), dtype=np.float64, order="C")
        # 1D scratch: contiguous, used as `out=` for the chi gamma draw.
        chi_scratch = np.empty(batch_size, dtype=np.float64)

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)
            view = buffer[:bs, :]

            # Fill the whole batch with standard normals...
            rng.standard_normal(out=cast(np.ndarray, view))

            # ...then overwrite the chi columns with raw chi-square draws.
            # Draw into a 1D contiguous scratch, then write across (the
            # destination is a strided 1D view, but the write itself is
            # not the `out=` target).
            for c in range(n_chi):
                chi_view = chi_scratch[:bs]
                rng.standard_gamma(chi_shape_halved[c], out=cast(np.ndarray, chi_view))
                chi_view *= 2.0
                buffer[:bs, chi_cols[c]] = chi_view

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
