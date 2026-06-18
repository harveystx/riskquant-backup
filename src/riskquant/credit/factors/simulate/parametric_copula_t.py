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

__all__ = [
    "TCopulaGenerator",
]

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.simulate.base import (
    CreditFactorBatch,
    CreditFactorGenerator,
)

if TYPE_CHECKING:
    from riskquant.credit.masters.factor_universe.compiled import (
        CompiledCreditFactorUniverse,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TCopulaGenerator(CreditFactorGenerator):
    """
    Zero-allocation Student-t-copula latent-factor generator.

    `chi_factor_columns` lists the factor-axis positions that carry a
    chi-square mixing variable $W$; `nu_by_chi` is the degrees-of-freedom
    array aligned to it. Every other column is a standard-normal latent
    factor.

    **CRITICAL KERNEL OPTIMIZATION:**
    The columns corresponding to `chi_factor_columns` DO NOT emit raw $\\chi^2$ draws.
    Instead, they emit the precomputed inverse square root: $1/\\sqrt{W}$.
    This allows the downstream JTD kernel to evaluate the Student-t variable as
    $\\sqrt{\\nu} \\cdot (1/\\sqrt{W}) \\cdot Z$, entirely eliminating the `np.sqrt`
    and division operations from the engine's innermost scenario loop.
    """

    available_factor_ids: npt.NDArray[np.int32]
    total_scenarios: int
    chi_factor_columns: npt.NDArray[np.int32]  # (n_chi,) positions in the F axis
    nu_by_chi: npt.NDArray[np.float64]  # (n_chi,) dof, aligned
    seed: int | None = field(default=None)

    @classmethod
    def from_universe(
        cls,
        universe: CompiledCreditFactorUniverse,
        *,
        total_scenarios: int,
        seed: int | None = None,
    ) -> TCopulaGenerator:
        """Build from the credit factor-universe master.

        `chi_factor_columns` and `nu_by_chi` come straight from the universe
        (`chi_uix` / `chi_nu_vector`, aligned by construction), so the dof a chi
        column is drawn with cannot drift from the dof JTD ingest reconciles
        obligor `nu` against — both read the one master. Sizes the matrix to the
        universe's factor axis; non-chi columns are standard normals, correct
        for t-copula portfolios (which read systemic / idiosyncratic / chi).
        """
        return cls(
            available_factor_ids=universe.factor_ids,
            total_scenarios=total_scenarios,
            chi_factor_columns=universe.chi_uix,
            nu_by_chi=universe.chi_nu_vector,
            seed=seed,
        )

    def chi_nu_by_uix(self) -> dict[int, float]:
        """{chi factor UIX -> dof} — the reconciliation map JTD ingest checks
        obligor ``nu`` against. This generator is the source of truth for the
        dof each chi column is drawn with; pass the result as
        ``compile_jtd_portfolio_from_df(..., chi_nu_by_uix=...)``."""
        return dict(
            zip(
                self.chi_factor_columns.tolist(),
                self.nu_by_chi.tolist(),
                strict=True,
            )
        )

    def iter_batches(self, *, batch_size: int) -> Iterator[CreditFactorBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chi_factor_columns.shape != self.nu_by_chi.shape:
            raise ValueError("chi_factor_columns and nu_by_chi must be aligned")
        if self.nu_by_chi.size and not np.all(self.nu_by_chi > 0.0):
            raise ValueError("nu_by_chi must be strictly positive")

        n_factors = int(self.available_factor_ids.shape[0])
        chi_cols = self.chi_factor_columns
        chi_shape_halved = self.nu_by_chi * 0.5  # chi^2(nu) = 2 * Gamma(nu/2)
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

            # ...then overwrite the chi columns with transformed chi-square draws.
            # Draw into a 1D contiguous scratch, then write across (the
            # destination is a strided 1D view, but the write itself is
            # not the `out=` target).

            for c in range(n_chi):
                chi_view = chi_scratch[:bs]
                rng.standard_gamma(chi_shape_halved[c], out=cast(np.ndarray, chi_view))
                chi_view *= 2.0  # W ~ chi^2(nu)

                # =====================================================================
                # MATHEMATICAL INVARIANT: PRECOMPUTED INVERSE SQUARE ROOT
                # We apply the 1/sqrt(W) transform in-place before writing to the
                # C-contiguous buffer. This ensures the downstream JTD kernel receives
                # the optimized factor and avoids calculating square roots in the hot path.
                # =====================================================================
                np.reciprocal(np.sqrt(chi_view), out=chi_view)

                buffer[:bs, chi_cols[c]] = chi_view

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
