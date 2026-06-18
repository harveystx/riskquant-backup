# src/riskquant/credit/factors/simulate/parametric_copula_gaussian.py
"""Gaussian-copula latent-factor generator (zero-allocation, streaming).

Emits independent standard-normal draws for the full credit-factor universe.
All dependence — single-factor `beta` mixing or multi-factor CSR loadings —
is applied downstream in the JTD kernels, never here.

CONTRACT: yielded `latent_factors` are views into a reused C-contiguous
buffer; consumers treat them as ephemeral (read-only) or copy.

LAYOUT: the buffer is C-order ``(batch_size, n_factors)``. The partial-
batch slice ``buffer[:bs, :]`` is therefore C-contiguous for any
``bs <= batch_size`` — required by ``rng.standard_normal(out=...)``,
which rejects non-contiguous output arrays. (An F-order layout would
break here on the final partial batch.)
"""

from __future__ import annotations

__all__ = [
    "GaussianCopulaGenerator",
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
class GaussianCopulaGenerator(CreditFactorGenerator):
    """Zero-allocation independent standard-normal latent-factor generator."""

    available_factor_ids: npt.NDArray[np.int32]
    total_scenarios: int
    seed: int | None = field(default=None)

    @classmethod
    def from_universe(
        cls,
        universe: CompiledCreditFactorUniverse,
        *,
        total_scenarios: int,
        seed: int | None = None,
    ) -> GaussianCopulaGenerator:
        """Build from the credit factor-universe master (sizes the matrix to
        the universe's factor axis). Fills EVERY column with standard normal, so it is
        correct only for portfolios that read systemic / idiosyncratic columns. Use
        MixedCreditFactorGenerator.from_universe for a matrix that also honours chi / uniform / indicator columns.
        """
        return cls(
            available_factor_ids=universe.factor_ids,
            total_scenarios=total_scenarios,
            seed=seed,
        )

    def iter_batches(self, *, batch_size: int) -> Iterator[CreditFactorBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        n_factors = int(self.available_factor_ids.shape[0])
        rng = np.random.default_rng(self.seed)

        # C-order: buffer[:bs, :] is C-contiguous for every bs <= batch_size,
        # the precondition `rng.standard_normal(out=...)` requires.
        buffer = np.empty((batch_size, n_factors), dtype=np.float64, order="C")

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)
            view = buffer[:bs, :]

            rng.standard_normal(out=cast(np.ndarray, view))

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
