# src/riskquant/credit/factors/simulate/parametric_uniform.py
"""Independent Uniform(0, 1) latent-factor generator (zero-allocation, streaming).

Emits independent U(0, 1) draws for the full credit-factor universe — the
factor source the Bernoulli-family JTD kernels consume (`u_row[s] < pd_i`).
This is the counterpart to `GaussianCopulaGenerator`: same streaming, zero-
allocation contract, different marginal.

There is no dependence here, by design and by definition. The independent-
Bernoulli model is independent; coupling a uniform row to a systemic factor
would make it a copula (use `COPULA_SF_GAUSSIAN` instead). When a portfolio
spans both Bernoulli and copula blocks, use `MixedCreditFactorGenerator`,
which fills normal, uniform, chi and indicator rows in one matrix.

CONTRACT: yielded `latent_factors` are views into a reused C-contiguous
buffer; consumers treat them as ephemeral (read-only) or copy.

LAYOUT: the buffer is C-order ``(batch_size, n_factors)``. The partial-batch
slice ``buffer[:bs, :]`` is therefore C-contiguous for any ``bs <=
batch_size`` — required by ``rng.random(out=...)``, which rejects non-
contiguous output arrays.
"""

from __future__ import annotations

__all__ = [
    "IndependentUniformGenerator",
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
class IndependentUniformGenerator(CreditFactorGenerator):
    """Zero-allocation independent Uniform(0, 1) latent-factor generator."""

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
    ) -> IndependentUniformGenerator:
        """Build from the credit factor-universe master (sizes the matrix to
        the universe's factor axis). Fills EVERY column with U(0, 1), so it is
        correct only for portfolios that read independent-Bernoulli (uniform) columns. Use
        MixedCreditFactorGenerator.from_universe for a matrix that mixes uniform with normal / chi / indicator columns.
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
        # the precondition `rng.random(out=...)` requires.
        buffer = np.empty((batch_size, n_factors), dtype=np.float64, order="C")

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)
            view = buffer[:bs, :]

            rng.random(out=cast(np.ndarray, view))

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
