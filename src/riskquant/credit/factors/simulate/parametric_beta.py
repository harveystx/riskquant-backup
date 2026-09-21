# src/riskquant/credit/factors/simulate/parametric_beta.py
"""
Independent Beta(alpha, beta) latent-factor generator.

Emits independent Beta(alpha, beta) draws for the specified factor rows of the
credit-factor universe.

The primary use case is the Beta-Binomial JTD model, where a scenario-level
Beta random variable represents the default environment:

    P_s ~ Beta(alpha, beta)

and obligor defaults are then generated conditionally:

    D_i,s | P_s ~ Bernoulli(P_s)

This generator produces the scenario-level Beta draws only. Bernoulli default
realization remains the responsibility of the JTD engine.

CONTRACT
--------
Yielded latent-factor matrices are views into a reused C-contiguous buffer.
Consumers must treat them as ephemeral (read-only) or copy them.

LAYOUT
------
The buffer is C-order:

    (batch_size, n_factors)

The partial-batch slice:

    buffer[:bs, :]

remains C-contiguous and is therefore suitable for direct NumPy random-number
generation.
"""

from __future__ import annotations

__all__ = [
    "BetaGenerator",
]

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BetaGenerator(CreditFactorGenerator):
    """
    Independent Beta(alpha, beta) factor generator.

    Each factor row i generates

        X_i,s ~ Beta(alpha_i, beta_i)

    independently across factors and scenarios.
    """

    available_factor_ids: npt.NDArray[np.int32]

    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]

    total_scenarios: int
    seed: int | None = field(default=None)

    @classmethod
    def from_universe(
        cls,
        universe: CompiledCreditFactorUniverse,
        *,
        total_scenarios: int,
        seed: int | None = None,
    ) -> BetaGenerator:
        """
        Construct a Beta generator over the universe's BETA_PD factors only.

        NOTE: this is a standalone, single-marginal generator and yields a
        beta-ONLY sub-matrix (one row per BETA_PD factor, indexed 0..n_beta-1) —
        its rows do NOT align to full-universe UIX. For the JTD pipeline, where a
        ``BetaBinomialJtdBlock.beta_uix`` indexes the full ``(F, S)`` matrix, use
        ``MixedCreditFactorGenerator.from_universe`` instead, which fills the
        beta rows in place at their universe UIX. This generator is kept as the
        single-marginal primitive (mirroring the Gaussian/Uniform generators).

        ``available_factor_ids``, ``alpha`` and ``beta`` are all the BETA_PD
        subset, aligned by construction.
        """
        return cls(
            available_factor_ids=universe.factor_ids[universe.beta_pd_uix],
            alpha=universe.beta_alpha,
            beta=universe.beta_beta,
            total_scenarios=total_scenarios,
            seed=seed,
        )

    def iter_batches(
        self,
        *,
        batch_size: int,
    ) -> Iterator[CreditFactorBatch]:

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rng = np.random.default_rng(self.seed)

        n_factors = int(self.available_factor_ids.shape[0])

        buffer = np.empty(
            (batch_size, n_factors),
            dtype=np.float64,
            order="C",
        )
        # Reused scratch for the Gamma-ratio Beta draw (zero-allocation).
        g1_scratch = np.empty(batch_size, dtype=np.float64)
        g2_scratch = np.empty(batch_size, dtype=np.float64)

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)

            view = buffer[:bs, :]

            # Beta(alpha, beta) = G1 / (G1 + G2) with G1 ~ Gamma(alpha),
            # G2 ~ Gamma(beta). standard_gamma supports out=, so no per-column
            # allocation (rng.beta has no out=).
            for j in range(n_factors):
                g1 = g1_scratch[:bs]
                g2 = g2_scratch[:bs]
                rng.standard_gamma(self.alpha[j], out=g1)
                rng.standard_gamma(self.beta[j], out=g2)
                np.add(g1, g2, out=g2)
                np.divide(g1, g2, out=g1)
                view[:, j] = g1

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )

            scenario_offset += bs
