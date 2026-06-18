# src/riskquant/credit/factors/simulate/base.py
"""
Credit-factor scenario generation contract.

A credit-factor batch holds LATENT draws (systemic Z, idiosyncratic eps,
chi-square mixing W) — not market shocks. There is no base level, shift type,
or shock scale: latent factors are a different sample space, so this is a
credit-native protocol, deliberately NOT `simlab`'s `ScenarioGenerator`.

`build_credit_factor_matrix` assembles streamed batches into the C-contiguous
`(num_credit_factors, num_scenarios)` `CreditFactorMatrix` the JTD kernels read.
"""

from __future__ import annotations

__all__ = [
    "build_credit_factor_matrix",
]

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix


@dataclass(frozen=True, slots=True)
class CreditFactorBatch:
    """
    A streamed batch of latent credit-factor draws.

    `latent_factors` is `(batch_size, F)` — scenario-major, the streaming-
    friendly layout. `build_credit_factor_matrix` transposes it into the
    factor-major `CreditFactorMatrix`.
    """

    latent_factors: npt.NDArray[np.float64]  # (batch_size, F)
    scenario_offset: int
    scenario_count_total: int


class CreditFactorGenerator(Protocol):
    """Streaming-first credit-factor generation."""

    total_scenarios: int
    available_factor_ids: npt.NDArray[np.int32]

    def iter_batches(self, *, batch_size: int) -> Iterator[CreditFactorBatch]: ...


def build_credit_factor_matrix(
    batches: Iterable[CreditFactorBatch],
    *,
    num_factors: int,
    num_scenarios: int,
) -> CreditFactorMatrix:
    """
    Assemble streamed `(batch, F)` batches into a `(F, S)` C-contiguous matrix.

    The transpose happens once, at the edge, during assembly — the JTD kernels
    then index factor rows with the scenario axis contiguous.
    """

    # =====================================================================
    # DOD MEMORY LAYOUT: C-CONTIGUOUS
    # =====================================================================
    # We deliberately violate the generic "scenario arrays are F-order" rule.
    # The JTD kernels process obligor-by-obligor, loading factor rows via:
    #     z_row = latent_factors[uix]
    #
    # C-order guarantees `z_row` is contiguous. This allows the innermost
    # `for s in range(scenarios)` loop to execute as straight-line SIMD
    # over unit-stride memory. F-order would result in strided reads.
    # =====================================================================
    latent: npt.NDArray[np.float64] = np.empty(
        (num_factors, num_scenarios),
        dtype=np.float64,
        order="C",
    )
    for batch in batches:
        off = batch.scenario_offset
        bs = batch.latent_factors.shape[0]
        latent[:, off : off + bs] = batch.latent_factors.T

    return CreditFactorMatrix(latent_factors=latent)
