# src/riskquant/credit/factors/simulate/base.py
"""
Credit-factor scenario generation contract.

A credit-factor batch holds LATENT draws (systemic Z, idiosyncratic eps,
chi-square mixing W) — not market shocks. There is no base level, shift type,
or shock scale: latent factors are a different sample space, so this is a
credit-native protocol, deliberately NOT `simlab`'s `ScenarioGenerator`.

`build_credit_factor_matrix` assembles streamed batches into the C-contiguous
`(num_credit_factors, num_scenarios)` `CreditFactorMatrix` the JTD kernels read.
The assembled matrix is allocated with `np.empty`, so the batches MUST partition
`[0, num_scenarios)` exactly; that partition is proved, not assumed (see
`riskquant.credit._scenario_coverage`).
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

from riskquant.credit._scenario_coverage import (
    assert_scenario_coverage_complete,
    new_scenario_coverage,
    record_scenario_span,
)
from riskquant.credit.factors.state import CreditFactorMatrix


@dataclass(frozen=True, slots=True, eq=False)
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

    Contract
    --------
    The batches MUST partition ``[0, num_scenarios)`` exactly: no gap, no
    overlap, no overrun, no truncation. The destination is `np.empty`, so an
    unwritten scenario column is uninitialised heap that would flow straight
    into the copula kernels. Coverage is therefore tracked per scenario and
    verified before the matrix is handed back.

    Raises
    ------
    ValueError
        If a batch's factor axis disagrees with ``num_factors``, or the batches
        do not partition ``[0, num_scenarios)``.
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
    ctx = "build_credit_factor_matrix"
    coverage = new_scenario_coverage(num_scenarios, ctx=ctx)

    for batch in batches:
        off = int(batch.scenario_offset)
        bs = int(batch.latent_factors.shape[0])
        batch_factors = int(batch.latent_factors.shape[1])
        if batch_factors != num_factors:
            raise ValueError(
                f"{ctx}: batch at scenario offset {off} carries "
                f"{batch_factors} factor column(s), expected {num_factors}."
            )
        record_scenario_span(coverage, offset=off, count=bs, ctx=ctx)
        latent[:, off : off + bs] = batch.latent_factors.T

    assert_scenario_coverage_complete(coverage, ctx=ctx)

    return CreditFactorMatrix(latent_factors=latent)
