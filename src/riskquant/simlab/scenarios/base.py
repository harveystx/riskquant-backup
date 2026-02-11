# riskquant/simlab/scenarios/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class ScenarioSetBatch:
    """
    A batch of scenarios represented in shock-space.

    shocks: (B, K) float32/64
        Shock matrix for this batch, in shock-key space
        (unique keys = (source_factor_id, required_shift_type_id)).

    required_to_shock_uix: (Nreq,) int32
        For each required factor (ordered by required_factor_ids contract),
        index into K shock columns.

    shock_scale: (Nreq,) float32
        Scaling to apply to shocks at application time (per required factor).

    required_shift_type_id: (Nreq,) int8
        Shift type id per required factor (determines how to apply shock to snapshot level).

    scenario_offset: int
        Global scenario offset (0-based) of this batch in the full scenario stream.

    scenario_count_total: Optional[int]
        Total scenario count if known (e.g., historical T-horizon), else None.
    """

    shocks: np.ndarray
    required_to_shock_uix: np.ndarray
    shock_scale: np.ndarray
    required_shift_type_id: np.ndarray
    scenario_offset: int
    scenario_count_total: Optional[int] = None


class ScenarioGenerator(Protocol):
    """
    Streaming-first scenario generation.
    """

    def iter_batches(
        self,
        *,
        required_factor_ids: np.ndarray,  # (Nreq,) int32
        required_shift_type_id: np.ndarray,  # (Nreq,) int8
        source_factor_ids: np.ndarray,  # (Nreq,) int32 (from ProxyResolution)
        shock_scale: np.ndarray,  # (Nreq,) float32 (from ProxyResolution)
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:
        ...


def generate_all(
    gen: ScenarioGenerator,
    *,
    required_factor_ids: np.ndarray,
    required_shift_type_id: np.ndarray,
    source_factor_ids: np.ndarray,
    shock_scale: np.ndarray,
    batch_size: int,
    max_total_scenarios: int = 50_000,
) -> ScenarioSetBatch:
    """
    Convenience materializer (for notebooks/tests).
    Raises if scenario count exceeds max_total_scenarios.
    """
    batches = list(
        gen.iter_batches(
            required_factor_ids=required_factor_ids,
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            shock_scale=shock_scale,
            batch_size=batch_size,
        )
    )
    if not batches:
        raise ValueError("ScenarioGenerator produced no batches.")

    total = 0
    K = batches[0].shocks.shape[1]
    for b in batches:
        if b.shocks.shape[1] != K:
            raise ValueError("Inconsistent K across scenario batches.")
        total += b.shocks.shape[0]

    if total > max_total_scenarios:
        raise ValueError(
            f"Refusing to materialize {total} scenarios in memory; "
            f"max_total_scenarios={max_total_scenarios}."
        )

    shocks = np.ascontiguousarray(np.vstack([b.shocks for b in batches]))
    return ScenarioSetBatch(
        shocks=shocks,
        required_to_shock_uix=batches[0].required_to_shock_uix,
        shock_scale=batches[0].shock_scale,
        required_shift_type_id=batches[0].required_shift_type_id,
        scenario_offset=0,
        scenario_count_total=total,
    )
