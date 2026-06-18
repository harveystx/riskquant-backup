# src/riskquant/simlab/scenarios/base.py
"""
Scenario-generator protocol and streaming-batch payload.

Design
------
`ScenarioGenerator` is a streaming protocol: generators yield
`ScenarioSetBatch` chunks of fixed `batch_size` instead of materialising
the full (T, K) shock matrix. The runner holds one pre-allocated
shocked-levels buffer and consumes each batch in place.

Universe resolution itself lives in `simlab.builder.shock_universe.
build_shock_universe` — generators call it directly. (The former
`resolve_shock_universe` wrapper here was a zero-logic, identical-
signature pass-through and has been removed.)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# ============================================================
# Streaming-batch payload
# ============================================================


@dataclass(frozen=True, slots=True)
class ScenarioSetBatch:
    """
    One streaming chunk of scenarios in shock-space.

    `shocks` is F-contiguous `(B, K_req)` so the apply-shocks kernel
    reads down the scenario axis with unit stride. `required_to_shock_uix`
    maps required-exposure rows back to columns of `shocks`.
    """

    shocks: np.ndarray
    required_to_shock_uix: np.ndarray
    shock_scale: np.ndarray
    required_shift_type_id: np.ndarray
    scenario_offset: int
    scenario_count_total: int | None = None


# ============================================================
# Generator protocol
# ============================================================


class ScenarioGenerator(Protocol):
    """Streaming-first scenario generation."""

    total_scenarios: int
    available_factor_ids: np.ndarray

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,  # (Nreq,) int8
        source_factor_ids: np.ndarray,  # (Nreq,) int32
        shock_scale: np.ndarray,  # (Nreq,) float32
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]: ...
