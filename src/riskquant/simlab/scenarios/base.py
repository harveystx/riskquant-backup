# src/riskquant/simlab/scenarios/base.py
"""
Scenario-generator protocol, batch container, and the one canonical entry
point for resolving a required-factor universe into a ShockUniverse +
ShockComputePlan.

Design
------
`ScenarioGenerator` is a streaming protocol: generators yield
`ScenarioSetBatch` chunks of fixed `batch_size` instead of materialising
the full (T, K) shock matrix. The runner holds one pre-allocated
shocked-levels buffer and consumes each batch in place.

`resolve_shock_universe` replaces the old dict-based
`build_factor_id_to_col_index` + `build_shock_key_plan` helpers. It
takes `RiskFactorMatrix.factor_ids` (sorted ascending int32) directly
and uses `np.searchsorted` for O(K log N) vectorised resolution.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from riskquant.simlab.builder.shock_universe import (
    ShockUniverse,
    build_shock_universe,
)

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


# ============================================================
# Vectorised universe resolution
# ============================================================


def resolve_shock_universe(
    *,
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
    available_factor_ids: npt.NDArray[np.int32],
) -> ShockUniverse:
    """
    Build a `ShockUniverse` from the run's resolved-source universe and
    the generator's available-factor index.

    `available_factor_ids` MUST be sorted ascending and unique (the
    `RiskFactorMatrix` invariant). This lets us replace the dict-based
    lookup with one vectorised `np.searchsorted` inside
    `build_shock_universe`.

    The previous two-stage helper pair (`build_factor_id_to_col_index`
    + `build_shock_key_plan`) is gone — both were Python-loop with
    dict-lookup anti-patterns. This is the single entry point all
    generators use.
    """
    return build_shock_universe(
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=source_factor_ids,
        available_factor_ids=available_factor_ids,
    )
