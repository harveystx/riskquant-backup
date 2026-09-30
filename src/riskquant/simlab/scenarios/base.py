# src/riskquant/simlab/scenarios/base.py
"""
Scenario-generator protocol and streaming-batch payload.

Design
------
`ScenarioGenerator` is a streaming protocol: generators yield
`ScenarioSetBatch` chunks of fixed `batch_size` instead of materialising
the full (T, K) matrix of world columns. The runner holds one pre-allocated
shocked-levels buffer and consumes each batch in place.

A generator is a scenario world (D088 §4). It is asked for K unique
`(driver_rf_id, chart_id)` world columns and returns them in request order at
its own horizon; `simlab.shock_plan.plan.build_world_column_plan` validates
the request and resolves the kernel layout.
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
    One streaming chunk of world columns.

    Column `k` of `shocks` is the world variable of request pair `k`
    (`(driver_rf_ids[k], chart_ids[k])` of `iter_batches`). `shocks` is
    F-contiguous `(B, K)` so consumers read down the scenario axis with unit
    stride.

    Lifetime: a yielded batch exposes generator-owned storage whose contents
    remain valid only until the next batch is produced (every generator
    family reuses one output buffer for a stream). A consumer that needs the
    values after that point must copy them.

    Canonical contract (validated at construction; O(1), no data scans):
      - `shocks`                 : 2-D float32, F-contiguous, shape (B, K).
      - `scenario_offset`        : int >= 0 (not bool).
      - `scenario_count_total`   : int > 0 or None (not bool).
    """

    shocks: np.ndarray
    scenario_offset: int
    scenario_count_total: int | None = None

    def __post_init__(self) -> None:
        s = self.shocks
        if s.ndim != 2 or s.dtype != np.float32:
            raise TypeError(f"shocks must be 2-D float32, got {s.ndim}-D {s.dtype}.")
        if not s.flags.f_contiguous:
            raise ValueError("shocks must be F-contiguous (unit-stride scenario axis).")
        if isinstance(self.scenario_offset, bool) or not isinstance(
            self.scenario_offset, (int, np.integer)
        ):
            raise TypeError("scenario_offset must be an integer.")
        if self.scenario_offset < 0:
            raise ValueError(
                f"scenario_offset must be >= 0, got {self.scenario_offset}."
            )
        if self.scenario_count_total is not None:
            if isinstance(self.scenario_count_total, bool) or not isinstance(
                self.scenario_count_total, (int, np.integer)
            ):
                raise TypeError("scenario_count_total must be an integer or None.")
            if self.scenario_count_total <= 0:
                raise ValueError(
                    f"scenario_count_total must be > 0, got {self.scenario_count_total}."
                )


def require_positive_int(value: object, name: str) -> int:
    """
    Strict discrete-quantity gate (F3). Accept a Python int or NumPy integer;
    reject bool and any float; require > 0. Return the canonical Python int so
    callers persist an integral value.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer (int or np.integer), got "
            f"{type(value).__name__}."
        )
    canonical = int(value)
    if canonical <= 0:
        raise ValueError(f"{name} must be > 0, got {canonical}.")
    return canonical


def require_float64_values(values: np.ndarray, name: str = "matrix.values") -> None:
    """
    Historical-family dtype boundary (F5, D017). The historical kernels are
    float64-pinned; a non-float64 matrix must fail here with a clear error
    rather than deep inside Numba. Do not cast — fix the data at its source.
    """
    if values.dtype != np.float64:
        raise TypeError(
            f"{name} must be float64 (D017: historical values are float64); got "
            f"{values.dtype}. Historical-family generators are float64-pinned and "
            "do not cast input data at the boundary."
        )


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
        driver_rf_ids: np.ndarray,  # (K,) int32, unique (d, tau) pairs with chart_ids
        chart_ids: np.ndarray,  # (K,) int8, world-column charts
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]: ...
