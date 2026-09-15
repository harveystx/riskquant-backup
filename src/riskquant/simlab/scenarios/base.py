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

    Canonical contract (validated at construction; all checks are O(1) — no
    data scans. The O(N) semantic invariant that every `required_to_shock_uix`
    entry indexes a valid shock column is guaranteed once upstream by
    `build_shock_universe`, so it is not re-checked per batch):
      - `shocks`                 : 2-D float32, F-contiguous, shape (B, K_req).
      - `required_to_shock_uix`  : 1-D int32,  length Nreq.
      - `shock_scale`            : 1-D floating (float32 canonical; float64
                                   tolerated — apply casts it), length Nreq.
      - `required_shift_type_id` : 1-D int8,   length Nreq.
      - `scenario_offset`        : int >= 0 (not bool).
      - `scenario_count_total`   : int > 0 or None (not bool).
    """

    shocks: np.ndarray
    required_to_shock_uix: np.ndarray
    shock_scale: np.ndarray
    required_shift_type_id: np.ndarray
    scenario_offset: int
    scenario_count_total: int | None = None

    def __post_init__(self) -> None:
        s = self.shocks
        if s.ndim != 2 or s.dtype != np.float32:
            raise TypeError(f"shocks must be 2-D float32, got {s.ndim}-D {s.dtype}.")
        if not s.flags.f_contiguous:
            raise ValueError("shocks must be F-contiguous (unit-stride scenario axis).")

        uix = self.required_to_shock_uix
        if uix.ndim != 1 or uix.dtype != np.int32:
            raise TypeError(
                f"required_to_shock_uix must be 1-D int32, got {uix.ndim}-D {uix.dtype}."
            )
        if not uix.flags.c_contiguous:
            raise ValueError("required_to_shock_uix must be contiguous.")

        scale = self.shock_scale
        if scale.ndim != 1 or not np.issubdtype(scale.dtype, np.floating):
            raise TypeError(
                f"shock_scale must be 1-D floating, got {scale.ndim}-D {scale.dtype}."
            )
        if not scale.flags.c_contiguous:
            raise ValueError("shock_scale must be contiguous.")

        shift = self.required_shift_type_id
        if shift.ndim != 1 or shift.dtype != np.int8:
            raise TypeError(
                f"required_shift_type_id must be 1-D int8, got {shift.ndim}-D {shift.dtype}."
            )
        if not shift.flags.c_contiguous:
            raise ValueError("required_shift_type_id must be contiguous.")

        n_req = uix.shape[0]
        if scale.shape[0] != n_req or shift.shape[0] != n_req:
            raise ValueError(
                "required_to_shock_uix, shock_scale, required_shift_type_id must "
                f"share length Nreq; got {uix.shape[0]}, {scale.shape[0]}, "
                f"{shift.shape[0]}."
            )

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
        required_shift_type_id: np.ndarray,  # (Nreq,) int8
        source_factor_ids: np.ndarray,  # (Nreq,) int32
        shock_scale: np.ndarray,  # (Nreq,) float32
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]: ...
