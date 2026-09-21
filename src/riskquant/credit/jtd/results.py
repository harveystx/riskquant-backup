# src/riskquant/credit/jtd/results.py
"""Resolved JTD-loss result container."""

from __future__ import annotations

__all__ = [
    "JtdLossResult",
    "StreamingJtdResult",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class JtdLossResult:
    """
    Jump-to-default loss, resolved per obligor.

    Row i of `loss` is the obligor identified by `uid[i]`.
    """

    uid: npt.NDArray[np.int64]  # (N,)    obligor identifier
    loss: npt.NDArray[np.float64]  # (N, S)  default loss per obligor, per scenario


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class StreamingJtdResult:
    """
    Accumulated JTD loss from a streaming run — the per-scenario PORTFOLIO total,
    never the per-obligor ``(N, S)`` matrix.

    `portfolio_loss` is the ``(S,)`` total the aggregation layer consumes
    directly. `model_loss`, when collected, is ``(n_models, S)`` with rows in
    `model_order` (JtdModel ids); each row is that model's per-scenario loss, and
    the rows sum to `portfolio_loss`. Per-obligor attribution is intentionally not
    retained — that is the memory trade the streaming path makes.
    """

    portfolio_loss: npt.NDArray[np.float64]  # (S,)
    total_obligors: int
    n_scenarios: int
    model_loss: npt.NDArray[np.float64] | None = None  # (n_models, S) | None
    model_order: tuple[int, ...] = ()  # JtdModel ids aligned to model_loss rows
