# src/riskquant/credit/jtd/results.py
"""Resolved JTD-loss result container."""

from __future__ import annotations

__all__ = [
    "JtdLossResult",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class JtdLossResult:
    """
    Jump-to-default loss, resolved per obligor.

    Row i of `loss` is the obligor identified by `uid[i]`.
    """

    uid: npt.NDArray[np.int64]  # (N,)    obligor identifier
    loss: npt.NDArray[np.float64]  # (N, S)  default loss per obligor, per scenario
