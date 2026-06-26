# src/riskquant/solutions/credit_var/_attribution.py
"""
Per-obligor loss attribution and the portfolio loss histogram.

Solution-internal numpy helpers over the raw ``(n_obligors, S)`` loss matrix. Two
complementary per-obligor views, both exact (no re-simulation — everything reuses
the loss matrix already in hand):

* **ES contribution** — the Euler tail allocation the aggregate layer defers. Each
  obligor's average loss over the engine's exact tail set (the worst
  ``tail_count = max(1, int((1 - alpha) * S))`` scenarios). These SUM to the
  portfolio ES, so the column foots. Answers "what share of the tail is obligor i".
* **Incremental VaR / ES (leave-one-out)** — the change in portfolio VaR/ES from
  removing the obligor outright: ``risk(L) - risk(L - L_i)``. We already have the
  N x S matrix, so this is just re-aggregating ``portfolio_losses - loss[i]`` and
  re-running the same tail kernel — O(N * S), no re-simulation. Not additive
  (diversification), but the intuitive "what if we exit this name" number. Reuses
  ``tail_var_es_by_group`` so it shares the reported VaR/ES convention exactly.

The leave-one-out pass is chunked over obligors so peak memory stays O(chunk * S)
rather than O(N * S), keeping faith with the engine's memory discipline.
"""

from __future__ import annotations

__all__ = [
    "ObligorAttribution",
    "obligor_tail_attribution",
    "loss_histogram",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.credit.aggregate.loss_metrics import (
    loss_var_es,
    tail_var_es_by_group,
)

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

# Cap the leave-one-out working set to ~50M float64 (~400 MB incl. the kernel's
# own scratchpad copy) per chunk.
_LOO_CELL_BUDGET = 50_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class ObligorAttribution:
    """Per-obligor arrays (aligned to the loss-matrix row order) + tail totals."""

    expected_loss: _F64  # (N,)  mean loss per obligor
    es_contribution: _F64  # (N,)  Euler ES contribution; sums to es_total
    incremental_var: _F64  # (N,)  VaR(L) - VaR(L - L_i), leave-one-out
    incremental_es: _F64  # (N,)  ES(L)  - ES(L - L_i),  leave-one-out
    es_total: float  # portfolio ES at the headline level (foots es_contribution)
    var_total: float  # portfolio VaR at the headline level


def obligor_tail_attribution(
    *,
    obligor_loss: _F64,
    portfolio_losses: _F64,
    confidence_level: float,
) -> ObligorAttribution:
    """Attribute portfolio tail risk to obligors at ``confidence_level``.

    ``obligor_loss`` is (N, S); ``portfolio_losses`` is its column sum (S,).
    """
    obligors, scenarios = int(obligor_loss.shape[0]), int(obligor_loss.shape[1])

    var_total, es_total = loss_var_es(portfolio_losses, confidence_level)

    # Euler ES contribution over the engine's exact tail set (worst tail_count).
    tail_count = max(1, int((1.0 - confidence_level) * scenarios))
    tail_idx = np.argpartition(portfolio_losses, scenarios - tail_count)[
        scenarios - tail_count :
    ]
    es_contribution = obligor_loss[:, tail_idx].mean(axis=1)
    expected_loss = obligor_loss.mean(axis=1)

    # Incremental (leave-one-out), chunked over obligors.
    incremental_var = np.empty(obligors, np.float64)
    incremental_es = np.empty(obligors, np.float64)
    chunk = max(1, _LOO_CELL_BUDGET // max(1, scenarios))
    for lo in range(0, obligors, chunk):
        hi = min(obligors, lo + chunk)
        without = np.ascontiguousarray(portfolio_losses[None, :] - obligor_loss[lo:hi])
        var_without, es_without = tail_var_es_by_group(without, confidence_level)
        incremental_var[lo:hi] = var_total - var_without
        incremental_es[lo:hi] = es_total - es_without

    return ObligorAttribution(
        expected_loss=np.ascontiguousarray(expected_loss),
        es_contribution=np.ascontiguousarray(es_contribution),
        incremental_var=incremental_var,
        incremental_es=incremental_es,
        es_total=float(es_total),
        var_total=float(var_total),
    )


# def loss_histogram(
#     portfolio_losses: _F64, *, nbins: int = 50
# ) -> tuple[list[float], list[int]]:
#     """Bin portfolio loss over ``[0, p99.9]`` (overflow folded into the last bin)
#     so the distribution chart isn't flattened by a handful of extreme scenarios.
#     Returns (bin_edges length nbins+1, counts length nbins)."""
#     upper = float(np.percentile(portfolio_losses, 99.9))
#     if upper <= 0.0:
#         upper = float(portfolio_losses.max())
#     if upper <= 0.0:
#         upper = 1.0
#     edges = np.linspace(0.0, upper, nbins + 1)
#     counts, _ = np.histogram(np.clip(portfolio_losses, 0.0, upper), bins=edges)
#     return [float(e) for e in edges], [int(c) for c in counts]


def loss_histogram(
    portfolio_losses: _F64,
    *,
    nbins: int = 50,
    display_percentile: float = 99.9,
) -> tuple[list[float], list[int]]:
    """
    Build a histogram suitable for plotting portfolio losses.

    The histogram spans [0, display_percentile]. Losses above the chosen
    percentile are accumulated into the final bin so that extreme tail events
    do not dominate the x-axis.
    """
    losses = np.asarray(portfolio_losses, dtype=np.float64)

    if losses.size == 0:
        raise ValueError("portfolio_losses must not be empty")

    xmax = float(np.percentile(losses, display_percentile))
    xmax = max(xmax, float(losses.max()), 1.0)

    edges = np.linspace(0.0, xmax, nbins + 1)

    counts, _ = np.histogram(
        losses,
        bins=edges,
    )

    overflow = losses > xmax
    if overflow.any():
        counts[-1] += int(overflow.sum())

    return [float(e) for e in edges], [int(c) for c in counts]
