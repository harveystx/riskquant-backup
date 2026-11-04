# src/riskquant/solutions/credit_var/_attribution.py
"""
Per-obligor loss attribution and the portfolio loss histogram.

Solution-internal numpy helpers over the raw ``(n_obligors, S)`` loss matrix. Two
complementary per-obligor views, both exact (no re-simulation — everything reuses
the loss matrix already in hand):

* **ES contribution** — the Euler tail allocation the aggregate layer defers. Each
  obligor's Euler allocation over the engine's exact fractional tail set,
  computed by the shared ``scenario_metrics`` contribution kernel. These SUM to
  the portfolio ES exactly (the column foots) for any (alpha, S). Answers
  "what share of the tail is obligor i".
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
    "loss_histogram",
    "obligor_tail_attribution",
    "obligor_ul_contribution",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.credit.aggregate.loss_metrics import (
    loss_var_es,
    loss_var_es_contribution,
    tail_var_es_by_group,
)

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

# Cap the leave-one-out working set to ~50M float64 (~400 MB incl. the kernel's
# own scratchpad copy) per chunk.
_LOO_CELL_BUDGET = 50_000_000


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ObligorAttribution:
    """Per-obligor arrays (aligned to the loss-matrix row order) + tail totals."""

    expected_loss: _F64  # (N,)  mean loss per obligor
    es_contribution: _F64  # (N,)  Euler ES contribution; sums to es_total
    incremental_var: _F64  # (N,)  VaR(L) - VaR(L - L_i), leave-one-out
    incremental_es: _F64  # (N,)  ES(L)  - ES(L - L_i),  leave-one-out
    # (N,)  Cov(L_i, L) / sigma_L. For loss_mean / loss_std_dev <= 1e10 the sum
    # foots to ul_total within the C.34 bound 1e-9 x loss_std_dev + sd(delta),
    # not exactly. Above that ratio C.34 makes no claim.
    ul_contribution: _F64
    es_total: float  # portfolio ES at the headline level (foots es_contribution)
    var_total: float  # portfolio VaR at the headline level
    # Portfolio UL: L.std(ddof=1), the RiskSummary.loss_std_dev value. For
    # loss_mean / loss_std_dev <= 1e10 the ul_contribution sum foots to it
    # within the C.34 bound 1e-9 x loss_std_dev + sd(delta); delta is the
    # rounding error of L. Above that ratio C.34 makes no claim.
    ul_total: float


def obligor_ul_contribution(
    obligor_loss: _F64,
    portfolio_losses: _F64,
) -> tuple[_F64, float]:
    """Per-obligor contribution to portfolio Unexpected Loss (the std-dev / Euler
    risk-contribution decomposition).

    RC_i = Cov(L_i, L) / sigma_L, which sums to sigma_L (since
    Sum_i Cov(L_i, L) = Var(L)). Unlike the ES contribution this is a
    distribution-wide measure, not a tail one: it answers "how much of the
    portfolio's loss volatility does obligor i drive", net of diversification.

    Rules (WS_CREDIT_VAR_API S7, C.33 to C.35)
    ------------------------------------------
    * Zero variance (C.33). When ``min(L) == max(L)``, the exact test of
      `run._distribution_metrics`, every contribution is 0.0 and the total is
      0.0. ``ddof = 1`` arithmetic on S equal values can give a nonzero std
      from rounding only; the exact test removes that noise.
    * Constant obligor (C.33). A row with ``min(L_i) == max(L_i)`` (for
      example a forced default) has zero covariance with any series, so its
      contribution is exactly 0.0: it adds to expected loss, not to
      unexpected loss.
    * Centred rows (C.34). Each covariance is
      ``sum_s (L_i,s - mean(L_i)) (L_s - mean(L)) / (S - 1)``. The half-centred
      form ``L_i . (L - mean(L))`` keeps the term ``mean(L_i) x sum_s(L_s -
      mean(L))``. That sum is zero in exact arithmetic but not in floating
      point, and a large constant mean(L_i) magnifies its rounding residue
      until the column does not foot.
    * Footing target (C.34, OD-3 (a) as amended in Revisions 8, 9 and 10).
      sigma_L is ``L.std(ddof=1)``, the same call that gives
      ``RiskSummary.loss_std_dev``, so the moment does not change. Domain:
      ``mean(L) / sigma_L <= 1e10``, with ``mean(L)`` the
      ``RiskSummary.loss_mean`` value. In the domain, the contributions foot
      to sigma_L within ``1e-9 x sigma_L + sd(delta)``, with ``delta_s`` the
      rounding error of the summed loss ``L_s`` against the exact sum of the
      obligor losses of scenario s (ddof = 1). sigma_L contains that
      rounding; no centred obligor row contains it, so it is the part of
      sigma_L that no obligor carries. By Cauchy-Schwarz its effect on the
      footing is at most sd(delta). Outside the domain, C.34 makes no claim.
      The rounding error of ``mean(L)`` scales with eps x mean(L). The
      centring puts its square into sigma_L and into no obligor row, and
      above the domain limit that part can exceed ``1e-9 x sigma_L``.
    * No BLAS (C.35). Every reduction is a NumPy ufunc reduction (``sum``,
      ``mean``, ``std``, ``min``, ``max``) along the contiguous scenario axis.
      Such a reduction uses one fixed pairwise summation order and no thread
      pool. A BLAS dot product (``@``, ``np.dot``) splits the sum by the
      BLAS thread count, so its bits depend on OPENBLAS/MKL/OMP_NUM_THREADS.

    Memory: the centred product is built per obligor chunk of
    ``_LOO_CELL_BUDGET // S`` rows, so extra memory stays O(chunk x S).

    Returns ``(ul_contribution (N,), ul_total)``.
    """
    scenarios = int(portfolio_losses.shape[0])
    obligors = int(obligor_loss.shape[0])
    contribution = np.zeros(obligors, np.float64)
    if scenarios < 2 or float(portfolio_losses.min()) == float(portfolio_losses.max()):
        return contribution, 0.0
    sigma = float(portfolio_losses.std(ddof=1))
    if not sigma > 0.0:  # min < max with an underflowed std: no UL to split
        return contribution, 0.0
    pf_centered = portfolio_losses - portfolio_losses.mean()
    scale = float(scenarios - 1) * sigma
    chunk = max(1, _LOO_CELL_BUDGET // scenarios)
    for lo in range(0, obligors, chunk):
        hi = min(obligors, lo + chunk)
        rows = obligor_loss[lo:hi]
        centred = rows - rows.mean(axis=1, keepdims=True)  # (chunk, S), C order
        np.multiply(centred, pf_centered[None, :], out=centred)
        covariance = centred.sum(axis=1)
        constant = rows.min(axis=1) == rows.max(axis=1)
        contribution[lo:hi] = np.where(constant, 0.0, covariance / scale)
    return contribution, sigma


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

    # Euler ES contribution via the shared scenario_metrics tail-allocation
    # kernel. It uses the SAME exact tail geometry as the headline ES (the same
    # var_index / full_tail_count / fractional_weight / tail_mass), so the column
    # foots to es_total even when (1 - alpha) * S is non-integral. (RQ-AUD-009.)
    _var_contribution, es_contribution = loss_var_es_contribution(
        obligor_loss, portfolio_losses, confidence_level
    )
    expected_loss = obligor_loss.mean(axis=1)

    # Distribution-wide UL (std-dev) contribution — foots to portfolio UL.
    ul_contribution, ul_total = obligor_ul_contribution(obligor_loss, portfolio_losses)

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
        ul_contribution=ul_contribution,
        es_total=float(es_total),
        var_total=float(var_total),
        ul_total=float(ul_total),
    )


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
