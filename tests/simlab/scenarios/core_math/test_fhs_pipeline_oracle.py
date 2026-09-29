# tests/simlab/scenarios/core_math/test_fhs_pipeline_oracle.py
"""
R4 — independent oracle for the FHS composed pipeline.

Stage oracles derive results straight from the mathematical definitions and
use a different computational structure from production:
  - AR(1) residuals: numpy closed-form OLS + a 95% t-test gate (production uses
    a two-pass running-sum Numba kernel).
  - EWMA vol ratios: the explicit geometric-weighted variance sum
    var_t = lam^(t-1) e_1^2 + (1-lam) sum_{k=2..t} lam^(t-k) e_k^2
    (production uses the forward recursion).
  - Filtered shocks: per-day f_t = e_t * v_t aggregated over the deterministic
    HistoricalFHS window [s+1 .. s+h], per shift-type definition.

Covers h=1 and h>1.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import (
    EWMA_DECAY_FACTOR,
    MAX_EWMA_VOL_RATIO,
    MIN_EWMA_VARIANCE,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios_from_residuals,
)
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator

_T_CRIT = 1.96


def _returns(vals, st):
    v = np.asarray(vals, dtype=np.float64)
    prev, curr = v[:-1], v[1:]
    if st == SHIFT_LOG_RETURN_ID:
        return np.log(curr / prev)
    if st == SHIFT_SIMPLE_RETURN_ID:
        return curr / prev - 1.0
    if st == SHIFT_DIFFERENCE_ID:
        return curr - prev
    if st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
        return (curr - prev) / np.abs(prev)
    raise AssertionError(st)


def _ar1_residuals_oracle(vals, st):
    """Closed-form OLS of r_t on r_{t-1} with a 95% t-test de-mean gate."""
    v = np.asarray(vals, dtype=np.float64)
    t_max = v.shape[0]
    res = np.zeros(t_max, dtype=np.float64)
    if t_max < 2:
        return res
    r = _returns(v, st)  # r[0]=r_1 .. r[T-2]=r_{T-1}
    n_obs = t_max - 2
    if n_obs <= 0:  # only r_1 exists; production skips pass-2
        res[1] = r[0]
        return res
    x, y = r[:-1], r[1:]  # (n_obs,)
    mx, my = x.mean(), y.mean()
    var_x = np.sum((x - mx) ** 2)
    cov = np.sum((x - mx) * (y - my))
    beta = 0.0 if var_x < 1e-12 else cov / var_x
    alpha = my - beta * mx
    var_y = np.sum((y - my) ** 2)
    rss = var_y - beta * cov
    df = max(n_obs - 2, 1)
    sigma2 = rss / df
    t_stat = abs(beta / np.sqrt(sigma2 / var_x)) if var_x > 1e-12 else 0.0
    if t_stat < _T_CRIT:
        beta, alpha = 0.0, my
    res[1] = r[0] - alpha
    for t in range(2, t_max):
        res[t] = r[t - 1] - (alpha + beta * r[t - 2])
    return res


def _ewma_vol_ratios_oracle(res, decay=EWMA_DECAY_FACTOR, cap=MAX_EWMA_VOL_RATIO):
    """Explicit geometric-weighted EWMA variance -> clamped sqrt ratios."""
    t_max = res.shape[0]
    out = np.empty(t_max, dtype=np.float64)
    if t_max == 1:
        out[0] = 1.0
        return out
    lam = decay
    var = np.zeros(t_max)
    for t in range(1, t_max):  # var defined for t = 1 .. T-1
        s = res[1] ** 2 * lam ** (t - 1)
        for k in range(2, t + 1):
            s += (1.0 - lam) * lam ** (t - k) * res[k] ** 2
        var[t] = s
    var_today = var[t_max - 1]

    def _ratio(v):
        if v < MIN_EWMA_VARIANCE:
            return 1.0
        return min(np.sqrt(var_today / v), cap)

    out[0] = out[1] = _ratio(var[1])  # t=0,1 share the seed var_1 = e_1^2
    for t in range(2, t_max):
        out[t] = _ratio(var[t])
    return out


def _matrix(cols):
    t = len(cols[0])
    vals = np.asfortranarray(np.array(cols, dtype=np.float64).T)
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(len(cols), dtype=np.int32),
        values=vals,
    )


# A smooth-ish positive series with mild structure (two columns to avoid the
# single-column layout path; oracle compares column 0).
_C0 = [
    100.0, 101.5, 100.8, 102.3, 103.1, 102.0, 104.4, 103.9,
    105.7, 104.8, 106.9, 108.2, 107.1, 109.5, 108.3, 110.8,
]
_C1 = [50.0, 51.0, 50.4, 52.1, 51.3, 53.0, 52.2, 54.1,
       53.4, 55.0, 54.2, 56.3, 55.1, 57.0, 56.2, 58.1]

_SHIFTS = [
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
]


@pytest.mark.parametrize("st", _SHIFTS)
def test_ar1_residuals_match_oracle(st) -> None:
    m = _matrix([_C0, _C1])
    t = len(_C0)
    out = np.zeros((t, 2), dtype=np.float64, order="F")
    kernel_ar1_residuals(
        m.values,
        np.array([0, 1], np.int32),
        np.array([st, st], np.int8),
        out,
    )
    oracle = _ar1_residuals_oracle(_C0, st)
    np.testing.assert_allclose(out[:, 0], oracle, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("st", _SHIFTS)
def test_ewma_vol_ratios_match_oracle(st) -> None:
    m = _matrix([_C0, _C1])
    t = len(_C0)
    res = np.zeros((t, 2), dtype=np.float64, order="F")
    kernel_ar1_residuals(
        m.values, np.array([0, 1], np.int32), np.array([st, st], np.int8), res
    )
    ratios = np.zeros((t, 2), dtype=np.float64, order="F")
    kernel_ewma_vol_ratios_from_residuals(
        res, EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO, ratios
    )
    oracle = _ewma_vol_ratios_oracle(res[:, 0].copy())
    np.testing.assert_allclose(ratios[:, 0], oracle, rtol=1e-9, atol=1e-10)


def _drive_fhs(gen, st):
    chunks = []
    for b in gen.iter_batches(
        chart_ids=np.array([st, st], np.int8),
        driver_rf_ids=np.array([0, 1], np.int32),
        batch_size=4,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


def _fhs_shock_oracle(vals, st, h):
    res = _ar1_residuals_oracle(vals, st)
    vol = _ewma_vol_ratios_oracle(res)
    f = res * vol
    t = len(vals)
    s_total = t - h
    out = np.empty(s_total, dtype=np.float32)
    for s in range(s_total):
        idx = range(s + 1, s + h + 1)  # window [s+1 .. s+h]
        if st in (SHIFT_LOG_RETURN_ID, SHIFT_DIFFERENCE_ID):
            out[s] = np.float32(sum(f[i] for i in idx))
        else:  # SIMPLE / ABSOLUTE -> multiplicative
            prod = 1.0
            for i in idx:
                prod *= 1.0 + f[i]
            out[s] = np.float32(prod - 1.0)
    return out


@pytest.mark.parametrize("st", _SHIFTS)
@pytest.mark.parametrize("horizon", [1, 3])
def test_fhs_filtered_shocks_match_oracle(st, horizon) -> None:
    """End-to-end: production FHS shocks == independently composed
    residual*vol aggregated over deterministic windows, h=1 and h>1."""
    m = _matrix([_C0, _C1])
    gen = FilteredHistoricalGenerator(matrix=m, horizon=horizon)
    prod = _drive_fhs(gen, st)[:, 0]
    oracle = _fhs_shock_oracle(_C0, st, horizon)
    assert prod.shape == oracle.shape
    np.testing.assert_allclose(prod, oracle, rtol=1e-5, atol=1e-6)
