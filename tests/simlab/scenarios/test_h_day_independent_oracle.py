# tests/simlab/scenarios/test_h_day_independent_oracle.py
"""
G2 hardening: independent h-day oracles for SBB, FHS and filtered bootstrap.

Every expected value here is computed from the raw synthetic levels with plain
Python loops and ``math`` (D091 §3 column meaning; D025 multi-day aggregation):

* a daily move at index ``t`` is the chart-native change ``V[t-1] -> V[t]``;
* LOG and DIFFERENCE h-day values ADD daily moves, SIMPLE COMPOUNDS them;
* on a contiguous block the composition equals the endpoint chart formula.

No generator, kernel, plan, filter or shock helper of RiskQuant computes an
expected value. The only shared inputs are the documented draw protocols
(``np.random.default_rng`` calls in documented order) and the documented
filter parameters. Each request mixes three factors and three charts in a
scrambled order, so a column that reads another column's chart, factor,
horizon or path fails here.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from riskquant.constants import (
    EWMA_DECAY_FACTOR,
    MAX_EWMA_VOL_RATIO,
    MIN_EWMA_VARIANCE,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator

LOG, SIMPLE, DIFF = 1, 2, 3
_T = 40
_HORIZONS = (2, 3, 5)
# Scrambled multi-factor, multi-chart request: (factor_id, chart).
_REQUEST = (
    (34, DIFF), (21, LOG), (55, SIMPLE),
    (21, DIFF), (34, LOG), (55, DIFF),
    (21, SIMPLE), (34, SIMPLE), (55, LOG),
)  # fmt: skip
# float32 output of float64 arithmetic: a few ULP. Every targeted mutation
# moves a column by more than 1e-3 relative.
_RTOL, _ATOL = 1e-6, 1e-7
# Documented AR(1) gate (simlab/scenarios/core_math/ar1_filter.py): keep beta
# only when |t| >= 1.96. The data below keep every |t| far from this value.
_T_CRIT = 1.96


def _levels() -> dict[int, list[float]]:
    g = np.random.default_rng(20261001)
    rw = 100.0 * np.exp(np.cumsum(np.r_[0.0, g.normal(0.0, 0.02, _T - 1)]))
    ar = [80.0]
    r_prev = 0.0
    for _ in range(_T - 1):
        r_prev = 0.85 * r_prev + g.normal(0.0, 0.004)
        ar.append(ar[-1] * math.exp(r_prev))
    mr = [50.0]
    for _ in range(_T - 1):
        mr.append(mr[-1] + g.normal(0.0, 1.0))
    return {21: [float(v) for v in rw], 34: ar, 55: mr}


_LEVELS = _levels()
_MATRIX = RiskFactorMatrix(
    dates=np.arange(_T, dtype=np.int32),
    factor_ids=np.array(sorted(_LEVELS), dtype=np.int32),
    values=np.asfortranarray(np.array([_LEVELS[f] for f in sorted(_LEVELS)]).T),
)
_DRIVERS = np.array([d for d, _ in _REQUEST], dtype=np.int32)
_CHARTS = np.array([c for _, c in _REQUEST], dtype=np.int8)


# ---------------------------------------------------------------- oracle ----
def _move(v1: float, v0: float, chart: int) -> float:
    if chart == LOG:
        return math.log(v1 / v0)
    if chart == SIMPLE:
        return v1 / v0 - 1.0
    return v1 - v0


def _compose(daily: list[float], chart: int) -> float:
    if chart == SIMPLE:
        acc = 1.0
        for r in daily:
            acc *= 1.0 + r
        return acc - 1.0
    acc = 0.0
    for r in daily:
        acc += r
    return acc


def _filtered_moves(v: list[float], chart: int) -> tuple[list[float], float]:
    """AR(1) residual times EWMA vol ratio per day, with the AR(1) |t|."""
    t_max = len(v)
    r = [0.0] + [_move(v[t], v[t - 1], chart) for t in range(1, t_max)]
    n = t_max - 2
    x = r[1 : t_max - 1]
    y = r[2:t_max]
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    syy = sum((b - my) ** 2 for b in y)
    beta = sxy / sxx
    t_stat = abs(beta) / math.sqrt(((syy - beta * sxy) / (n - 2)) / sxx)
    alpha = my - beta * mx
    if t_stat < _T_CRIT:
        alpha, beta = my, 0.0
    e = [0.0, r[1] - alpha] + [
        r[t] - (alpha + beta * r[t - 1]) for t in range(2, t_max)
    ]
    var = [0.0, e[1] * e[1]]
    for t in range(2, t_max):
        var.append(
            (1.0 - EWMA_DECAY_FACTOR) * e[t] * e[t] + EWMA_DECAY_FACTOR * var[-1]
        )
    today = var[-1]
    z = [0.0]
    for t in range(1, t_max):
        ratio = (
            1.0
            if var[t] < MIN_EWMA_VARIANCE
            else min(math.sqrt(today / var[t]), MAX_EWMA_VOL_RATIO)
        )
        z.append(e[t] * ratio)
    return z, t_stat


def _sbb_paths(seed: int, n: int, h: int, ebs: float) -> list[list[int]]:
    """Documented SBB protocol: geometric lengths, then starts, circular wrap."""
    rng = np.random.default_rng(seed)
    size = n * h
    p = 1.0 / ebs
    lengths = [1] * size if p >= 1.0 else [int(d) for d in rng.geometric(p, size=size)]
    starts = [int(s) for s in rng.integers(low=1, high=_T, size=size, dtype=np.int32)]
    paths, cursor = [], 0
    for _ in range(n):
        path: list[int] = []
        while len(path) < h:
            take = min(lengths[cursor], h - len(path))
            path += [1 + ((starts[cursor] + i - 1) % (_T - 1)) for i in range(take)]
            cursor += 1
        paths.append(path)
    return paths


def _stream(gen: object, batch_size: int) -> npt.NDArray[np.float64]:
    parts = [
        np.array(b.shocks, dtype=np.float64, copy=True)
        for b in gen.iter_batches(  # type: ignore[attr-defined]
            driver_rf_ids=_DRIVERS, chart_ids=_CHARTS, batch_size=batch_size
        )
    ]
    return np.vstack(parts)


def _assert_columns(got: npt.NDArray[np.float64], want: list[list[float]]) -> None:
    expected = np.array(want, dtype=np.float64)
    assert got.shape == expected.shape
    for k, (factor, chart) in enumerate(_REQUEST):
        np.testing.assert_allclose(
            got[:, k],
            expected[:, k],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"column {k} (factor {factor}, chart {chart})",
        )


# ----------------------------------------------------------------- tests ----
def test_oracle_data_keeps_every_ar1_gate_far_from_its_threshold() -> None:
    """Guards the oracle: no (factor, chart) sits near the |t| = 1.96 gate."""
    for factor, chart in _REQUEST:
        _, t_stat = _filtered_moves(_LEVELS[factor], chart)
        assert abs(t_stat - _T_CRIT) > 0.5, (factor, chart, t_stat)


@pytest.mark.parametrize("h", _HORIZONS)
def test_sbb_mixed_request_matches_the_documented_protocol(h: int) -> None:
    paths = _sbb_paths(seed=17, n=64, h=h, ebs=3.0)
    gen = StationaryBlockBootstrapGenerator(
        matrix=_MATRIX, total_scenarios=64, expected_block_size=3.0, horizon=h, seed=17
    )
    want = [
        [
            _compose([_move(_LEVELS[f][t], _LEVELS[f][t - 1], c) for t in path], c)
            for f, c in _REQUEST
        ]
        for path in paths
    ]
    _assert_columns(_stream(gen, batch_size=7), want)


@pytest.mark.parametrize("h", _HORIZONS)
def test_sbb_contiguous_block_equals_the_endpoint_window(h: int) -> None:
    """One block per path: a non-wrapping path IS the window (s-1, s+h-1]."""
    paths = _sbb_paths(seed=29, n=96, h=h, ebs=1e6)
    gen = StationaryBlockBootstrapGenerator(
        matrix=_MATRIX, total_scenarios=96, expected_block_size=1e6, horizon=h, seed=29
    )
    got = _stream(gen, batch_size=16)
    rows = [b for b, p in enumerate(paths) if p[-1] == p[0] + h - 1]
    assert len(rows) >= 48
    for k, (f, c) in enumerate(_REQUEST):
        want = [
            _move(_LEVELS[f][paths[b][-1]], _LEVELS[f][paths[b][0] - 1], c)
            for b in rows
        ]
        np.testing.assert_allclose(got[rows, k], want, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("h", _HORIZONS)
def test_fhs_mixed_request_matches_the_rolling_window_oracle(h: int) -> None:
    z = {(f, c): _filtered_moves(_LEVELS[f], c)[0] for f, c in _REQUEST}
    want = [
        [_compose(z[(f, c)][s + 1 : s + h + 1], c) for f, c in _REQUEST]
        for s in range(_T - h)
    ]
    gen = FilteredHistoricalGenerator(matrix=_MATRIX, horizon=h)
    _assert_columns(_stream(gen, batch_size=5), want)


@pytest.mark.parametrize("h", _HORIZONS)
def test_filtered_bootstrap_mixed_request_matches_the_iid_oracle(h: int) -> None:
    """Documented protocol: one integers((B, h)) draw over days [1, T-1]."""
    z = {(f, c): _filtered_moves(_LEVELS[f], c)[0] for f, c in _REQUEST}
    days = np.random.default_rng(23).integers(
        low=1, high=_T, size=(64, h), dtype=np.int32
    )
    want = [
        [_compose([z[(f, c)][int(t)] for t in row], c) for f, c in _REQUEST]
        for row in days
    ]
    gen = FilteredBootstrapGenerator(
        matrix=_MATRIX, total_scenarios=64, horizon=h, seed=23
    )
    _assert_columns(_stream(gen, batch_size=9), want)
