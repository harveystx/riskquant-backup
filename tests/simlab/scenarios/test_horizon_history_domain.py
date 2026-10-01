# tests/simlab/scenarios/test_horizon_history_domain.py
"""
G5: horizon domain against the history length T (D026 F3 clarification).

Rolling-window engines (Historical, FilteredHistorical) have
``total_scenarios = T - h`` and the domain ``1 <= h <= T - 1``. Path engines
(SBB, FilteredBootstrap) compose h drawn daily moves from moves ``1 .. T - 1``
and accept every ``h >= 1``, including ``h >= T``.

Oracles use raw levels and ``math`` only, with two stated exceptions:

* SBB oracles are index-free. A constant-move series gives h equal moves on
  every path. One circularly wrapped block of ``m (T - 1)`` moves visits every
  daily move exactly m times, whatever its start.
* Filtered oracles read the per-day filtered move from the ``h = 1``
  FilteredHistorical stream. The filter itself is pinned independently by
  ``test_h_day_independent_oracle.py`` (G2). This file pins the h-domain and
  the path composition over the documented draw
  ``default_rng(seed).integers(1, T, size=(S, h))``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator

LOG, SIMPLE, DIFF, ABS = 1, 2, 3, 4
_T = 7
# float64 arithmetic rounded once to the float32 shock output.
_RTOL, _ATOL = 1e-6, 1e-7
_F32_EPS = 2.0**-24

_POS = [100.0, 103.0, 99.0, 104.0, 101.0, 106.0, 102.0]
_NEG = [-100.0, -96.0, -103.0, -99.0, -108.0, -101.0, -110.0]
_LEVELS = (_POS, _NEG)
_REQUEST = ((0, LOG), (0, SIMPLE), (0, DIFF), (0, ABS), (1, DIFF), (1, ABS))

# Constant daily move per requested (factor, chart); factor 2 is strictly
# negative with a constant ABS move of 0.01.
_CONST_LEVELS = (
    [100.0 * 1.01**t for t in range(_T)],
    [50.0 + 2.0 * t for t in range(_T)],
    [-100.0 * 0.99**t for t in range(_T)],
)
_CONST_REQUEST = ((0, LOG), (0, SIMPLE), (0, ABS), (1, DIFF), (2, ABS))


def _matrix(cols: tuple[list[float], ...]) -> RiskFactorMatrix:
    values = np.asfortranarray(np.array(cols, dtype=np.float64).T)
    return RiskFactorMatrix(
        dates=np.arange(values.shape[0], dtype=np.int32),
        factor_ids=np.arange(values.shape[1], dtype=np.int32),
        values=values,
    )


_MATRIX = _matrix(_LEVELS)
_CONST = _matrix(_CONST_LEVELS)


def _move(v1: float, v0: float, chart: int) -> float:
    """Daily chart move ``v0 -> v1`` (D025, D088 §2)."""
    if chart == LOG:
        return math.log(v1 / v0)
    if chart == SIMPLE:
        return v1 / v0 - 1.0
    if chart == DIFF:
        return v1 - v0
    return (v1 - v0) / abs(v0)


def _compose(moves: list[float], chart: int) -> float:
    """D025: LOG and DIFF add daily moves; SIMPLE and ABS compound them."""
    if chart in (SIMPLE, ABS):
        acc = 1.0
        for r in moves:
            acc *= 1.0 + r
        return acc - 1.0
    return math.fsum(moves)


def _daily(levels: list[float], chart: int) -> list[float]:
    return [_move(levels[t], levels[t - 1], chart) for t in range(1, len(levels))]


def _stream(
    gen: object, request: tuple[tuple[int, int], ...]
) -> npt.NDArray[np.float64]:
    parts = [
        np.array(b.shocks, dtype=np.float64, copy=True)
        for b in gen.iter_batches(  # type: ignore[attr-defined]
            driver_rf_ids=np.array([d for d, _ in request], dtype=np.int32),
            chart_ids=np.array([c for _, c in request], dtype=np.int8),
            batch_size=4,
        )
    ]
    return np.vstack(parts)


def _filtered_daily_moves() -> npt.NDArray[np.float64]:
    """Row t holds the filtered daily move z[t], t = 1 .. T-1; row 0 is unused."""
    one_day = _stream(FilteredHistoricalGenerator(matrix=_MATRIX, horizon=1), _REQUEST)
    return np.vstack([np.zeros((1, len(_REQUEST))), one_day])


def _atol(z: npt.NDArray[np.float64], h: int) -> float:
    """h float32-rounded daily moves plus one float32 output rounding."""
    return 4.0 * h * float(np.abs(z).max()) * _F32_EPS + _ATOL


def _const_move(f: int, c: int) -> float:
    moves = _daily(_CONST_LEVELS[f], c)
    assert max(moves) - min(moves) <= 1e-12 * max(1.0, abs(moves[0]))
    return moves[0]


# ------------------------------------------------- rolling-window engines ----
def test_historical_horizon_t_minus_one_is_one_full_window() -> None:
    gen = HistoricalGenerator(matrix=_MATRIX, horizon=_T - 1)
    assert gen.total_scenarios == 1
    want = [[_move(_LEVELS[f][_T - 1], _LEVELS[f][0], c) for f, c in _REQUEST]]
    np.testing.assert_allclose(_stream(gen, _REQUEST), want, rtol=_RTOL, atol=_ATOL)


def test_fhs_horizon_t_minus_one_composes_every_filtered_move() -> None:
    gen = FilteredHistoricalGenerator(matrix=_MATRIX, horizon=_T - 1)
    assert gen.total_scenarios == 1
    z = _filtered_daily_moves()
    want = [[_compose(list(z[1:, k]), c) for k, (_, c) in enumerate(_REQUEST)]]
    got = _stream(gen, _REQUEST)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=_atol(z, _T - 1))


@pytest.mark.parametrize("h", [_T, _T + 1])
@pytest.mark.parametrize("gen_cls", [HistoricalGenerator, FilteredHistoricalGenerator])
def test_rolling_window_engines_refuse_horizon_at_or_above_t(
    gen_cls: type, h: int
) -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        gen_cls(matrix=_MATRIX, horizon=h)


# ----------------------------------------------------------- path engines ----
@pytest.mark.parametrize("h", [_T, _T + 1, 3 * _T])
def test_sbb_accepts_horizon_at_and_above_t(h: int) -> None:
    n = 16
    gen = StationaryBlockBootstrapGenerator(
        matrix=_CONST, total_scenarios=n, expected_block_size=3.0, horizon=h, seed=11
    )
    want = [_compose([_const_move(f, c)] * h, c) for f, c in _CONST_REQUEST]
    got = _stream(gen, _CONST_REQUEST)
    assert got.shape == (n, len(_CONST_REQUEST))
    np.testing.assert_allclose(got, np.tile(want, (n, 1)), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("cycles", [2, 3])
def test_sbb_wrapped_block_visits_every_daily_move_once_per_cycle(
    cycles: int,
) -> None:
    """One block per path (huge mean block): h = cycles * (T - 1) > T."""
    n, h = 16, cycles * (_T - 1)
    gen = StationaryBlockBootstrapGenerator(
        matrix=_MATRIX, total_scenarios=n, expected_block_size=1e9, horizon=h, seed=29
    )
    want = [_compose(_daily(_LEVELS[f], c) * cycles, c) for f, c in _REQUEST]
    got = _stream(gen, _REQUEST)
    np.testing.assert_allclose(got, np.tile(want, (n, 1)), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("h", [_T, 2 * _T + 1])
def test_filtered_bootstrap_accepts_horizon_at_and_above_t(h: int) -> None:
    n, seed = 16, 23
    gen = FilteredBootstrapGenerator(
        matrix=_MATRIX, total_scenarios=n, horizon=h, seed=seed
    )
    z = _filtered_daily_moves()
    days = np.random.default_rng(seed).integers(
        low=1, high=_T, size=(n, h), dtype=np.int32
    )
    # h >= T: every path repeats at least one daily move (pigeonhole).
    assert all(len(set(row.tolist())) < h for row in days)
    want = [
        [
            _compose([float(z[int(t), k]) for t in row], c)
            for k, (_, c) in enumerate(_REQUEST)
        ]
        for row in days
    ]
    got = _stream(gen, _REQUEST)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=_atol(z, h))
