# tests/simlab/scenarios/test_shift_types.py
"""
R1 shift-taxonomy correctness.

Covers the completed shift taxonomy:
  * enum / constant / registry / valid-set,
  * ABSOLUTE_BASE_RETURN forward return against a definition-derived scalar
    oracle (incl. negative-base cases), and its historical and SBB generator
    endpoints,
  * shift-specific level-domain fail-fast, including NO_SHIFT (no domain check).

A world request never carries NO_SHIFT (D088 §13.4); `build_world_column_plan`
refuses it (tests/simlab/shock_plan). Level reconstruction for every chart is
tested with the Slice 2 kernel (tests/simlab/shocks).

Oracles are written from the mathematical definition, never by calling the
production kernel under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
    SHIFT_TYPE_NAME_TO_ID_REGISTRY,
    SHIFT_TYPE_NAMES,
    VALID_SHIFT_TYPE_IDS,
    ShiftType,
)
from riskquant.simlab.builder.shift_domain import validate_shift_domain
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import (
    StationaryBlockBootstrapGenerator,
    _kernel_fused_sbb_h_day_shock,
)
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator

# --------------------------------------------------------------------- taxonomy


def test_taxonomy_values_and_registry() -> None:
    assert int(ShiftType.NO_SHIFT) == -1
    assert int(ShiftType.LOG_RETURN) == 1
    assert int(ShiftType.SIMPLE_RETURN) == 2
    assert int(ShiftType.DIFFERENCE) == 3
    assert int(ShiftType.ABSOLUTE_BASE_RETURN) == 4
    assert frozenset({-1, 1, 2, 3, 4}) == VALID_SHIFT_TYPE_IDS
    assert set(SHIFT_TYPE_NAMES) == {
        "NO_SHIFT",
        "LOG_RETURN",
        "SIMPLE_RETURN",
        "DIFFERENCE",
        "ABSOLUTE_BASE_RETURN",
    }
    assert SHIFT_TYPE_NAME_TO_ID_REGISTRY["ABSOLUTE_BASE_RETURN"] == 4
    # 0 is intentionally not a valid id.
    assert 0 not in VALID_SHIFT_TYPE_IDS


# ------------------------------------------------------------- universe guard


# ----------------------------------------------- ABSOLUTE_BASE_RETURN oracle


def _forward_absolute_base(base: float, shocked: float) -> float:
    """Definition: r = (v_t - v_{t-1}) / |v_{t-1}|."""
    return (shocked - base) / abs(base)


def _reconstruct(base: float, shock: float, scale: float, st: int) -> float:
    """Independent scalar reconstruction oracle, from each shift's definition."""
    if st == int(SHIFT_LOG_RETURN_ID):
        return base * math.exp(shock * scale)
    if st == int(SHIFT_SIMPLE_RETURN_ID):
        return base * (1.0 + shock * scale)
    if st == int(SHIFT_DIFFERENCE_ID):
        return base + shock * scale
    if st == int(SHIFT_ABSOLUTE_BASE_RETURN_ID):
        return base + shock * scale * abs(base)
    if st == int(SHIFT_NO_SHIFT_ID):
        return base
    raise AssertionError(st)


@pytest.mark.parametrize(
    "base,shocked,expected_r",
    [
        (-100.0, -90.0, 0.10),    # negative base, up move -> +0.10
        (-100.0, -110.0, -0.10),  # negative base, down move -> -0.10
        (100.0, 110.0, 0.10),     # positive base
        (50.0, 45.0, -0.10),
    ],
)
def test_absolute_base_return_forward(base, shocked, expected_r) -> None:
    assert _forward_absolute_base(base, shocked) == pytest.approx(expected_r, abs=1e-12)


# ----------------------------------------------------------------- NO_SHIFT


def _matrix(levels_by_col: list[list[float]]) -> RiskFactorMatrix:
    arr = np.asfortranarray(np.array(levels_by_col, dtype=np.float64).T)  # (T,K)
    t = arr.shape[0]
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(arr.shape[1], dtype=np.int32),
        values=arr,
    )


def _drive(gen, factor_ids, shift_ids, batch_size=4) -> np.ndarray:
    # The generator reuses a single streaming `shocks` buffer, so each batch
    # must be copied before the next is drawn (a later batch can alias earlier
    # rows). Copy inside the loop, not via collect-then-concatenate.
    out = []
    for b in gen.iter_batches(
        chart_ids=np.asarray(shift_ids, dtype=np.int8),
        driver_rf_ids=np.asarray(factor_ids, dtype=np.int32),
        batch_size=batch_size,
    ):
        out.append(np.array(b.shocks, copy=True))
    return np.concatenate(out, axis=0)


def _pos_matrix() -> RiskFactorMatrix:
    # strictly positive levels: ABSOLUTE must equal SIMPLE everywhere.
    return _matrix(
        [
            [100.0, 101.0, 99.0, 103.0, 102.0, 105.0, 104.0, 108.0, 107.0, 110.0],
            [50.0, 51.0, 52.0, 50.5, 53.0, 52.0, 54.0, 55.0, 54.0, 56.0],
        ]
    )


def _mk(gen_cls, matrix, horizon):
    kw = dict(matrix=matrix, horizon=horizon)
    if gen_cls is StationaryBlockBootstrapGenerator:
        kw.update(total_scenarios=24, expected_block_size=2.0, seed=7)
    elif gen_cls is FilteredBootstrapGenerator:
        kw.update(total_scenarios=24, seed=7)
    return gen_cls(**kw)


# --- endpoint oracle (Historical), h=1 and h>1, positive AND negative -------


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize(
    "col",
    [
        [100.0, 98.0, 101.0, 103.0, 99.5, 104.0, 102.0, 105.0],   # positive
        [-100.0, -98.0, -101.0, -103.0, -99.5, -104.0, -102.0, -105.0],  # negative
    ],
)
def test_absolute_historical_endpoint_oracle(horizon, col) -> None:
    """Historical ABSOLUTE == (V[t] - V[t-h]) / |V[t-h]| (endpoint), any sign."""
    m = _matrix([col])
    shocks = _drive(
        HistoricalGenerator(matrix=m, horizon=horizon),
        [0],
        [SHIFT_ABSOLUTE_BASE_RETURN_ID],
    )[:, 0]
    t = len(col)
    oracle = np.array(
        [(col[e] - col[e - horizon]) / abs(col[e - horizon]) for e in range(horizon, t)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(shocks, oracle, rtol=1e-6, atol=1e-6)


# --- multiplicative oracle (SBB kernel, fixed path), h=1 and h>1, both signs -


def _sbb_kernel(col, idx, h, st):
    # Use a genuine (T,2) F-contiguous values (column 0 is the target) so the
    # test exercises ABSOLUTE aggregation without tripping the known
    # single-column (T,1) C/F bootstrap-layout defect (F4, R3 scope).
    t = len(col)
    values = np.zeros((t, 2), dtype=np.float64, order="F")
    values[:, 0] = col
    values[:, 1] = col
    idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int32))       # (B,h)
    b = idx.shape[0]
    out = np.zeros((b, 1), dtype=np.float32, order="F")
    _kernel_fused_sbb_h_day_shock(
        values,
        np.array([0], dtype=np.int32),
        np.array([st], dtype=np.int8),
        np.array([0], dtype=np.int32),
        idx,
        np.int64(h),
        np.int64(b),
        out,
    )
    return out[:, 0]


@pytest.mark.parametrize(
    "col",
    [
        [100.0, 101.0, 103.0, 102.0, 106.0, 104.0, 108.0],       # positive
        [-100.0, -96.0, -103.0, -99.0, -108.0, -101.0, -110.0],  # negative
    ],
)
def test_absolute_sbb_multiplicative_oracle(col) -> None:
    """SBB ABSOLUTE == prod_j (1 + (V[t1]-V[t1-1])/|V[t1-1]|) - 1 for h=1 and h>1."""
    rng = np.random.default_rng(0)
    for h in (1, 3):
        idx = rng.integers(low=1, high=len(col), size=(5, h)).astype(np.int32)
        got = _sbb_kernel(col, idx, h, int(SHIFT_ABSOLUTE_BASE_RETURN_ID))
        oracle = np.empty(idx.shape[0], dtype=np.float32)
        for b in range(idx.shape[0]):
            prod = 1.0
            for step in range(h):
                t1 = int(idx[b, step])
                prod *= 1.0 + (col[t1] - col[t1 - 1]) / abs(col[t1 - 1])
            oracle[b] = np.float32(prod - 1.0)
        np.testing.assert_allclose(got, oracle, rtol=1e-6, atol=1e-6)


def test_absolute_and_simple_diverge_only_for_multi_day_and_negative() -> None:
    """Contract witnesses: additive is wrong; endpoint != multiplicative on
    negative multi-day paths; all agree at h=1."""
    # h=2 positive: multiplicative == endpoint == SIMPLE (0.21), additive (0.20) differs.
    col = [100.0, 110.0, 121.0]
    mult = 1.0
    for j in (1, 2):
        mult *= 1.0 + (col[j] - col[j - 1]) / abs(col[j - 1])
    mult -= 1.0
    endpoint = (col[2] - col[0]) / abs(col[0])
    simple = col[2] / col[0] - 1.0
    additive = sum((col[j] - col[j - 1]) / abs(col[j - 1]) for j in (1, 2))
    assert mult == pytest.approx(0.21)
    assert endpoint == pytest.approx(0.21)
    assert simple == pytest.approx(0.21)
    assert additive == pytest.approx(0.20)  # the rejected convention
    # h=2 negative: endpoint, multiplicative, additive all differ.
    neg = [-100.0, -90.0, -80.0]
    e = (neg[2] - neg[0]) / abs(neg[0])
    m = 1.0
    for j in (1, 2):
        m *= 1.0 + (neg[j] - neg[j - 1]) / abs(neg[j - 1])
    m -= 1.0
    assert e == pytest.approx(0.20)
    assert m == pytest.approx(0.222222, abs=1e-6)
    assert e != pytest.approx(m)
    # h=1: single-period definition, all conventions coincide.
    one = [-100.0, -90.0]
    assert (one[1] - one[0]) / abs(one[0]) == pytest.approx(0.10)


# --- the canonical invariant: positive levels => ABSOLUTE == SIMPLE ---------


@pytest.mark.parametrize(
    "gen_cls",
    [
        HistoricalGenerator,
        FilteredHistoricalGenerator,
        StationaryBlockBootstrapGenerator,
        FilteredBootstrapGenerator,
    ],
)
@pytest.mark.parametrize("horizon", [1, 3])
def test_absolute_equals_simple_on_positive_levels(gen_cls, horizon) -> None:
    """Canonical contract: on strictly positive levels, ABSOLUTE_BASE_RETURN is
    identical to SIMPLE_RETURN for every generator and every horizon. Bootstrap
    generators use a fixed seed so both shift types draw the same path."""
    m = _pos_matrix()
    simple = _drive(_mk(gen_cls, m, horizon), [0, 1], [SHIFT_SIMPLE_RETURN_ID] * 2)
    absolute = _drive(
        _mk(gen_cls, m, horizon), [0, 1], [SHIFT_ABSOLUTE_BASE_RETURN_ID] * 2
    )
    np.testing.assert_allclose(absolute, simple, rtol=1e-6, atol=1e-6)


def test_absolute_diverges_from_simple_on_negative_levels() -> None:
    """Where SIMPLE is ill-posed (negative base), ABSOLUTE is the sign-robust
    variant and MUST differ. Historical direct path, h=1."""
    m = _matrix([[-100.0, -90.0, -110.0, -95.0, -105.0, -100.0]])
    simple = _drive(HistoricalGenerator(matrix=m, horizon=1), [0], [SHIFT_SIMPLE_RETURN_ID])[:, 0]
    absolute = _drive(
        HistoricalGenerator(matrix=m, horizon=1), [0], [SHIFT_ABSOLUTE_BASE_RETURN_ID]
    )[:, 0]
    assert np.any(np.abs(simple - absolute) > 1e-3)
    # ABSOLUTE endpoint oracle still holds on the negative path.
    col = [-100.0, -90.0, -110.0, -95.0, -105.0, -100.0]
    oracle = np.array(
        [(col[i] - col[i - 1]) / abs(col[i - 1]) for i in range(1, len(col))],
        dtype=np.float32,
    )
    np.testing.assert_allclose(absolute, oracle, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------- domain fail-fast


def _dom(values, cols, sts, factor_ids=None):
    values = np.asfortranarray(np.array(values, dtype=np.float64))
    k = values.shape[1]
    validate_shift_domain(
        values=values,
        cols_sorted=np.asarray(cols, dtype=np.int32),
        shift_type_ids=np.asarray(sts, dtype=np.int8),
        factor_ids=np.arange(k, dtype=np.int32) if factor_ids is None else factor_ids,
    )


def test_domain_log_requires_positive() -> None:
    with pytest.raises(ValueError, match="LOG_RETURN"):
        _dom([[100.0], [-1.0], [50.0]], [0], [SHIFT_LOG_RETURN_ID])


def test_domain_simple_and_absolute_reject_zero() -> None:
    for st in (SHIFT_SIMPLE_RETURN_ID, SHIFT_ABSOLUTE_BASE_RETURN_ID):
        with pytest.raises(ValueError):
            _dom([[100.0], [0.0], [50.0]], [0], [st])


def test_domain_difference_and_absolute_admit_negative() -> None:
    _dom([[-100.0], [-101.0], [-99.0]], [0], [SHIFT_DIFFERENCE_ID])
    _dom([[-100.0], [-101.0], [-99.0]], [0], [SHIFT_ABSOLUTE_BASE_RETURN_ID])


def test_domain_no_shift_skips_all_checks() -> None:
    # Non-finite and zero are fine for a held-fixed factor (never read).
    _dom([[np.inf], [0.0], [-3.0]], [0], [SHIFT_NO_SHIFT_ID])


def test_domain_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _dom([[100.0], [np.nan], [50.0]], [0], [SHIFT_DIFFERENCE_ID])
