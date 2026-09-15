# tests/simlab/shocks/test_apply_shocks_oracle.py
"""
R4 — independent oracle for apply-shocks level reconstruction.

The oracle derives each shocked level directly from the mathematical
definition of the shift type, vectorised in NumPy — it does not reuse the
production per-exposure Numba loop. Covers every shift id, positive and
negative base levels, and a non-unit proxy `shock_scale`.

Definitions (v = base level, s = shock, c = scale):
    NO_SHIFT              -> v
    LOG_RETURN            -> v * exp(s * c)
    SIMPLE_RETURN         -> v * (1 + s * c)
    DIFFERENCE            -> v + s * c
    ABSOLUTE_BASE_RETURN  -> v + s * c * |v|
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels


def _oracle(base: float, shock_col: np.ndarray, scale: float, st: int) -> np.ndarray:
    s = shock_col.astype(np.float64)
    if st == SHIFT_NO_SHIFT_ID:
        return np.full(s.shape, base, dtype=np.float64)
    if st == SHIFT_LOG_RETURN_ID:
        return base * np.exp(s * scale)
    if st == SHIFT_SIMPLE_RETURN_ID:
        return base * (1.0 + s * scale)
    if st == SHIFT_DIFFERENCE_ID:
        return base + s * scale
    if st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
        return base + s * scale * abs(base)
    raise AssertionError(st)


_SHIFTS = [
    SHIFT_NO_SHIFT_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
]
_BASES = [100.0, -100.0, 0.5, -0.03, 250.0]  # positive and negative valid levels
_SCALES = [1.0, 0.5, 1.75]


@pytest.mark.parametrize("scale", _SCALES)
def test_apply_matches_oracle_all_shift_types_and_signs(scale: float) -> None:
    """Every (shift type, base sign) reconstructs to the definitional level."""
    n = len(_SHIFTS) * len(_BASES)
    b = 6
    # one distinct shock column per exposure (K = N); F-contiguous float32.
    rng = np.random.default_rng(0)
    shocks = np.asfortranarray(
        rng.uniform(-0.4, 0.4, size=(b, n)).astype(np.float32)
    )
    base_levels = np.empty(n, dtype=np.float64)
    shift_ids = np.empty(n, dtype=np.int8)
    uix = np.arange(n, dtype=np.int32)
    scales = np.full(n, scale, dtype=np.float32)

    i = 0
    for st in _SHIFTS:
        for base in _BASES:
            base_levels[i] = base
            shift_ids[i] = st
            i += 1

    batch = ScenarioSetBatch(
        shocks=shocks,
        required_to_shock_uix=uix,
        shock_scale=scales,
        required_shift_type_id=shift_ids,
        scenario_offset=0,
        scenario_count_total=b,
    )
    out = np.empty((n, b), dtype=np.float64, order="C")
    apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=out)

    i = 0
    for st in _SHIFTS:
        for base in _BASES:
            expected = _oracle(base, shocks[:, i], scale, int(st))
            np.testing.assert_allclose(out[i], expected, rtol=1e-12, atol=1e-12)
            i += 1


def test_absolute_equals_simple_on_positive_base_negative_differs() -> None:
    """ABSOLUTE == SIMPLE for base > 0; the two diverge for base < 0."""
    b = 5
    shocks = np.asfortranarray(
        np.linspace(-0.3, 0.3, b).reshape(b, 1).astype(np.float32)
    )
    scale = 1.25

    def _run(base, st):
        batch = ScenarioSetBatch(
            shocks=shocks,
            required_to_shock_uix=np.array([0], np.int32),
            shock_scale=np.array([scale], np.float32),
            required_shift_type_id=np.array([st], np.int8),
            scenario_offset=0,
        )
        out = np.empty((1, b), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(
            base_levels=np.array([base], np.float64), batch=batch, out=out
        )
        return out[0]

    pos_abs = _run(80.0, SHIFT_ABSOLUTE_BASE_RETURN_ID)
    pos_simple = _run(80.0, SHIFT_SIMPLE_RETURN_ID)
    np.testing.assert_allclose(pos_abs, pos_simple, rtol=1e-12, atol=1e-12)

    neg_abs = _run(-80.0, SHIFT_ABSOLUTE_BASE_RETURN_ID)
    neg_simple = _run(-80.0, SHIFT_SIMPLE_RETURN_ID)
    assert np.any(np.abs(neg_abs - neg_simple) > 1e-6)
    # ABSOLUTE on a negative base moves WITH the shock sign (|base| increment).
    np.testing.assert_allclose(
        neg_abs, -80.0 + shocks[:, 0].astype(np.float64) * scale * 80.0,
        rtol=1e-12, atol=1e-12,
    )
