# tests/solutions/scenario_models/test_proxy_v1_seam_oracles.py
"""
D088 §11 acceptance oracles not covered elsewhere, on the production seam
(lower -> bind -> world columns -> Slice 2). Each reference is a closed form
evaluated in float64 on the stored float32 world value (D088 §13.6: rtol
1e-12, atol 1e-15); the §11 literals are checked at the precision the float32
world allows.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_world_column_seam import _col, _hist, _matrix, _model, _run  # noqa: E402

from riskquant.simlab.shocks.column_binding import bind_term_columns  # noqa: E402
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels  # noqa: E402

LOG, SIMPLE, DIFF = 1, 2, 3
RTOL, ATOL = 1e-12, 1e-15


def test_negative_loading_cross_identity() -> None:
    """EURCHF = USDCHF / USDEUR, loadings (+1, -1), A_0 = 0.90/0.92:
    A_s = 0.978261 e^0.029853 = 1.00791 = 0.918 / 0.9108."""
    chf, eur, cross = 10, 11, 12
    paths = {chf: [0.90, 0.918], eur: [0.92, 0.9108]}
    a0 = 0.90 / 0.92
    model = _model([(cross, [(chf, LOG, 1.0), (eur, LOG, -1.0)])])
    out, cols, wb, _ = _run(model, [cross], [LOG], [a0], _hist(), _matrix([chf, eur], paths))
    y_chf, y_eur = float(_col(wb, cols, chf, LOG)[0]), float(_col(wb, cols, eur, LOG)[0])
    np.testing.assert_allclose(out[0, 0], a0 * math.exp(1.0 * y_chf + -1.0 * y_eur), rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - 1.00791) < 1e-5
    assert abs(out[0, 0] / (0.918 / 0.9108) - 1.0) < 1e-6  # float32 world storage


def test_fx_log_identity_holds_only_with_the_consistent_base() -> None:
    """A = B * P, A_0 = B_0 * P_0 = 73.6: A_s = 82 * 0.93 = 76.26. With A_0 = 70
    the operator still runs; the level identity does not hold."""
    b, p, a = 20, 21, 22
    paths = {b: [80.0, 82.0], p: [0.92, 0.93]}
    model = _model([(a, [(b, LOG, 1.0), (p, LOG, 1.0)])])
    consistent, *_ = _run(model, [a], [LOG], [73.6], _hist(), _matrix([b, p], paths))
    assert abs(consistent[0, 0] / 76.26 - 1.0) < 1e-6
    inconsistent, *_ = _run(model, [a], [LOG], [70.0], _hist(), _matrix([b, p], paths))
    assert abs(inconsistent[0, 0] - 76.26) > 3.0
    np.testing.assert_allclose(inconsistent[0, 0], 70.0 * consistent[0, 0] / 73.6, rtol=1e-12)


def test_simple_beta_applies_to_the_world_horizon_variable() -> None:
    """h = 2, driver SIMPLE per-step (0.10, 0.10): the world column is the
    two-step simple return 0.21, so X = 0.5 * 0.21 = 0.105, not 0.1025."""
    b, a = 30, 31
    model = _model([(a, [(b, SIMPLE, 0.5)])])
    out, cols, wb, _ = _run(model, [a], [SIMPLE], [1.0], _hist(2), _matrix([b], {b: [100.0, 110.0, 121.0]}))
    y = float(_col(wb, cols, b, SIMPLE)[0])
    assert abs(y - 0.21) < 1e-7
    np.testing.assert_allclose(out[0, 0], 1.0 * (1.0 + 0.5 * y), rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - 1.105) < 1e-7 and abs(out[0, 0] - 1.1025) > 1e-3


@pytest.mark.parametrize("h", [1, 5])
def test_direct_shock_space_column_gives_the_same_level_at_any_declared_horizon(h: int) -> None:
    """Direct world column (no level path) Y = ln(84/80) at h = 1 and h = 5,
    beta = 0.95: A_s = 70 e^(0.95 Y) = 73.321 at both horizons."""
    b = 40
    y = np.asfortranarray(np.array([[math.log(84.0 / 80.0)]], dtype=np.float32))
    uix = bind_term_columns(
        term_driver_rf_ids=np.array([b], np.int32), term_chart_ids=np.array([LOG], np.int8),
        requested_horizon=h, world_column_driver_rf_ids=np.array([b], np.int32),
        world_column_chart_ids=np.array([LOG], np.int8), world_horizon=h,
    )
    out = np.empty((1, 1), dtype=np.float64)
    apply_linear_terms_to_base_levels(
        base_levels=np.array([70.0]), row_term_offsets=np.array([0, 1], np.int32),
        term_column_uix=uix, term_loadings=np.array([0.95]),
        target_shift_types=np.array([LOG], np.int8), shocks=y, out=out,
    )
    np.testing.assert_allclose(out[0, 0], 70.0 * math.exp(0.95 * float(y[0, 0])), rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - 73.321) < 1e-3


def test_a_direct_column_declared_at_one_horizon_does_not_bind_at_another() -> None:
    with pytest.raises(ValueError):
        bind_term_columns(
            term_driver_rf_ids=np.array([40], np.int32), term_chart_ids=np.array([LOG], np.int8),
            requested_horizon=1, world_column_driver_rf_ids=np.array([40], np.int32),
            world_column_chart_ids=np.array([LOG], np.int8), world_horizon=5,
        )
