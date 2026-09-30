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

from riskquant.simlab.scenarios.parametric_data import ParametricRiskData  # noqa: E402
from riskquant.simlab.shocks.column_binding import bind_term_columns  # noqa: E402
from riskquant.solutions.scenario_models.api import (  # noqa: E402
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix  # noqa: E402
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


def test_a_direct_column_declared_at_one_horizon_does_not_bind_at_another() -> None:
    with pytest.raises(ValueError):
        bind_term_columns(
            term_driver_rf_ids=np.array([40], np.int32), term_chart_ids=np.array([LOG], np.int8),
            requested_horizon=1, world_column_driver_rf_ids=np.array([40], np.int32),
            world_column_chart_ids=np.array([LOG], np.int8), world_horizon=5,
        )


# ---- horizon is observable: distinct y at h = 1 and h = 5 -------------------------------------


def _direct(y: float, h: int, requested: int) -> float:
    """A <- 0.95 * Y(40, LOG, h) against a direct world column worth y at h."""
    uix = bind_term_columns(
        term_driver_rf_ids=np.array([40], np.int32), term_chart_ids=np.array([LOG], np.int8),
        requested_horizon=requested, world_column_driver_rf_ids=np.array([40], np.int32),
        world_column_chart_ids=np.array([LOG], np.int8), world_horizon=h,
    )
    out = np.empty((1, 1), dtype=np.float64)
    apply_linear_terms_to_base_levels(
        base_levels=np.array([70.0]), row_term_offsets=np.array([0, 1], np.int32),
        term_column_uix=uix, term_loadings=np.array([0.95]),
        target_shift_types=np.array([LOG], np.int8),
        shocks=np.asfortranarray(np.array([[y]], dtype=np.float32)), out=out,
    )
    return float(out[0, 0])


def test_the_operator_reads_the_bound_horizon_value_and_refuses_another() -> None:
    """Direct columns: (40, LOG, 1) -> y1 = ln(81/80), (40, LOG, 5) -> y5 = ln(84/80).
    The same operator gives 70 e^(0.95 y1) against the h = 1 world, 70 e^(0.95 y5)
    = 73.321 against the h = 5 world, and refuses a mismatched binding."""
    y1, y5 = float(np.float32(math.log(81 / 80))), float(np.float32(math.log(84 / 80)))
    assert y1 != y5
    np.testing.assert_allclose(_direct(y1, 1, 1), 70.0 * math.exp(0.95 * y1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_direct(y5, 5, 5), 70.0 * math.exp(0.95 * y5), rtol=RTOL, atol=ATOL)
    assert abs(_direct(y5, 5, 5) - 73.321) < 1e-3 and abs(_direct(y1, 1, 1) - 73.321) > 1.0
    with pytest.raises(ValueError):
        _direct(y5, 5, 1)


def test_level_backed_columns_at_h1_and_h5_differ_through_the_seam() -> None:
    """One level path 80, 81, 82, 83, 84, 86: the h = 1 world's last column is
    ln(86/84), the h = 5 world's only column ln(86/80); the operator reads each."""
    b, a = 41, 42
    path = {b: [80.0, 81.0, 82.0, 83.0, 84.0, 86.0]}
    model = _model([(a, [(b, LOG, 0.95)])])
    out1, cols1, wb1, _ = _run(model, [a], [LOG], [70.0], _hist(1), _matrix([b], path))
    out5, cols5, wb5, _ = _run(model, [a], [LOG], [70.0], _hist(5), _matrix([b], path))
    y1_last = float(_col(wb1, cols1, b, LOG)[-1])
    y5 = float(_col(wb5, cols5, b, LOG)[0])
    assert abs(y1_last - math.log(86 / 84)) < 1e-7 and abs(y5 - math.log(86 / 80)) < 1e-7
    np.testing.assert_allclose(out1[0, -1], 70.0 * math.exp(0.95 * y1_last), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(out5[0, 0], 70.0 * math.exp(0.95 * y5), rtol=RTOL, atol=ATOL)
    assert out1.shape[1] == 5 and out5.shape[1] == 1


# ---- V1-08 (a): level-domain obligation only for level-backed columns that are used ---------


def _stream(gen, drivers, charts) -> np.ndarray:
    # A batch's arrays are valid only until the next batch is drawn (the
    # generator reuses its output buffer), so each batch is copied.
    return np.vstack([np.array(b.shocks, copy=True) for b in gen.iter_batches(
        driver_rf_ids=np.array(drivers, np.int32), chart_ids=np.array(charts, np.int8), batch_size=4)])


def test_level_domain_is_an_obligation_only_of_used_level_backed_columns() -> None:
    """D088 §4: a level-backed column's inputs must satisfy the forward chart's
    domain; a direct column has no level path and no level-domain obligation.
    Factor 51's history contains a non-positive level."""
    matrix = RiskFactorMatrix(
        dates=np.arange(4, dtype=np.int32), factor_ids=np.array([50, 51], np.int32),
        values=np.asfortranarray(np.array([[80.0, 5.0], [81.0, -1.0], [82.0, 2.0], [83.0, 3.0]])),
    )
    hist = build_scenario_generator(parse_scenario_model_config({"model": "historical", "horizon": 1}), data=matrix)
    assert _stream(hist, [50], [LOG]).shape == (3, 1)  # 51 is not used: no obligation
    assert _stream(hist, [51], [DIFF]).shape == (3, 1)  # DIFFERENCE: no positivity domain
    with pytest.raises(ValueError):
        _stream(hist, [51], [LOG])  # LOG needs positive levels
    direct = ParametricRiskData(
        factor_ids=np.array([50, 51], np.int32), cholesky_matrix=np.ascontiguousarray(np.eye(2)),
        volatilities=np.array([0.01, 0.02]), effective_period_count=1,
        effective_period_unit=np.int8(1), chart_ids=np.array([LOG, LOG], np.int8),
    )
    cfg = parse_scenario_model_config({"model": "parametric", "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
                                       "total_scenarios": 8, "seed": 3, "distribution": {"kind": "standard_normal"}})
    assert _stream(build_scenario_generator(cfg, data=direct), [51], [LOG]).shape == (8, 1)  # direct: no levels


# ---- V1-08 (b): D091 regime covariance, analytic ----------------------------------------------


def test_regime_switching_covariance_equals_the_d091_formula(monkeypatch) -> None:
    """Recover the generator's two linear maps Y = A_c z (calm c = 0, crisis
    c = 1) from its realised stream and its recorded draws (crisis iff u < p,
    D091), then evaluate
    Cov = (1 - p) A_0 A_0' + p A_1 A_1' analytically and compare with
    D_sigma [(1 - p) R + p D_m R D_m] D_sigma and Var_i = sigma_i^2 [(1 - p) + p m_i^2].
    No moment is estimated. Tolerance 1e-5 relative: the stream is stored as
    float32 (2^-24 relative per value) and the maps are recovered by a
    well-conditioned least-squares solve over 400 draws."""
    corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    chol = np.ascontiguousarray(np.linalg.cholesky(corr))
    sigma, m, p, seed, s_total = np.array([0.01, 0.02, 0.03]), np.array([2.0, 1.5, 3.0]), 0.3, 11, 400
    ids = np.array([60, 61, 62], np.int32)
    data = ParametricRiskData(
        factor_ids=ids, cholesky_matrix=chol, volatilities=sigma, effective_period_count=1,
        effective_period_unit=np.int8(1), chart_ids=np.array([LOG, LOG, LOG], np.int8),
        crisis_vol_multipliers=m,
    )
    cfg = parse_scenario_model_config({
        "model": "parametric", "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": s_total, "seed": seed,
        "distribution": {"kind": "regime_switching_gmm", "prob_crisis": p, "crisis_vol_multipliers": "per_factor"},
    })
    # Record the realised random inputs the generator draws (the normals and the
    # regime uniforms); the check below then uses no RNG convention at all.
    real_rng, drawn = np.random.default_rng, {"z": [], "u": []}

    class _Recording:
        def __init__(self, seed_value):
            self._rng = real_rng(seed_value)

        def standard_normal(self, *args, **kwargs):
            out = self._rng.standard_normal(*args, **kwargs)
            drawn["z"].append(np.array(out, copy=True))
            return out

        def random(self, *args, **kwargs):
            out = self._rng.random(*args, **kwargs)
            drawn["u"].append(np.array(out, copy=True))
            return out

        def __getattr__(self, name):
            return getattr(self._rng, name)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _Recording(seed))
    y = _stream(build_scenario_generator(cfg, data=data), ids.tolist(), [LOG, LOG, LOG]).astype(np.float64)
    assert len(drawn["z"]) == 1 and len(drawn["u"]) == 1  # one whole-run realisation (D087)
    z = drawn["z"][0].reshape(s_total, 3)
    crisis = drawn["u"][0].reshape(s_total) < p
    assert 0 < crisis.sum() < s_total
    a = [np.linalg.lstsq(z[mask], y[mask], rcond=None)[0].T for mask in (~crisis, crisis)]
    np.testing.assert_allclose(a[0], np.diag(sigma) @ chol, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(a[1], np.diag(m * sigma) @ chol, rtol=1e-5, atol=1e-9)
    cov_impl = (1 - p) * a[0] @ a[0].T + p * a[1] @ a[1].T
    r = chol @ chol.T
    cov_formula = np.diag(sigma) @ ((1 - p) * r + p * np.diag(m) @ r @ np.diag(m)) @ np.diag(sigma)
    np.testing.assert_allclose(cov_impl, cov_formula, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(np.diag(cov_impl), sigma**2 * ((1 - p) + p * m**2), rtol=1e-5)
