# tests/solutions/scenario_models/test_world_column_seam.py
"""
D088 Slice 3: compiled model -> execution-axis lowering -> (d, tau) world
request -> world columns at h -> bind_term_columns -> Slice 2 -> T^-1_kappa.

Oracles are D088 §11 closed forms. A float64 Python reference uses the D088
§13.6 tolerance (rtol 1e-12, atol 1e-15) and takes as input the stored
float32 world value, which is itself checked against the chart definition;
determinism checks are bitwise.
"""

from __future__ import annotations

import ast
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.run_identity.hashing import hash_proxy_model
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1
RTOL, ATOL = 1e-12, 1e-15

# Driver nodes of the level-backed matrix world (h = 1, one scenario).
B, S, P, S2, N, C, AB = 1, 2, 3, 4, 5, 6, 7
LEVELS = {  # start -> end
    B: (80.0, 82.0),
    S: (2.00, 2.50),
    P: (80.0, 88.0),
    S2: (0.0300, 0.0325),
    N: (80.0, 82.0),
    C: (100.0, 110.0),
    AB: (-10.0, -9.0),
}


# ---- helpers ----------------------------------------------------------------------------


def _model(rows: list[tuple[int, list[tuple[int, int, float]]]]) -> CompiledProxyModel:
    offsets, d, c, b = [0], [], [], []
    targets = []
    for target, terms in rows:
        targets.append(target)
        for drv, chart, beta in terms:
            d.append(drv)
            c.append(chart)
            b.append(beta)
        offsets.append(len(d))
    return CompiledProxyModel(
        row_target_rf_ids=np.array(targets, dtype=np.int32),
        row_term_offsets=np.array(offsets, dtype=np.int32),
        term_driver_rf_ids=np.array(d, dtype=np.int32),
        term_chart_ids=np.array(c, dtype=np.int8),
        term_loadings=np.array(b, dtype=np.float64),
    )


def _matrix(ids: list[int], paths: dict[int, list[float]] | None = None) -> RiskFactorMatrix:
    paths = paths or {i: list(LEVELS[i]) for i in ids}
    t = len(next(iter(paths.values())))
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.array(sorted(ids), dtype=np.int32),
        values=np.asfortranarray(np.array([paths[i] for i in sorted(ids)]).T),
    )


def _hist(h: int = 1) -> Any:
    return parse_scenario_model_config({"model": "historical", "horizon": h})


def _param_cfg(count: int = 1, **extra: Any) -> Any:
    return parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": count, "unit": "BUSINESS_DAY"},
        "total_scenarios": extra.pop("total_scenarios", 40),
        "seed": 5,
        "distribution": {"kind": "standard_normal"},
    })


def _param_data(ids: list[int], charts: list[int], count: int = 1) -> ParametricRiskData:
    k = len(ids)
    return ParametricRiskData(
        factor_ids=np.array(ids, dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.eye(k)),
        volatilities=np.full(k, 0.02),
        effective_period_count=count,
        effective_period_unit=np.int8(1),
        chart_ids=np.array(charts, dtype=np.int8),
    )


def _run(model, axis, kappa, base, config, data, batch_size=64):
    """The Slice 3 composition, exactly as the Stage D run will call it."""
    op = lower_axis_operator(
        model=model,
        axis_rf_ids=np.array(axis, dtype=np.int32),
        axis_shift_type_ids=np.array(kappa, dtype=np.int8),
    )
    wb = bind_operator_to_world(op, config, data)
    gen = build_scenario_generator(config, data=data)
    outs, cols = [], []
    for wc in gen.iter_batches(driver_rf_ids=wb.request_driver_rf_ids,
        chart_ids=wb.request_chart_ids,
        batch_size=batch_size,
    ):
        shocks_before = np.array(wc.shocks, copy=True)
        out = np.empty((len(axis), wc.shocks.shape[0]), dtype=np.float64)
        apply_linear_terms_to_base_levels(
            base_levels=np.array(base, dtype=np.float64),
            row_term_offsets=op.row_term_offsets,
            term_column_uix=wb.term_column_uix,
            term_loadings=op.term_loadings,
            target_shift_types=op.target_shift_type_ids,
            shocks=wc.shocks,
            out=out,
        )
        assert wc.shocks.tobytes() == shocks_before.tobytes()  # the model changes no column
        outs.append(out)
        cols.append(shocks_before)
    return np.hstack(outs), np.vstack(cols), wb, op


def _col(wb, cols, d, tau) -> np.ndarray:
    k = [
        i
        for i, (dd, tt) in enumerate(
            zip(wb.request_driver_rf_ids.tolist(), wb.request_chart_ids.tolist(), strict=True)
        )
        if (dd, tt) == (d, tau)
    ]
    assert len(k) == 1
    return cols[:, k[0]]


# ---- D088 §11 closed-form oracles through the whole seam ---------------------------------


def test_closed_form_oracles_through_the_seam() -> None:
    # Targets 100..105 (not world factors): each row is one §11 oracle.
    model = _model([
        (100, [(S, DIFF, 0.04)]),               # LOG <- DIFFERENCE
        (101, [(B, LOG, 0.95)]),                # LOG <- LOG
        (102, [(P, SIMPLE, 0.5)]),              # SIMPLE reconstruction
        (103, [(S2, DIFF, 1.0)]),               # DIFFERENCE <- DIFFERENCE
        (104, [(P, LOG, 0.002)]),               # DIFFERENCE <- LOG
        (105, [(B, LOG, 0.9), (S, DIFF, 0.02)]),  # mixed charts
    ])
    axis = [100, 101, 102, 103, 104, 105]
    kappa = [LOG, LOG, SIMPLE, DIFF, DIFF, LOG]
    base = [70.0, 70.0, 50.0, 0.045, 0.025, 70.0]
    out, cols, wb, _ = _run(model, axis, kappa, base, _hist(), _matrix([B, S, P, S2]))

    y_sd = _col(wb, cols, S, DIFF)[0]
    y_bl = _col(wb, cols, B, LOG)[0]
    y_ps = _col(wb, cols, P, SIMPLE)[0]
    y_s2 = _col(wb, cols, S2, DIFF)[0]
    y_pl = _col(wb, cols, P, LOG)[0]
    # The world columns are the chart definitions of the level paths.
    assert y_sd == np.float32(2.50 - 2.00)
    assert y_bl == np.float32(math.log(82.0 / 80.0))
    assert y_ps == np.float32(88.0 / 80.0 - 1.0)
    assert y_s2 == np.float32(0.0325 - 0.0300)
    assert y_pl == np.float32(math.log(88.0 / 80.0))

    ref = [
        70.0 * math.exp(0.04 * float(y_sd)),
        70.0 * math.exp(0.95 * float(y_bl)),
        50.0 * (1.0 + 0.5 * float(y_ps)),
        0.045 + 1.0 * float(y_s2),
        0.025 + 0.002 * float(y_pl),
        70.0 * math.exp(0.9 * float(y_bl) + 0.02 * float(y_sd)),
    ]
    np.testing.assert_allclose(out[:, 0], ref, rtol=RTOL, atol=ATOL)
    # §11 literals (float32 world storage).
    for got, lit, tol in zip(
        out[:, 0], [71.4141, 71.6615, 52.5, 0.0475, 0.0251906, 72.292],
        [1e-4, 1e-4, 1e-5, 1e-9, 1e-7, 1e-3], strict=True,
    ):
        assert abs(got - lit) < tol


def test_log_target_from_difference_driver_is_not_a_log_interpretation() -> None:
    model = _model([(100, [(S, DIFF, 0.04)])])
    out, *_ = _run(model, [100], [LOG], [70.0], _hist(), _matrix([S]))
    assert abs(out[0, 0] - 70.0 * math.exp(0.02)) < 1e-9
    assert abs(out[0, 0] - 70.0 * math.exp(0.04 * math.log(2.50 / 2.00))) > 0.1


# ---- charts and keys -------------------------------------------------------------------------


def test_same_node_under_two_charts_binds_two_distinct_columns() -> None:
    """§11: row B under DIFFERENCE gives B_0 + 2; row A via (B, LOG) gives
    71.6615. One-factor matrix world (C0 layout)."""
    model = _model([(100, [(B, LOG, 0.95)]), (B, [(B, DIFF, 1.0)])])
    out, cols, wb, _ = _run(model, [100, B], [LOG, DIFF], [70.0, 80.0], _hist(), _matrix([B]))
    assert list(zip(wb.request_driver_rf_ids.tolist(), wb.request_chart_ids.tolist(), strict=True)) == [
        (B, LOG), (B, DIFF)
    ]
    assert wb.term_column_uix.tolist() == [0, 1]
    assert abs(out[1, 0] - 82.0) < 1e-12
    assert abs(out[0, 0] - 71.6615) < 1e-4


def test_parametric_same_factor_other_chart_is_a_missing_column_before_generation(
    monkeypatch,
) -> None:
    """The parametric generator is chart-blind by D091 design: asked directly, it serves (B, DIFF) from the
    (B, LOG) column. The Slice 3 seam must refuse it before generation."""
    data = _param_data([B], [LOG])
    cfg = _param_cfg()
    # Asked directly, the chart-blind generator returns one column twice.
    direct = build_scenario_generator(cfg, data=data)
    batch = next(direct.iter_batches(
        chart_ids=np.array([LOG, DIFF], dtype=np.int8),
        driver_rf_ids=np.array([B, B], dtype=np.int32),
        batch_size=8,
    ))
    s = np.asarray(batch.shocks)
    assert s[:, 0].tobytes() == s[:, 1].tobytes()  # DIFF request got the LOG numbers

    def _no_generation(*_a: Any, **_k: Any) -> None:
        raise AssertionError("world generation must not start")

    monkeypatch.setattr(ParametricGenerator, "iter_batches", _no_generation)
    op = lower_axis_operator(
        model=_model([(100, [(B, DIFF, 1.0)])]),
        axis_rf_ids=np.array([100], dtype=np.int32),
        axis_shift_type_ids=np.array([LOG], dtype=np.int8),
    )
    with pytest.raises(ValueError, match=r"does not supply: \[\(1, 3, 1\)\]"):
        bind_operator_to_world(op, cfg, data)


def test_parametric_world_binds_declared_charts_and_x_is_beta_times_y() -> None:
    """(B, LOG, 1) declared, sigma = 0.02; A <- 0.95 (B, LOG): per scenario
    ln(A_s / A_0) = 0.95 * Y_B (sigma_A = 0.019)."""
    data = _param_data([B, S], [LOG, DIFF])
    model = _model([(100, [(B, LOG, 0.95)]), (101, [(S, DIFF, 1.0)])])
    out, cols, wb, _ = _run(model, [100, 101], [LOG, DIFF], [70.0, 2.0], _param_cfg(), data)
    y_b, y_s = _col(wb, cols, B, LOG), _col(wb, cols, S, DIFF)
    np.testing.assert_allclose(np.log(out[0] / 70.0), 0.95 * y_b.astype(np.float64), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(out[1], 2.0 + y_s.astype(np.float64), rtol=RTOL, atol=ATOL)


def test_matrix_world_refuses_a_driver_it_does_not_hold() -> None:
    op = lower_axis_operator(
        model=_model([(100, [(C, LOG, 1.0)])]),
        axis_rf_ids=np.array([100], dtype=np.int32),
        axis_shift_type_ids=np.array([LOG], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="not in the matrix world"):
        bind_operator_to_world(op, _hist(), _matrix([B]))


# ---- horizon -----------------------------------------------------------------------------------


def test_the_world_owns_h() -> None:
    op = lower_axis_operator(
        model=_model([(100, [(B, LOG, 1.0)])]),
        axis_rf_ids=np.array([100], dtype=np.int32),
        axis_shift_type_ids=np.array([LOG], dtype=np.int8),
    )
    paths = {B: [80.0, 81.0, 82.0, 83.0, 83.5, 84.0, 85.0]}
    assert bind_operator_to_world(op, _hist(5), _matrix([B], paths)).horizon == 5
    assert bind_operator_to_world(op, _param_cfg(5), _param_data([B], [LOG], 5)).horizon == 5


def test_requested_and_world_periods_must_match_exactly() -> None:
    op = lower_axis_operator(
        model=_model([(100, [(B, LOG, 1.0)])]),
        axis_rf_ids=np.array([100], dtype=np.int32),
        axis_shift_type_ids=np.array([LOG], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="D090"):
        bind_operator_to_world(op, _param_cfg(1), _param_data([B], [LOG], 5))


def test_beta_is_horizon_invariant() -> None:
    """§11 three-world test, level-backed: h = 1 (80 -> 84) and h = 5 (a path
    ending at 84) give Y = ln(84/80) and 70 e^(0.95 Y) = 73.321; no sqrt(h)."""
    model = _model([(100, [(B, LOG, 0.95)])])
    one, *_ = _run(model, [100], [LOG], [70.0], _hist(1), _matrix([B], {B: [80.0, 84.0]}))
    five, *_ = _run(
        model, [100], [LOG], [70.0], _hist(5),
        _matrix([B], {B: [80.0, 81.0, 82.0, 83.0, 83.5, 84.0]}),
    )
    assert one.tobytes() == five.tobytes()
    assert abs(one[0, 0] - 73.321) < 1e-3
    y = float(np.float32(math.log(84.0 / 80.0)))
    assert abs(one[0, 0] - 70.0 * math.exp(0.95 * math.sqrt(5.0) * y)) > 1.0


# ---- native rows ---------------------------------------------------------------------------------


def test_no_shift_row_needs_no_column_and_its_data_column_still_drives_others() -> None:
    """§11: NO_SHIFT node N held at its base; a separate (N, LOG) column drives
    A: ln(82/80), A_s = 71.6615."""
    model = _model([(100, [(N, LOG, 0.95)])])
    out, cols, wb, op = _run(model, [N, 100], [NO_SHIFT, LOG], [80.0, 70.0], _hist(), _matrix([N]))
    assert (N, NO_SHIFT) not in list(zip(wb.request_driver_rf_ids.tolist(), wb.request_chart_ids.tolist(), strict=True))
    assert op.row_term_offsets[1] - op.row_term_offsets[0] == 0
    assert out[0, 0] == 80.0
    assert abs(out[1, 0] - 71.6615) < 1e-4


def test_absolute_base_row_uses_its_native_column_in_a_matrix_world() -> None:
    """ABS row: -10 -> -9, Y_ABS = 0.1, A_s = -10 + 0.1 * |-10| = -9."""
    empty = _model([])
    out, cols, wb, _ = _run(empty, [AB], [ABS], [-10.0], _hist(), _matrix([AB]))
    y = _col(wb, cols, AB, ABS)[0]
    assert y == np.float32(0.1)
    np.testing.assert_allclose(out[0, 0], -10.0 + float(y) * 10.0, rtol=RTOL, atol=ATOL)


def test_absolute_base_row_on_a_parametric_world_is_a_missing_column() -> None:
    op = lower_axis_operator(
        model=_model([]),
        axis_rf_ids=np.array([AB], dtype=np.int32),
        axis_shift_type_ids=np.array([ABS], dtype=np.int8),
    )
    with pytest.raises(ValueError, match=r"does not supply: \[\(7, 4, 1\)\]"):
        bind_operator_to_world(op, _param_cfg(), _param_data([AB], [DIFF]))


# ---- no chains, no mutation, no identity change ------------------------------------------------


def test_a_term_reads_a_world_column_never_another_targets_shock() -> None:
    """A <- (B, LOG, 1); B <- (C, LOG, 2). A_s = A_0 e^(Y_B), not e^(X_B)."""
    model = _model([(B, [(C, LOG, 2.0)]), (100, [(B, LOG, 1.0)])])
    out, cols, wb, _ = _run(model, [B, 100], [LOG, LOG], [80.0, 70.0], _hist(), _matrix([B, C]))
    y_b, y_c = float(_col(wb, cols, B, LOG)[0]), float(_col(wb, cols, C, LOG)[0])
    np.testing.assert_allclose(out[:, 0], [80.0 * math.exp(2.0 * y_c), 70.0 * math.exp(y_b)], rtol=RTOL, atol=ATOL)
    assert abs(out[1, 0] - 70.0 * math.exp(2.0 * y_c)) > 1.0


def test_binding_changes_neither_the_model_nor_its_identity() -> None:
    model = _model([(100, [(B, LOG, 0.9), (S, DIFF, 0.04)])])
    before = hash_proxy_model(model)
    _run(model, [100], [LOG], [70.0], _hist(), _matrix([B, S]))
    assert hash_proxy_model(model) == before


@pytest.mark.parametrize("batch_size", [1, 7, 40, 1000])
def test_the_seam_is_batch_invariant(batch_size: int) -> None:
    data = _param_data([B, S], [LOG, DIFF])
    model = _model([(100, [(B, LOG, 0.95), (S, DIFF, 0.3)])])
    ref, *_ = _run(model, [100], [LOG], [70.0], _param_cfg(), data, batch_size=13)
    got, *_ = _run(model, [100], [LOG], [70.0], _param_cfg(), data, batch_size=batch_size)
    assert got.tobytes() == ref.tobytes()


# ---- the D013 boundary: absent ---------------------------------------------------------------------------

_SEAM = [
    "src/riskquant/simlab/shocks/axis_operator.py",
    "src/riskquant/solutions/scenario_models/world_columns.py",
]


def _repo() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("path", _SEAM)
def test_seam_modules_import_nothing_from_the_d013_mechanism(path: str) -> None:
    tree = ast.parse((_repo() / path).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    forbidden = {
        "riskquant.proxy_resolver.proxy",
        "riskquant.proxy_resolver.proxy_audit",
        "riskquant.simlab.shocks.apply",
        "riskquant.simlab.shocks.shock_kernel",
        "riskquant.simlab.builder.shock_universe",
    }
    assert not (imported & forbidden)
