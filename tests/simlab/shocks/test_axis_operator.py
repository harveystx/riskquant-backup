# tests/simlab/shocks/test_axis_operator.py
"""
D088 Slice 3, C1: numerical lowering of the compiled model onto the execution
axis, and the (d, tau) world request. Expected arrays are literal.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.simlab.shocks.axis_operator import lower_axis_operator, world_request
from riskquant.solutions.run_identity.hashing import hash_proxy_model

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1

# Axis: 10 LOG, 20 ABS, 30 NO_SHIFT, 40 SIMPLE, 50 DIFF.
AXIS = np.array([10, 20, 30, 40, 50], dtype=np.int32)
KAPPA = np.array([LOG, ABS, NO_SHIFT, SIMPLE, DIFF], dtype=np.int8)


def _model(targets=(10, 40, 50)) -> CompiledProxyModel:
    """Rows 10: (60,DIFF,0.04),(70,LOG,0.9); 40: (70,LOG,1.5); 50: (50,DIFF,1.0)."""
    rows = {
        10: [(60, DIFF, 0.04), (70, LOG, 0.9)],
        40: [(70, LOG, 1.5)],
        50: [(50, DIFF, 1.0)],
    }
    offsets, drivers, charts, loads = [0], [], [], []
    for t in targets:
        for d, c, b in rows[t]:
            drivers.append(d)
            charts.append(c)
            loads.append(b)
        offsets.append(len(drivers))
    return CompiledProxyModel(
        row_target_rf_ids=np.array(targets, dtype=np.int32),
        row_term_offsets=np.array(offsets, dtype=np.int32),
        term_driver_rf_ids=np.array(drivers, dtype=np.int32),
        term_chart_ids=np.array(charts, dtype=np.int8),
        term_loadings=np.array(loads, dtype=np.float64),
    )


def test_lowering_produces_the_literal_execution_operator() -> None:
    op = lower_axis_operator(model=_model(), axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)
    assert op.row_target_rf_ids.tolist() == [10, 20, 30, 40, 50]
    assert op.target_shift_type_ids.tolist() == [LOG, ABS, NO_SHIFT, SIMPLE, DIFF]
    assert op.row_term_offsets.tolist() == [0, 2, 3, 3, 4, 5]
    assert op.term_driver_rf_ids.tolist() == [60, 70, 20, 70, 50]
    assert op.term_chart_ids.tolist() == [DIFF, LOG, ABS, LOG, DIFF]
    assert op.term_loadings.tolist() == [0.04, 0.9, 1.0, 1.5, 1.0]
    for arr in (op.row_term_offsets, op.term_driver_rf_ids, op.term_loadings):
        assert not arr.flags.writeable and arr.flags.c_contiguous


def test_driver_chart_comes_from_the_term_not_the_target() -> None:
    """Row 10 has kappa = LOG; its first term keeps tau = DIFF."""
    op = lower_axis_operator(model=_model(), axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)
    assert op.target_shift_type_ids[0] == LOG
    assert op.term_chart_ids[0] == DIFF


def test_native_rows_are_lowered_outside_the_model() -> None:
    """ABS: one unit term on its own native column. NO_SHIFT: zero terms."""
    op = lower_axis_operator(model=_model(), axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)
    off = op.row_term_offsets
    assert (off[2] - off[1], off[3] - off[2]) == (1, 0)
    assert (op.term_driver_rf_ids[off[1]], op.term_chart_ids[off[1]]) == (20, ABS)
    assert op.term_loadings[off[1]] == 1.0


def test_lowering_leaves_the_model_and_its_identity_unchanged() -> None:
    model = _model()
    before = hash_proxy_model(model)
    snapshot = [
        a.tobytes()
        for a in (model.row_target_rf_ids, model.row_term_offsets,
                  model.term_driver_rf_ids, model.term_chart_ids, model.term_loadings)
    ]
    lower_axis_operator(model=model, axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)
    after = [
        a.tobytes()
        for a in (model.row_target_rf_ids, model.row_term_offsets,
                  model.term_driver_rf_ids, model.term_chart_ids, model.term_loadings)
    ]
    assert after == snapshot
    assert hash_proxy_model(model) == before


@pytest.mark.parametrize(
    "targets",
    [(40, 10, 50), (10, 40), (10, 50), (10, 40, 50, 20)],
    ids=["reordered", "missing_row", "missing_row_2", "extra_row"],
)
def test_model_rows_must_equal_the_v1_axis_subsequence_exactly(targets) -> None:
    rows = {10, 40, 50, 20}
    if 20 in targets:  # a row for the ABS node: a native node forced into the model
        model = CompiledProxyModel(
            row_target_rf_ids=np.array([10, 40, 50, 20], dtype=np.int32),
            row_term_offsets=np.array([0, 1, 2, 3, 4], dtype=np.int32),
            term_driver_rf_ids=np.array([70, 70, 50, 20], dtype=np.int32),
            term_chart_ids=np.array([LOG, LOG, DIFF, LOG], dtype=np.int8),
            term_loadings=np.array([1.0, 1.0, 1.0, 1.0]),
        )
    else:
        assert set(targets) <= rows
        model = _model(targets)
    with pytest.raises(ValueError, match="not bound to this execution axis"):
        lower_axis_operator(model=model, axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)


def test_a_no_shift_node_cannot_be_a_model_row() -> None:
    model = CompiledProxyModel(
        row_target_rf_ids=np.array([10, 30, 40, 50], dtype=np.int32),
        row_term_offsets=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        term_driver_rf_ids=np.array([70, 30, 70, 50], dtype=np.int32),
        term_chart_ids=np.array([LOG, LOG, LOG, DIFF], dtype=np.int8),
        term_loadings=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="not bound to this execution axis"):
        lower_axis_operator(model=model, axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)


@pytest.mark.parametrize(
    ("axis", "kappa", "exc"),
    [
        (np.array([10, 10], dtype=np.int32), np.array([LOG, LOG], dtype=np.int8), ValueError),
        (AXIS, np.array([LOG, ABS, 0, SIMPLE, DIFF], dtype=np.int8), ValueError),
        (AXIS, KAPPA[:4], ValueError),
        (AXIS.astype(np.int64), KAPPA, TypeError),
        (AXIS, KAPPA.astype(np.int32), TypeError),
    ],
    ids=["duplicate_node", "unsupported_kappa", "length", "axis_dtype", "kappa_dtype"],
)
def test_invalid_axes_are_rejected(axis, kappa, exc) -> None:
    with pytest.raises(exc):
        lower_axis_operator(model=_model(), axis_rf_ids=axis, axis_shift_type_ids=kappa)


def test_world_request_is_unique_pairs_in_first_seen_order() -> None:
    """Terms in order: (60,DIFF) (70,LOG) (20,ABS) (70,LOG) (50,DIFF).
    Sorted order would be (20,ABS) (50,DIFF) (60,DIFF) (70,LOG)."""
    op = lower_axis_operator(model=_model(), axis_rf_ids=AXIS, axis_shift_type_ids=KAPPA)
    d, tau = world_request(op)
    assert list(zip(d.tolist(), tau.tolist(), strict=True)) == [
        (60, DIFF), (70, LOG), (20, ABS), (50, DIFF)
    ]
    assert d.dtype == np.int32 and tau.dtype == np.int8


def test_one_driver_under_two_charts_is_two_requests() -> None:
    model = CompiledProxyModel(
        row_target_rf_ids=np.array([10], dtype=np.int32),
        row_term_offsets=np.array([0, 2], dtype=np.int32),
        term_driver_rf_ids=np.array([70, 70], dtype=np.int32),
        term_chart_ids=np.array([LOG, SIMPLE], dtype=np.int8),
        term_loadings=np.array([0.5, 0.25]),
    )
    op = lower_axis_operator(
        model=model,
        axis_rf_ids=np.array([10], dtype=np.int32),
        axis_shift_type_ids=np.array([LOG], dtype=np.int8),
    )
    d, tau = world_request(op)
    assert list(zip(d.tolist(), tau.tolist(), strict=True)) == [(70, LOG), (70, SIMPLE)]


def test_an_all_native_no_shift_axis_requests_nothing() -> None:
    empty = CompiledProxyModel(
        row_target_rf_ids=np.empty(0, dtype=np.int32),
        row_term_offsets=np.array([0], dtype=np.int32),
        term_driver_rf_ids=np.empty(0, dtype=np.int32),
        term_chart_ids=np.empty(0, dtype=np.int8),
        term_loadings=np.empty(0, dtype=np.float64),
    )
    op = lower_axis_operator(
        model=empty,
        axis_rf_ids=np.array([30], dtype=np.int32),
        axis_shift_type_ids=np.array([NO_SHIFT], dtype=np.int8),
    )
    assert op.row_term_offsets.tolist() == [0, 0]
    d, tau = world_request(op)
    assert d.size == 0 and tau.size == 0
