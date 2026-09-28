# tests/simlab/shocks/test_column_binding.py
"""World-column binding (d, tau) x h -> column index (D088 §4, §12.5, §13.4)."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.simlab.shocks.column_binding import bind_term_columns

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1


def _bind(term_d, term_tau, world_d, world_tau, requested=1, world_h=1):
    return bind_term_columns(
        term_driver_rf_ids=np.array(term_d, dtype=np.int32),
        term_chart_ids=np.array(term_tau, dtype=np.int8),
        requested_horizon=requested,
        world_column_driver_rf_ids=np.array(world_d, dtype=np.int32),
        world_column_chart_ids=np.array(world_tau, dtype=np.int8),
        world_horizon=world_h,
    )


def test_terms_resolve_to_their_own_driver_and_chart_columns():
    cols = _bind(
        [9000, 9000, 11000, 6],
        [LOG, DIFF, LOG, ABS],
        [11000, 9000, 6, 9000],
        [LOG, DIFF, ABS, LOG],
    )
    assert cols.tolist() == [3, 1, 0, 2]
    assert cols.dtype == np.int32 and cols.flags.c_contiguous and cols.flags.writeable


def test_missing_column_is_rejected_without_substitution():
    # The world has (9000, LOG, 1) but the term asks for (9000, DIFF, 1).
    with pytest.raises(ValueError, match=r"\(9000, 3, 1\)"):
        _bind([9000], [DIFF], [9000], [LOG])
    with pytest.raises(ValueError, match="does not supply"):
        _bind([9000], [LOG], [], [])


def test_horizon_mismatch_is_rejected():
    with pytest.raises(ValueError, match="horizon mismatch"):
        _bind([9000], [LOG], [9000], [LOG], requested=5, world_h=1)


@pytest.mark.parametrize("horizon", [True, 1.0, np.int64(1)])
def test_horizons_must_be_plain_ints(horizon):
    with pytest.raises(TypeError, match="int"):
        _bind([9000], [LOG], [9000], [LOG], requested=horizon)


def test_nonpositive_horizon_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        _bind([9000], [LOG], [9000], [LOG], requested=0, world_h=0)


def test_duplicate_world_keys_are_ambiguous():
    with pytest.raises(ValueError, match="duplicate"):
        _bind([9000], [LOG], [9000, 9000], [LOG, LOG])


@pytest.mark.parametrize("chart", [NO_SHIFT, 0, 7])
def test_unbindable_term_charts_are_rejected(chart):
    with pytest.raises(ValueError, match="term_chart_ids"):
        _bind([9000], [chart], [9000], [LOG])


def test_invalid_world_charts_and_ids_are_rejected():
    with pytest.raises(ValueError, match="world_column_chart_ids"):
        _bind([9000], [LOG], [9000], [9])
    with pytest.raises(ValueError, match="non-negative"):
        _bind([-1], [LOG], [9000], [LOG])


def test_dtype_and_shape_violations_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        bind_term_columns(
            term_driver_rf_ids=np.array([9000], dtype=np.int64),
            term_chart_ids=np.array([LOG], dtype=np.int8),
            requested_horizon=1,
            world_column_driver_rf_ids=np.array([9000], dtype=np.int32),
            world_column_chart_ids=np.array([LOG], dtype=np.int8),
            world_horizon=1,
        )
    with pytest.raises(ValueError, match="one length"):
        _bind([9000, 1], [LOG], [9000], [LOG])


def test_empty_term_set_binds_to_empty_indices():
    cols = _bind([], [], [9000], [LOG])
    assert cols.shape == (0,) and cols.dtype == np.int32


def test_read_only_model_arrays_are_accepted_and_not_mutated():
    d = np.array([9000, 11000], dtype=np.int32)
    tau = np.array([DIFF, LOG], dtype=np.int8)
    for a in (d, tau):
        a.setflags(write=False)
    cols = bind_term_columns(
        term_driver_rf_ids=d,
        term_chart_ids=tau,
        requested_horizon=3,
        world_column_driver_rf_ids=np.array([11000, 9000], dtype=np.int32),
        world_column_chart_ids=np.array([LOG, DIFF], dtype=np.int8),
        world_horizon=3,
    )
    assert cols.tolist() == [1, 0]
    assert d.tolist() == [9000, 11000] and tau.tolist() == [DIFF, LOG]
