# tests/market_data/risk_factors/test_unit_scale_guard.py
"""Regression test: the snapshot compiler fails loud on an un-occupied unit-scale
slot instead of silently propagating a sentinel/NaN (RQ-AUD-012)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import pytest

from riskquant.constants import INVALID_UNIT_SCALE
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)


@dataclass
class _StubStaticConfig:
    """Minimal duck-typed stand-in exposing only the two attributes the
    snapshot compiler reads."""

    rel_name_to_rf_id: dict[str, int]
    rf_id_to_unit_scale_factor: npt.NDArray[np.float64]


def _stub(n_slots: int, occupied: dict[int, float]) -> _StubStaticConfig:
    arr = np.full(n_slots, float(INVALID_UNIT_SCALE), dtype=np.float64)
    for rf_id, scale in occupied.items():
        arr[rf_id] = scale
    rel = {f"rf.{rf_id}": rf_id for rf_id in occupied}
    return _StubStaticConfig(rel_name_to_rf_id=rel, rf_id_to_unit_scale_factor=arr)


def test_occupied_slot_prices_correctly():
    cfg = _stub(4, {0: 1.0, 1: 0.01})
    snap = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=date(2026, 1, 1),
        levels_dict={"rf.0": 100.0, "rf.1": 250.0},
        static_config=cfg,
    )
    # 100 * 1.0 and 250 * 0.01
    levels = dict(zip(snap.factor_ids.tolist(), snap.factor_levels.tolist()))
    assert levels[0] == pytest.approx(100.0)
    assert levels[1] == pytest.approx(2.5)


def test_unoccupied_slot_raises_keyerror_not_nan():
    # rf_id 2 exists in the rel map but its unit-scale slot is the sentinel.
    cfg = _stub(4, {0: 1.0})
    cfg.rel_name_to_rf_id["rf.2"] = 2  # points at an un-occupied (sentinel) slot
    with pytest.raises(KeyError, match="un-occupied"):
        compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=date(2026, 1, 1),
            levels_dict={"rf.2": 100.0},
            static_config=cfg,
        )
