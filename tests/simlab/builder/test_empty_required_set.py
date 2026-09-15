# tests/simlab/builder/test_empty_required_set.py
"""
R6 / F11 — empty required set is a legitimate no-shock operation.

`build_shock_universe` deliberately admits a zero-length required set (explicit
`size == 0` early return in `_smoke_test_inputs`): a portfolio with no
simulatable risk factors yields an empty shock universe and `(B, 0)` shocks,
which the apply path processes as a no-op. This is intended behaviour, not a
missing guard — it is pinned here so it is not "hardened" away by mistake.

(Duplicate-required-id dedup and first-seen mapping are covered in
`test_shock_universe_dedup.py`.)
"""

from __future__ import annotations

import numpy as np

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels


def test_empty_required_set_builds_empty_universe() -> None:
    u = build_shock_universe(
        required_shift_type_id=np.array([], dtype=np.int8),
        source_factor_ids=np.array([], dtype=np.int32),
        available_factor_ids=np.array([10, 20], dtype=np.int32),
    )
    assert u.shock_source_ids.shape == (0,)
    assert u.shock_shift_type_ids.shape == (0,)
    assert u.required_to_shock_uix.shape == (0,)


def test_empty_required_set_applies_as_no_op() -> None:
    """End-to-end: an empty required set yields a (0, B) apply that runs
    cleanly and writes nothing (no exposures to shock)."""
    b_scn = 8
    batch = ScenarioSetBatch(
        shocks=np.zeros((b_scn, 0), dtype=np.float32, order="F"),
        required_to_shock_uix=np.zeros(0, dtype=np.int32),
        shock_scale=np.zeros(0, dtype=np.float32),
        required_shift_type_id=np.zeros(0, dtype=np.int8),
        scenario_offset=0,
        scenario_count_total=b_scn,
    )
    out = np.empty((0, b_scn), dtype=np.float64, order="C")
    apply_shocks_to_base_levels(
        base_levels=np.zeros(0, dtype=np.float64), batch=batch, out=out
    )
    assert out.shape == (0, b_scn)


def test_non_empty_still_works_control() -> None:
    """Control: a normal single-factor required set is unaffected."""
    u = build_shock_universe(
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8),
        source_factor_ids=np.array([10], dtype=np.int32),
        available_factor_ids=np.array([10, 20], dtype=np.int32),
    )
    assert u.shock_source_ids.shape == (1,)
    assert u.required_to_shock_uix.tolist() == [0]
