# tests/config/static_config/test_unit_scale_sentinel.py
"""Regression tests: rf_id_to_unit_scale_factor uses INVALID_UNIT_SCALE, not NaN
(RQ-AUD-012)."""
from __future__ import annotations

import inspect

import numpy as np

import riskquant.config.static_config.static_config as sc
from riskquant.constants import INVALID_UNIT_SCALE


def test_sentinel_is_finite_negative():
    assert np.isfinite(INVALID_UNIT_SCALE)
    assert float(INVALID_UNIT_SCALE) < 0.0


def test_static_config_source_uses_sentinel_not_nan():
    src = inspect.getsource(sc)
    assert "np.full(n, np.nan" not in src, "unit_scale must not init with np.nan"
    assert "INVALID_UNIT_SCALE" in src, "unit_scale must init with the sentinel"
