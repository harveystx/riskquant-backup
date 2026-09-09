# tests/config/static_config/test_ccy_id_bounds.py
"""Regression tests for ccy_id int8 bounds validation (RQ-AUD-014)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskquant.config.static_config.validate import _validate_ccy_id_bounds
from riskquant.exceptions import ConfigurationError


def test_exists_and_callable():
    assert callable(_validate_ccy_id_bounds)


@pytest.mark.parametrize("bad", [128, 200, 10_000, -1, -128])
def test_rejects_out_of_int8_range(bad):
    dfs = {"ccy": pd.DataFrame({"ccy_name": ["X"], "ccy_id": [bad]})}
    with pytest.raises(ConfigurationError, match="ccy_id"):
        _validate_ccy_id_bounds(dfs)


@pytest.mark.parametrize("good", [[0], [127], [0, 1, 2, 127]])
def test_accepts_valid_range(good):
    dfs = {
        "ccy": pd.DataFrame(
            {"ccy_name": [f"C{i}" for i in good], "ccy_id": good}
        )
    }
    _validate_ccy_id_bounds(dfs)  # must not raise


def test_error_names_offending_ids():
    dfs = {"ccy": pd.DataFrame({"ccy_name": ["A", "B"], "ccy_id": [5, 130]})}
    with pytest.raises(ConfigurationError) as exc:
        _validate_ccy_id_bounds(dfs)
    assert "130" in str(exc.value)
    assert "5" not in str(exc.value).split("[")[1].split("]")[0]


def test_np_int8_would_overflow_above_127():
    # Documents the failure mode the validator pre-empts.
    with pytest.raises(OverflowError):
        np.int8(128)
