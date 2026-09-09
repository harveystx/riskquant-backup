# tests/valuation/pricer_kernels/test_spot_kernel_run.py
"""Regression test for RQ-AUD-005: the spot pricer kernel must import cleanly
and JIT-compile/execute with the canonical `riskquant.jit` array-type aliases
(previously redefined locally as `_F64_2D`, `_F64_1D`, `_I32_1D`)."""
from __future__ import annotations

import numpy as np


def test_spot_kernel_compiles_and_runs():
    from riskquant.valuation.pricer_kernels.spot import _kernel_price_spot

    n_trades, n_prices, s = 3, 2, 4
    out = np.zeros((n_trades, s), dtype=np.float64)
    prices = np.array(
        [[100.0, 101.0, 99.0, 102.0], [50.0, 51.0, 49.0, 52.0]], dtype=np.float64
    )
    qty = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    uix = np.array([0, 1, 0], dtype=np.int32)

    _kernel_price_spot(out, prices, qty, uix)

    np.testing.assert_allclose(out[0], prices[0])
    np.testing.assert_allclose(out[1], -2.0 * prices[1])
    np.testing.assert_allclose(out[2], 0.5 * prices[0])
