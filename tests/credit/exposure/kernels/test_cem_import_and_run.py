# tests/credit/exposure/kernels/test_cem_import_and_run.py
"""Regression test for RQ-AUD-005: the CEM kernel must import cleanly and
JIT-compile/execute with the canonical `riskquant.jit` array-type aliases
(previously redefined locally as `_F64_2D`, `_F64_1D`, `_I64_1D`)."""
from __future__ import annotations

import numpy as np


def test_cem_kernel_compiles_and_runs():
    from riskquant.credit.exposure.evaluators.kernels.current_exposure_method import (
        _kernel_current_exposure_method,
    )

    n_net_sets, m_total, s_scenarios = 2, 3, 4
    out = np.zeros((n_net_sets, s_scenarios), dtype=np.float64)
    mtm = np.random.default_rng(0).standard_normal((m_total, s_scenarios))
    mtm_row_perm = np.array([0, 1, 2], dtype=np.int64)
    ns_offsets = np.array([0, 2, 3], dtype=np.int64)
    collateral = np.array([0.0, 0.0], dtype=np.float64)
    gross_pfe = np.array([1.0, 2.0, 0.5], dtype=np.float64)

    _kernel_current_exposure_method(
        out, mtm, mtm_row_perm, ns_offsets, collateral, gross_pfe, 0.4, 0.6
    )

    assert np.all(np.isfinite(out))
    assert out.shape == (n_net_sets, s_scenarios)


def test_replacement_cost_and_saccr_kernels_import():
    import riskquant.credit.exposure.evaluators.kernels.replacement_cost  # noqa: F401
    import riskquant.credit.exposure.evaluators.kernels.saccr  # noqa: F401
