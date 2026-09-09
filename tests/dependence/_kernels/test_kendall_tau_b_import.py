# tests/dependence/_kernels/test_kendall_tau_b_import.py
"""Regression test for RQ-AUD-001: dependence._kernels._kendall_tau_b must
import cleanly.

Previously imported `_F64_2D_RO`, `_I32_1D_RO`, `_I32_2D_RO` — none of which
exist in `riskquant.jit` (only the layout-qualified `_F64_2D_C_RO`,
`_I32_1D_C_RO`, `_I32_2D_C_RO` do). This raised ImportError at module load.
"""
from __future__ import annotations


def test_kendall_tau_b_module_imports():
    import riskquant.dependence._kernels._kendall_tau_b as m

    assert hasattr(m, "kernel_kendall_tau_b_upper")


def test_dependence_estimators_kendall_imports_downstream():
    import riskquant.dependence.estimators.kendall  # noqa: F401
