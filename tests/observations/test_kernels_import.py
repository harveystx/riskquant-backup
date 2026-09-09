# tests/observations/test_kernels_import.py
"""Regression test for RQ-AUD-001: observations._kernels must import cleanly.

Previously imported non-existent layout-suffix-less jit aliases
(`_F64_2D_RO`, `_I32_2D_RO`), which do not exist in `riskquant.jit` — only the
layout-qualified forms (`_F64_2D_C_RO`, `_I32_2D_C_RO`) do. This raised
ImportError at module load, making the entire `observations` package (and
everything downstream, including the dependence/tail-risk estimators)
unusable.
"""
from __future__ import annotations


def test_kernels_module_imports():
    import riskquant.observations._kernels as m

    assert hasattr(m, "kernel_first_non_finite_flat_index")
    assert hasattr(m, "kernel_sum_of_squares")
    assert hasattr(m, "kernel_center_rows")
    assert hasattr(m, "kernel_tied_pair_counts")
    assert hasattr(m, "kernel_tie_rate")


def test_observations_compile_imports_downstream_of_kernels():
    import riskquant.observations.compile  # noqa: F401
