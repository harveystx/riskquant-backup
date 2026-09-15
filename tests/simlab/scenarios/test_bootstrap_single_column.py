# tests/simlab/scenarios/test_bootstrap_single_column.py
"""
R3 — F4 bootstrap single-column layout completeness.

A single-column (T, 1) float64 matrix is reported C-contiguous by NumPy, so
Numba resolves the fused SBB / FilteredBootstrap kernels to a C layout. The
kernels were `_F64_2D_F` input-only; this pinned the D024 twin defect. Both now
carry the single-column `_F64_2D_C` input overload.

These tests prove:
  - every dispatch shape (K_total, K_req, B, partial batch) runs, and
  - the single-column result equals the corresponding column of a multi-factor
    matrix to the bit (the column result must not depend on how many other
    columns share the matrix).
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.bootstrap_fhs import (
    FilteredBootstrapGenerator,
    _kernel_fused_filtered_bootstrap_h_day_shock,
)
from riskquant.simlab.scenarios.bootstrap_sbb import (
    StationaryBlockBootstrapGenerator,
    _kernel_fused_sbb_h_day_shock,
)

_TARGET = [100.0, 101.0, 103.0, 102.0, 106.0, 104.0, 108.0, 107.0, 110.0, 109.0]
_OTHER = [50.0, 51.0, 52.0, 50.5, 53.0, 52.0, 54.0, 55.0, 54.0, 56.0]


def _matrix(cols, fids) -> RiskFactorMatrix:
    t = len(cols[0])
    vals = np.asfortranarray(np.array(cols, dtype=np.float64).T)
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.array(fids, dtype=np.int32),
        values=vals,
    )


def _drive(gen, req_fids, batch_size) -> np.ndarray:
    n = len(req_fids)
    chunks = []
    for b in gen.iter_batches(
        required_shift_type_id=np.full(n, SHIFT_LOG_RETURN_ID, np.int8),
        source_factor_ids=np.array(req_fids, np.int32),
        shock_scale=np.ones(n, np.float32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


def _make(gen_cls, matrix, total_scenarios, horizon):
    kw = dict(matrix=matrix, total_scenarios=total_scenarios, horizon=horizon, seed=3)
    if gen_cls is StationaryBlockBootstrapGenerator:
        kw["expected_block_size"] = 2.0
    return gen_cls(**kw)


_GENS = [StationaryBlockBootstrapGenerator, FilteredBootstrapGenerator]

# (label, K_total, B, batch_size) — K_req is always 1 (single required factor)
_CASES = [
    ("Ktotal1_Breq1_Bgt1", 1, 6, 6),
    ("Ktotal2_Breq1_Bgt1", 2, 6, 6),
    ("Ktotal1_Breq1_Beq1", 1, 1, 1),
    ("Ktotal2_Breq1_Beq1", 2, 1, 1),
    ("partial_final_batch", 1, 7, 4),
]


@pytest.mark.parametrize("gen_cls", _GENS)
@pytest.mark.parametrize("label,ktotal,total,bs", _CASES)
def test_single_column_dispatch_runs(gen_cls, label, ktotal, total, bs) -> None:
    """Every dispatch shape runs without a Numba 'No matching definition'."""
    if ktotal == 1:
        m = _matrix([_TARGET], [10])
    else:
        m = _matrix([_TARGET, _OTHER], [10, 20])
    shocks = _drive(_make(gen_cls, m, total, 2), [10], bs)
    assert shocks.shape == (total, 1)
    assert shocks.dtype == np.float32
    assert np.all(np.isfinite(shocks))


@pytest.mark.parametrize("gen_cls", _GENS)
@pytest.mark.parametrize("horizon", [1, 3])
def test_single_column_equals_multi_factor_column(gen_cls, horizon) -> None:
    """The single-column (T,1) result equals the shared column of a (T,2)
    matrix to the bit: the column's output must not depend on sibling columns.
    Same seed => identical SBB/bootstrap path, so the only variable is layout.
    """
    m1 = _matrix([_TARGET], [10])
    m2 = _matrix([_TARGET, _OTHER], [10, 20])
    r1 = _drive(_make(gen_cls, m1, 8, horizon), [10], 8)[:, 0]
    r2 = _drive(_make(gen_cls, m2, 8, horizon), [10], 8)[:, 0]
    assert np.array_equal(r1, r2)  # exact — not just close


def test_single_column_c_overload_is_compiled() -> None:
    """Dispatch proof: after exercising a single-column matrix, each kernel has
    a C-contiguous float64 input signature (the D024 single-column overload)."""
    for gen_cls in _GENS:
        _drive(_make(gen_cls, _matrix([_TARGET], [10]), 4, 2), [10], 4)

    def _has_c_float64_input(dispatcher) -> bool:
        for sig in dispatcher.signatures:
            first = sig[0]  # first float64 2-D input array
            if (
                str(first.dtype) == "float64"
                and first.ndim == 2
                and first.layout == "C"
            ):
                return True
        return False

    assert _has_c_float64_input(_kernel_fused_sbb_h_day_shock)
    assert _has_c_float64_input(_kernel_fused_filtered_bootstrap_h_day_shock)
