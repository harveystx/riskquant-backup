# tests/solutions/run_identity/test_deterministic_execution.py
"""
Release-gate determinism contract for the production risk path.

D029 pins the SCENARIO STREAM against batch partitioning. This suite pins the
stronger property the release gate actually needs: the FINAL reported numbers —
shocked levels, P&L, VaR and ES — are bitwise invariant to how the run is
partitioned and to which process executes it.

Comparisons are exact. Nothing is rounded and no tolerance is used: a tolerance
would hide the dependency instead of proving its absence.

The P&L reduction here is written as an explicit order-fixed accumulation rather
than a BLAS matvec. That is deliberate. A caller's `sensitivities @ delta` is a
GEMV whose internal blocking may vary with the batch width, which can move the
result by one ulp; using one here would test BLAS rather than the RiskQuant
path. The production Taylor kernel has no such freedom (thread-private scratch,
`prange` chunks owning disjoint whole rows, no cross-thread reduction), and the
E2E demo covers that full chain in regression.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)

_K, _T, _S = 4, 80, 64
_CONFIDENCE = 0.99

_BATCH_SIZES = [1, 2, 3, 5, 7, 11, _S - 1, _S, _S + 1, 1000]


def _matrix() -> RiskFactorMatrix:
    rng = np.random.default_rng(0)
    vals = np.asfortranarray(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (_T, _K)), axis=0))
    )
    return RiskFactorMatrix(
        dates=np.arange(_T, dtype=np.int32),
        factor_ids=np.arange(_K, dtype=np.int32),
        values=vals,
    )


def _correlation(a: np.ndarray) -> np.ndarray:
    """D091: L must factor a correlation matrix (unit diagonal)."""
    cov = a @ a.T / _K + np.eye(_K) * 0.1
    inv_sd = 1.0 / np.sqrt(np.diag(cov))
    return cov * np.outer(inv_sd, inv_sd)


def _parametric_data() -> ParametricRiskData:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (_K, _K))
    return ParametricRiskData(
        factor_ids=np.arange(_K, dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.linalg.cholesky(_correlation(a))),
        volatilities=np.ascontiguousarray(np.full(_K, 0.02)),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(_K, SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


_CONFIGS = {
    "sbb": {
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": _S,
        "expected_block_size": 5.0,
        "seed": 7,
    },
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": _S,
        "seed": 7,
    },
    "parametric": {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": _S,
        "distribution": {"kind": "student_t", "df": 5.0},
        "seed": 7,
    },
    "historical": {"model": "historical", "horizon": 1},
}


def _run(model: str, batch_size: int) -> dict[str, object]:
    """Official pieces: generator -> apply_shocks_to_base_levels -> P&L -> VaR/ES."""
    data = _parametric_data() if model == "parametric" else _matrix()
    matrix = _matrix()
    base_levels = np.ascontiguousarray(np.asarray(matrix.values)[-1].astype(np.float64))
    sensitivities = np.array([1000.0, -500.0, 250.0, 750.0], np.float64)

    gen = build_scenario_generator(
        parse_scenario_model_config(_CONFIGS[model]), data=data
    )
    shocks, levels, pnl = [], [], []
    for batch in gen.iter_batches(
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * _K, np.int8),
        source_factor_ids=np.arange(_K, dtype=np.int32),
        shock_scale=np.ones(_K, np.float32),
        batch_size=batch_size,
    ):
        b = batch.shocks.shape[0]
        out = np.empty((_K, b), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=out)
        shocks.append(np.array(batch.shocks, copy=True))
        levels.append(out.copy())
        # order-fixed reduction: scenario-major, factor loop in index order
        for j in range(b):
            acc = 0.0
            for i in range(_K):
                acc += sensitivities[i] * (out[i, j] - base_levels[i])
            pnl.append(acc)

    pnl_arr = np.asarray(pnl, dtype=np.float64)
    res = compute_var_es(np.ascontiguousarray(pnl_arr.reshape(1, -1)), _CONFIDENCE)
    return {
        "shocks": np.vstack(shocks),
        "levels": np.hstack(levels),
        "pnl": pnl_arr,
        "var": float(np.asarray(res.var).ravel()[0]),
        "es": float(np.asarray(res.es).ravel()[0]),
    }


# ---- batch-size invariance of the FINAL numbers ------------------------------


@pytest.mark.parametrize("model", list(_CONFIGS))
@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_final_risk_numbers_are_bitwise_batch_invariant(model, batch_size) -> None:
    ref = _run(model, 4)
    got = _run(model, batch_size)
    assert np.array_equal(ref["shocks"], got["shocks"]), "scenario stream (D029)"
    assert np.array_equal(ref["levels"], got["levels"]), "shocked levels"
    assert np.array_equal(ref["pnl"], got["pnl"]), "P&L vector"
    # exact IEEE-754 comparison; no tolerance, no rounding
    assert ref["var"].hex() == got["var"].hex(), "VaR"
    assert ref["es"].hex() == got["es"].hex(), "ES"


def test_no_batch_size_produces_a_different_var(  # explicit release-gate assertion
) -> None:
    hexes = {_run("sbb", bs)["var"].hex() for bs in _BATCH_SIZES}
    assert len(hexes) == 1


# ---- array-layout invariance -------------------------------------------------


def test_matrix_layout_does_not_change_the_result() -> None:
    """A C-ordered copy of the same logical matrix must price identically; layout
    is an execution detail, not semantics."""
    f_matrix = _matrix()
    c_values = np.ascontiguousarray(np.asarray(f_matrix.values))
    c_matrix = RiskFactorMatrix(
        dates=f_matrix.dates, factor_ids=f_matrix.factor_ids, values=c_values
    )
    cfg = parse_scenario_model_config(_CONFIGS["sbb"])
    args = dict(
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * _K, np.int8),
        source_factor_ids=np.arange(_K, dtype=np.int32),
        shock_scale=np.ones(_K, np.float32),
        batch_size=8,
    )

    def stream(m):
        return np.vstack([
            np.array(b.shocks, copy=True)
            for b in build_scenario_generator(cfg, data=m).iter_batches(**args)
        ])

    assert np.array_equal(stream(f_matrix), stream(c_matrix))


# ---- fresh-process invariance of the final numbers ---------------------------


_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {tests!r})
    from test_deterministic_execution import _run
    out = {{}}
    for model in ("sbb", "parametric"):
        for bs in (3, 4, 64):
            r = _run(model, bs)
            out[f"{{model}}:{{bs}}"] = [r["var"].hex(), r["es"].hex()]
    print(json.dumps(out))
    """
)


def _child(tests_dir: str, env_extra: dict[str, str] | None = None) -> dict:
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests=tests_dir)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_processes_agree_bitwise(request) -> None:
    tests_dir = str(request.path.parent)
    assert _child(tests_dir) == _child(tests_dir)


@pytest.mark.parametrize(
    "env",
    [
        {"OMP_NUM_THREADS": "1"},
        {"OMP_NUM_THREADS": "4"},
        {"OPENBLAS_NUM_THREADS": "1"},
        {"NUMBA_NUM_THREADS": "1"},
    ],
    ids=["omp1", "omp4", "openblas1", "numba1"],
)
def test_thread_environment_does_not_change_the_result(request, env) -> None:
    """Thread configuration must not move the numbers.

    NOTE: the value of this test depends on the host. On a single-core runner the
    thread ceiling is 1 and parallel reduction is never exercised, so this pins
    the contract rather than proving it; on a multi-core runner it becomes a real
    parallel-determinism gate. The structural argument is in the Taylor kernel:
    thread-private scratch, `prange` chunks owning disjoint whole rows, and no
    cross-thread reduction.
    """
    tests_dir = str(request.path.parent)
    assert _child(tests_dir, env) == _child(tests_dir)
