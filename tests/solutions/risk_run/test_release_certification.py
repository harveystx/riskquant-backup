# tests/solutions/risk_run/test_release_certification.py
"""
Release-gate certification of the official scenario-VaR path under GENUINE
multi-thread execution.

Rule of evidence: requesting threads is not executing them. Every multi-thread
claim here is backed by `numba.get_thread_id()` values observed INSIDE the
parallel region, so a host that silently clamps the request cannot satisfy the
test. Child processes are used because `NUMBA_NUM_THREADS` fixes the thread
ceiling at import time.

Comparisons are exact: sha256 over dtype|shape|bytes for arrays, `float.hex()`
for scalars. No tolerance is used anywhere a bitwise claim is made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

#: Thread counts to certify. Numba's OMP layer spawns real worker threads even
#: when the CPU count is lower, so the partitioning path is genuinely exercised.
_THREADS = [1, 2, 4, 8]
_BATCHES = [1, 3, 7, 64, 2000, 2001]
_MODELS = ["historical", "sbb", "filtered_bootstrap", "parametric"]


def _run_child(script: str, env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


_THREAD_PROBE = textwrap.dedent(
    """
    import json
    import numpy as np, numba
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _ids(out):
        for i in prange(out.shape[0]):
            out[i] = np.int64(numba.get_thread_id())

    o = np.zeros(8192, dtype=np.int64)
    _ids(o)
    print(json.dumps({"configured": int(numba.get_num_threads()),
                      "observed": int(np.unique(o).size),
                      "layer": str(numba.threading_layer())}))
    """
)


@pytest.mark.parametrize("threads", _THREADS)
def test_requested_threads_actually_execute(threads: int) -> None:
    """Proof of execution, not of configuration: count DISTINCT thread ids seen
    inside the parallel region."""
    got = _run_child(_THREAD_PROBE, {"NUMBA_NUM_THREADS": str(threads)})
    assert got["configured"] == threads
    assert got["observed"] == threads, (
        f"requested {threads} threads but only {got['observed']} executed the "
        f"parallel region (layer={got['layer']})"
    )


_CERT = textwrap.dedent(
    """
    import json, sys
    import numpy as np, polars as pl, numba
    from _certfix import inputs, FIDS, K
    from riskquant.solutions.risk_run.api import run_scenario_var_es
    from riskquant.solutions.scenario_models.api import build_scenario_generator
    from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
    from riskquant.solutions.sensitivity_pnl.api import FactorAxis
    from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
    from riskquant.solutions.run_identity.hashing import (
        hash_parametric_risk_data, hash_risk_factor_matrix)
    import hashlib

    def h(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(
            f"{{a.dtype.str}}|{{a.shape}}|".encode() + a.tobytes()).hexdigest()

    model, bs = sys.argv[1] if len(sys.argv) > 1 else "{model}", {bs}
    inp = inputs(model="{model}", batch_size={bs})
    rr = run_scenario_var_es(**inp)

    gen = build_scenario_generator(inp["scenario_config"], data=inp["data"])
    shocks, chunks = [], []
    # Reference: identity Proxy V1 rows (own column, own chart, loading 1)
    # applied with the Slice 2 kernel.
    for b in gen.iter_batches(chart_ids=inp["required_shift_type_id"],
                              driver_rf_ids=inp["required_factor_ids"],
                              batch_size={bs}):
        out = np.empty((K, b.shocks.shape[0]), dtype=np.float64, order="C")
        apply_linear_terms_to_base_levels(
            base_levels=inp["base_levels"], row_term_offsets=np.arange(K + 1, dtype=np.int32),
            term_column_uix=np.arange(K, dtype=np.int32), term_loadings=np.ones(K),
            target_shift_types=inp["required_shift_type_id"], shocks=b.shocks, out=out)
        shocks.append(np.array(b.shocks, copy=True)); chunks.append(out)
    stream = np.vstack(shocks); levels = np.ascontiguousarray(np.hstack(chunks))
    axis = FactorAxis.from_keys(pl.Series("factor_id", FIDS.tolist(), dtype=pl.Int32))
    trade = inp["taylor_derived"].calculator.run(
        axis.levels(base_levels=inp["base_levels"], shocked_levels=levels)).pnl

    data = inp["data"]
    print(json.dumps({{
        "threads": int(numba.get_num_threads()),
        "fixture_digest": hash_parametric_risk_data(data)
        if isinstance(data, ParametricRiskData) else hash_risk_factor_matrix(data),
        "scenario_stream": h(stream),
        "shocked_levels": h(levels),
        "trade_pnl": h(np.asarray(trade)),
        "var": [float(v).hex() for v in rr.var],
        "es": [float(v).hex() for v in rr.es],
        "portfolio_var": float(rr.portfolio_var).hex(),
        "identity": rr.identity,
        "reproducible": rr.reproducible,
    }}))
    """
)


def _cert(model: str, bs: int, threads: int) -> dict:
    got = _run_child(
        _CERT.format(model=model, bs=bs), {"NUMBA_NUM_THREADS": str(threads)}
    )
    assert got["threads"] == threads, "thread ceiling was clamped"
    return got


#: Ordered exactly as the release gate requires, so a failure names the EARLIEST
#: divergent object rather than only the final number.
_ORDER = ["scenario_stream", "shocked_levels", "trade_pnl", "var", "es", "identity"]


def _assert_same(a: dict, b: dict, ctx: str) -> None:
    for key in _ORDER:
        assert a[key] == b[key], f"first divergence at {key} ({ctx})"


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("threads", _THREADS)
def test_thread_count_does_not_change_any_result(model: str, threads: int) -> None:
    """Genuine multi-thread execution must not move any object in the chain."""
    ref = _cert(model, 64, 1)
    got = _cert(model, 64, threads)
    _assert_same(ref, got, f"{model} threads 1 vs {threads}")


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("batch_size", _BATCHES)
def test_batch_size_does_not_change_any_result_under_threads(
    model: str, batch_size: int
) -> None:
    """D041 re-proved through the official API while 8 threads actually run."""
    ref = _cert(model, 64, 8)
    got = _cert(model, batch_size, 8)
    _assert_same(ref, got, f"{model} bs 64 vs {batch_size} at 8 threads")


_FREEZE = textwrap.dedent(
    """
    import json
    from _certfix import freeze_parametric_data
    print(json.dumps({{"digest": freeze_parametric_data({path!r})}}))
    """
)


def test_parametric_blas_is_deterministic_under_thread_environments(tmp_path) -> None:
    """The one BLAS site (`np.matmul` in parametric.py) gets its own gate: the
    Taylor kernel's structural argument does not cover it.

    Building the parametric data is itself BLAS-sensitive, so it is frozen ONCE
    outside every attacked configuration and every child LOADS those exact
    bytes. Equal input is asserted first; only then is the result compared.
    """
    path = str(tmp_path / "parametric_data.npz")
    frozen = _run_child(_FREEZE.format(path=path), {"NUMBA_NUM_THREADS": "8"})["digest"]
    pinned = {"NUMBA_NUM_THREADS": "8", "RQ_CERT_PARAMETRIC_DATA": path}
    ref = _run_child(_CERT.format(model="parametric", bs=64), pinned)
    assert ref["threads"] == 8, "thread ceiling was clamped"
    assert ref["fixture_digest"] == frozen, "baseline did not price the frozen data"
    for env in (
        {"OMP_NUM_THREADS": "1"},
        {"OMP_NUM_THREADS": "8"},
        {"OPENBLAS_NUM_THREADS": "1"},
        {"OPENBLAS_NUM_THREADS": "8"},
        {"MKL_NUM_THREADS": "1"},
    ):
        got = _run_child(_CERT.format(model="parametric", bs=64), {**pinned, **env})
        assert got["fixture_digest"] == frozen, (
            f"certification construction failure under {env}: the child priced "
            "different parametric data than the frozen input"
        )
        _assert_same(ref, got, f"parametric under {env}")


def test_fresh_processes_agree_under_multithreading() -> None:
    a = _cert("sbb", 256, 8)
    b = _cert("sbb", 256, 8)
    _assert_same(a, b, "fresh processes at 8 threads")
    assert a["portfolio_var"] == b["portfolio_var"]


def test_identity_is_invariant_to_execution_configuration() -> None:
    """Threads and batch size are operational: they must not touch identity."""
    identities = {
        _cert(m, bs, t)["identity"]
        for m in ("sbb",)
        for bs in (3, 64, 2001)
        for t in (1, 8)
    }
    assert len(identities) == 1
