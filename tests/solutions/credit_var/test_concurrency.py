# tests/solutions/credit_var/test_concurrency.py
"""
WS_CREDIT_VAR_API S3: process state and concurrent calls (C.22 to C.25).

S3.T1 (C.22): `MODEL_CATALOG` and `SUPPORTED_RECIPES` are read-only mappings.

S3.T2 (C.23): a fresh child process runs one serial call per model, then 12
calls on a ThreadPoolExecutor of 4 threads. Each threaded result is byte-equal
to the serial result of its request, and the child exits with code 0. The
child is a separate process so that an abort of the interpreter (a threading
layer that rejects concurrent parallel regions, F-18) fails this test and does
not end the pytest process.

S3.T3 (C.24): two fresh child processes run the same requests, one with
``NUMBA_NUM_THREADS=1`` and one with the Numba default. The results are
byte-equal. The requests cover each of the 9 JTD kernels and both tail kernels.
On a host with one CPU the default is 1 thread, so the comparison has no power
there; the owner gate on a multi-core host is the evidence (S3 Exit).

S3.T4 (C.25): a probe replaces each import-site reference to a parallel Numba
dispatcher on the credit_var engine path. The process lock of `run.py` is
replaced by a lock that records its owner thread. 8 calls on 4 threads (both
entry points, sf_gaussian and mf_students_t) then show that each kernel entry
happens while the calling thread holds the lock, and that at most one thread
is inside a probe at a time.

Every comparison excludes ``provenance.generated_at`` (A_CONTRACT section 4.3).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from numba.core.dispatcher import Dispatcher

import riskquant
from riskquant.solutions.credit_var import api
from riskquant.solutions.credit_var import catalog as credit_var_catalog
from riskquant.solutions.credit_var import recipes as credit_var_recipes
from riskquant.solutions.credit_var import run as credit_var_run

_SEED = 20261103
_ALPHAS = (0.95, 0.99, 0.999)

# ---------------------------------------------------------------------------
# Request book
# ---------------------------------------------------------------------------


def _obligors(n: int) -> list[dict[str, Any]]:
    """Deterministic heterogeneous book: distinct EADs, PD in [0.005, 0.023].

    The lowest PD (0.005) is above the marshall_olkin systemic_pd of the book,
    as that model requires.
    """
    return [
        {
            "obligor_uid": uid,
            "ead": 1.0e5 * float(1 + (uid * 37) % 11),
            "pd": 0.005 + 0.002 * float(uid % 10),
            "lgd": 0.45 + 0.01 * float(uid % 5),
        }
        for uid in range(1, n + 1)
    ]


def _loadings(n: int) -> list[dict[str, Any]]:
    """Two sectors: the first half loads F1 = 0.5; the second half loads
    F1 = 0.3 and F2 = 0.4."""
    rows: list[dict[str, Any]] = []
    for uid in range(1, n + 1):
        if uid <= n // 2:
            rows.append({"obligor_uid": uid, "factor": "F1", "loading": 0.5})
        else:
            rows.append({"obligor_uid": uid, "factor": "F1", "loading": 0.3})
            rows.append({"obligor_uid": uid, "factor": "F2", "loading": 0.4})
    return rows


def _request(model: str, *, n_scenarios: int, **parts: Any) -> dict[str, Any]:
    n = 24
    return {
        "model": model,
        "obligors": _obligors(n),
        "n_scenarios": n_scenarios,
        "alphas": list(_ALPHAS),
        "seed": _SEED,
        **parts,
    }


def _book(n_scenarios: int) -> dict[str, dict[str, Any]]:
    """One request per JTD kernel (9 kernels). sf_gaussian also carries a
    forced-default obligor and a stress-test flag, so the overlay path and the
    stress section run too."""
    sf_gaussian = _request("sf_gaussian", n_scenarios=n_scenarios, params={"rho": 0.2})
    sf_gaussian["obligors"][23]["force_default"] = True
    sf_gaussian["obligors"][0]["in_stress_test"] = True
    return {
        "forced_default": _request("forced_default", n_scenarios=n_scenarios),
        "independent_bernoulli": _request(
            "independent_bernoulli", n_scenarios=n_scenarios
        ),
        "stress_regime_bernoulli": _request(
            "stress_regime_bernoulli",
            n_scenarios=n_scenarios,
            params={"stress_prob": 0.1, "stress_pd_multiplier": 3.0},
        ),
        "beta_binomial": _request(
            "beta_binomial",
            n_scenarios=n_scenarios,
            params={"alpha": 2.0, "beta": 98.0},
        ),
        "marshall_olkin": _request(
            "marshall_olkin", n_scenarios=n_scenarios, params={"systemic_pd": 0.004}
        ),
        "sf_gaussian": sf_gaussian,
        "sf_students_t": _request(
            "sf_students_t", n_scenarios=n_scenarios, params={"rho": 0.2, "nu": 6.0}
        ),
        "mf_gaussian": _request(
            "mf_gaussian", n_scenarios=n_scenarios, loadings=_loadings(24)
        ),
        "mf_students_t": _request(
            "mf_students_t",
            n_scenarios=n_scenarios,
            loadings=_loadings(24),
            params={"nu": 7.0},
        ),
    }


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

# Reads {"requests", "calls", "workers"} on stdin. Runs every request once in
# series, then each key of "calls" on a pool of "workers" threads (pool.map
# keeps the order). Writes the dumps and the Numba environment on stdout.
_CHILD = """
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numba

from riskquant.solutions.credit_var.api import run_credit_var

spec = json.loads(sys.stdin.buffer.read().decode("utf-8"))
requests = spec["requests"]


def dump(key):
    result = run_credit_var(requests[key])
    return result.model_dump_json(exclude={"provenance": {"generated_at"}})


serial = {key: dump(key) for key in requests}
concurrent = []
if spec["calls"]:
    with ThreadPoolExecutor(max_workers=spec["workers"]) as pool:
        concurrent = [
            [key, text]
            for key, text in zip(spec["calls"], pool.map(dump, spec["calls"]))
        ]
out = {
    "serial": serial,
    "concurrent": concurrent,
    "num_threads": numba.get_num_threads(),
    "threading_layer": numba.threading_layer(),
    "cpu_count": os.cpu_count(),
}
sys.stdout.buffer.write(json.dumps(out).encode("utf-8"))
"""


def _child_env(numba_num_threads: str | None) -> dict[str, str]:
    """The parent environment, the riskquant source root first on PYTHONPATH,
    and NUMBA_NUM_THREADS set to the given value or removed (Numba default)."""
    assert riskquant.__file__ is not None
    source_root = str(Path(riskquant.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        source_root,
        *filter(None, [env.get("PYTHONPATH")]),
    ])
    env.pop("NUMBA_NUM_THREADS", None)
    if numba_num_threads is not None:
        env["NUMBA_NUM_THREADS"] = numba_num_threads
    return env


def _run_child(
    spec: dict[str, Any], *, numba_num_threads: str | None
) -> tuple[int, dict[str, Any], str]:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=json.dumps(spec).encode("utf-8"),
        capture_output=True,
        env=_child_env(numba_num_threads),
        timeout=900,
        check=False,
    )
    stderr = child.stderr.decode("utf-8", errors="replace")
    payload: dict[str, Any] = (
        json.loads(child.stdout.decode("utf-8")) if child.returncode == 0 else {}
    )
    return child.returncode, payload, stderr


def _environment(payload: dict[str, Any]) -> str:
    return (
        f"threading_layer={payload.get('threading_layer')}, "
        f"num_threads={payload.get('num_threads')}, "
        f"cpu_count={payload.get('cpu_count')}"
    )


# ---------------------------------------------------------------------------
# S3.T1 (C.22)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mapping"),
    [
        ("MODEL_CATALOG", credit_var_catalog.MODEL_CATALOG),
        ("SUPPORTED_RECIPES", credit_var_recipes.SUPPORTED_RECIPES),
    ],
)
def test_s3_t1_process_global_mappings_are_read_only(name: str, mapping: Any) -> None:
    before = dict(mapping)
    existing = next(iter(before))
    with pytest.raises(TypeError):
        mapping[existing] = before[existing]
    with pytest.raises(TypeError):
        mapping["s3_t1_new_key"] = before[existing]
    with pytest.raises(TypeError):
        del mapping[existing]
    assert dict(mapping) == before, name


def test_s3_t1_api_exports_the_read_only_catalog() -> None:
    assert api.MODEL_CATALOG is credit_var_catalog.MODEL_CATALOG
    with pytest.raises(TypeError):
        api.MODEL_CATALOG["forced_default"] = api.MODEL_CATALOG["sf_gaussian"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# S3.T2 (C.23)
# ---------------------------------------------------------------------------

_T2_KEYS = ("forced_default", "sf_gaussian", "mf_students_t")


def test_s3_t2_concurrent_calls_equal_serial_calls() -> None:
    book = _book(n_scenarios=40_000)
    spec: dict[str, Any] = {
        "requests": {key: book[key] for key in _T2_KEYS},
        "calls": [key for _ in range(4) for key in _T2_KEYS],
        "workers": 4,
    }
    returncode, payload, stderr = _run_child(spec, numba_num_threads=None)
    assert returncode == 0, stderr
    concurrent = payload["concurrent"]
    assert len(concurrent) == 12
    mismatched = [
        index
        for index, (key, text) in enumerate(concurrent)
        if text != payload["serial"][key]
    ]
    assert mismatched == [], (
        f"threaded calls {mismatched} differ from their serial runs "
        f"({_environment(payload)})"
    )


# ---------------------------------------------------------------------------
# S3.T3 (C.24)
# ---------------------------------------------------------------------------


def test_s3_t3_results_do_not_depend_on_numba_thread_count() -> None:
    spec: dict[str, Any] = {
        "requests": _book(n_scenarios=20_000),
        "calls": [],
        "workers": 1,
    }
    code_one, one, stderr_one = _run_child(spec, numba_num_threads="1")
    assert code_one == 0, stderr_one
    code_default, default, stderr_default = _run_child(spec, numba_num_threads=None)
    assert code_default == 0, stderr_default
    assert one["num_threads"] == 1
    assert set(one["serial"]) == set(spec["requests"])
    differing = sorted(
        key for key in one["serial"] if one["serial"][key] != default["serial"][key]
    )
    assert differing == [], (
        f"results differ between NUMBA_NUM_THREADS=1 and the default "
        f"({_environment(default)}): {differing}"
    )


# ---------------------------------------------------------------------------
# S3.T4 (C.25)
# ---------------------------------------------------------------------------

#: The parallel-kernel entries of F-17, by import site: each evaluator module
#: calls its JTD kernel, and `_computer` calls the two tail kernels.
_NAMED_ENTRIES: tuple[tuple[str, str], ...] = (
    ("riskquant.credit.jtd.evaluators.beta_binomial", "kernel_beta_binomial_jtd"),
    ("riskquant.credit.jtd.evaluators.forced_default", "kernel_forced_default_jtd"),
    (
        "riskquant.credit.jtd.evaluators.independent_bernoulli",
        "kernel_independent_bernoulli_jtd",
    ),
    ("riskquant.credit.jtd.evaluators.marshall_olkin", "kernel_marshall_olkin_jtd"),
    (
        "riskquant.credit.jtd.evaluators.mf_copula_gaussian",
        "kernel_mf_gaussian_copula_jtd",
    ),
    (
        "riskquant.credit.jtd.evaluators.mf_copula_students_t",
        "kernel_mf_students_t_copula_jtd",
    ),
    (
        "riskquant.credit.jtd.evaluators.sf_copula_gaussian",
        "kernel_sf_gaussian_copula_jtd",
    ),
    (
        "riskquant.credit.jtd.evaluators.sf_copula_students_t",
        "kernel_sf_students_t_copula_jtd",
    ),
    (
        "riskquant.credit.jtd.evaluators.stress_regime_bernoulli",
        "kernel_stress_regime_bernoulli_jtd",
    ),
    ("riskquant.scenario_metrics._computer", "_kernel_lower_tail_var_es"),
    ("riskquant.scenario_metrics._computer", "_kernel_lower_tail_contribution"),
)

#: The entries that 8 calls over sf_gaussian and mf_students_t must reach.
_T4_REACHED: frozenset[tuple[str, str]] = frozenset({
    (
        "riskquant.credit.jtd.evaluators.sf_copula_gaussian",
        "kernel_sf_gaussian_copula_jtd",
    ),
    (
        "riskquant.credit.jtd.evaluators.mf_copula_students_t",
        "kernel_mf_students_t_copula_jtd",
    ),
    ("riskquant.scenario_metrics._computer", "_kernel_lower_tail_var_es"),
    ("riskquant.scenario_metrics._computer", "_kernel_lower_tail_contribution"),
})

_ENGINE_PREFIXES = (
    "riskquant.credit.",
    "riskquant.scenario_metrics.",
    "riskquant.solutions.credit_var.",
)

# Time a probe holds before it calls the kernel. Without the lock, two threads
# then overlap inside the probes even on one CPU, so the counter sees it.
_PROBE_DWELL_S = 0.01


def _parallel_entry_points() -> dict[tuple[str, str], ModuleType]:
    """Each module attribute on the engine path that refers to a parallel Numba
    dispatcher defined in another module: the import site through which Python
    code enters the kernel. The defining module is not an entry; patching it
    could also break the compilation of a jitted caller."""
    found: dict[tuple[str, str], ModuleType] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(_ENGINE_PREFIXES):
            continue
        for attr, value in list(vars(module).items()):
            if (
                isinstance(value, Dispatcher)
                and value.targetoptions.get("parallel")
                and value.py_func.__module__ != module_name
            ):
                found[module_name, attr] = module
    return found


class _OwnerLock:
    """Stand-in for `run._PARALLEL_KERNEL_LOCK` that records its owner thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: int | None = None

    def __enter__(self) -> _OwnerLock:
        self._lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.owner = None
        self._lock.release()


class _ProbeLog:
    """Shared record of all probes: hits, lock violations, peak occupancy."""

    def __init__(self, lock: _OwnerLock) -> None:
        self._lock = lock
        self._guard = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.hits: Counter[tuple[str, str]] = Counter()
        self.unlocked: list[tuple[str, str]] = []

    def wrap(
        self, entry: tuple[str, str], kernel: Callable[..., Any]
    ) -> Callable[..., Any]:
        def probe(*args: Any, **kwargs: Any) -> Any:
            with self._guard:
                self.hits[entry] += 1
                if self._lock.owner != threading.get_ident():
                    self.unlocked.append(entry)
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                time.sleep(_PROBE_DWELL_S)
                return kernel(*args, **kwargs)
            finally:
                with self._guard:
                    self._active -= 1

        return probe


def test_s3_t4_parallel_kernel_entry_holds_the_process_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(credit_var_run._PARALLEL_KERNEL_LOCK, type(threading.Lock()))
    entries = _parallel_entry_points()
    assert set(_NAMED_ENTRIES) <= set(entries), sorted(
        set(_NAMED_ENTRIES) - set(entries)
    )

    lock = _OwnerLock()
    log = _ProbeLog(lock)
    monkeypatch.setattr(credit_var_run, "_PARALLEL_KERNEL_LOCK", lock)
    for entry, module in entries.items():
        kernel = getattr(module, entry[1])
        monkeypatch.setattr(module, entry[1], log.wrap(entry, kernel))

    book = _book(n_scenarios=20_000)
    entry_points: tuple[Callable[[dict[str, Any]], object], ...] = (
        api.run_credit_var,
        api.simulate_portfolio_losses,
    )
    calls = [
        (entry_point, book[key])
        for _ in range(2)
        for key in ("sf_gaussian", "mf_students_t")
        for entry_point in entry_points
    ]
    assert len(calls) == 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(call, request) for call, request in calls]:
            future.result()

    assert sorted(_T4_REACHED - set(log.hits)) == []
    assert log.unlocked == [], f"kernel entries without the lock: {log.unlocked}"
    assert log.max_active == 1, f"threads inside the probes: {log.max_active}"
