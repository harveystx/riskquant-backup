# scripts/solutions/credit_var/bench_credit_var_api.py
"""
Benchmark of the public credit-VaR entry point (WS_CREDIT_VAR_API, slice S6, C.32).

Measurements
------------
Each case runs in a fresh child process, ``--repetitions`` times (S6: 5). The
report gives min, median and max for each case.

M1  Wall time of ``import riskquant.solutions.credit_var.api``, empty Numba
    cache, then warm Numba cache.
M2  Wall time of the first and the second ``run_credit_var`` call in one
    process, for each grid model at N = 100, empty and warm Numba cache.
M3  Wall time and peak RSS of ``run_credit_var`` for N in ``--n-obligors`` and
    each grid model, at ``--n-scenarios`` (S6: 250_000).
M4  The M3 grid for ``simulate_portfolio_losses`` (default ``max_samples``).
M5  Wall time of the first and the second ``engine_version`` computation in a
    process. Needs S1; before S1 the report states OMITTED and the reason.

Why child processes
-------------------
Module import state, Numba dispatcher state and peak RSS are process-lifetime
quantities. Peak RSS is a high-water mark: it never decreases. Only a fresh
process gives an isolated value per case. The M1 child is a minimal ``-c``
program that loads only ``sys`` and ``time`` before the timed import, and it
asserts that no module of the api import closure is loaded before that point.
The other children run this script; its top-level imports are standard library
only, so NumPy, Numba and riskquant load only where a child needs them.

Why NUMBA_CACHE_DIR
-------------------
The credit_var kernels declare explicit Numba signatures, so Numba compiles
them at decoration, that is at import. With ``cache=True`` the compiled code
goes to the on-disk cache. The script sets ``NUMBA_CACHE_DIR`` for every child
to a directory under ``./.bench_credit_var/<UTC stamp>/``: one fresh empty
directory per cold repetition, and one shared warm directory that the
environment probe fills first. The script deletes no file. Each run adds one
stamp directory; the owner removes old ones by hand.

Why a warm-up call in M3 and M4
-------------------------------
M2 measures the first-call cost. M3 and M4 measure the steady-state call, so
each child first runs the same model at N = 10 and S = 1_000 (same argument
types, so the same Numba specializations), then times the full-size call. The
warm-up is far smaller than the timed call, so the timed call sets the process
peak RSS. That peak includes the import baseline; the report also gives the
peak before the timed call.

Usage (repository root, riskquant importable)::

    python scripts/solutions/credit_var/bench_credit_var_api.py

Exit code: 0 when every case passes, 1 when a case fails (the report lists it),
2 when a precondition fails before any measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

_SCRIPT_PATH: Final[Path] = Path(__file__).resolve()

_API_MODULE: Final[str] = "riskquant.solutions.credit_var.api"
_PROVENANCE_MODULE: Final[str] = "riskquant.solutions.credit_var.provenance"
_DEFAULT_ENGINE_VERSION_CALLABLE: Final[str] = f"{_PROVENANCE_MODULE}:engine_version"
# Modules that the timed M1 import must load for the first time.
_IMPORT_CLOSURE_ROOTS: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "numba",
    "pandas",
    "pydantic",
    "riskquant",
)

_WORK_ROOT_NAME: Final[str] = ".bench_credit_var"
_RESULT_MARKER: Final[str] = "BENCH_CHILD_RESULT "
_STDERR_TAIL_LINES: Final[int] = 25

# S6 specification values. A run with other values is not S6 evidence; the
# report says so in its first lines.
_SPEC_REPETITIONS: Final[int] = 5
_SPEC_N_SCENARIOS: Final[int] = 250_000
_SPEC_N_OBLIGORS: Final[tuple[int, ...]] = (10, 100, 1000)

_MODELS: Final[tuple[str, ...]] = ("forced_default", "sf_gaussian", "mf_students_t")
_ENTRY_POINTS: Final[tuple[str, ...]] = ("run_credit_var", "simulate_portfolio_losses")

# Synthetic book (S6 "Book"). Explicit model params and alphas pin the benchmark
# request: a later change of a catalog default does not change the measured run.
_BOOK_SEED: Final[int] = 20261030
_RUN_SEED: Final[int] = 12345
_PD_LOW: Final[float] = 0.005
_PD_HIGH: Final[float] = 0.05
_LGD: Final[float] = 0.45
_EAD_MEDIAN: Final[float] = 1_000_000.0  # lognormal median exp(mu)
_EAD_SIGMA: Final[float] = 1.0  # lognormal shape (log-space standard deviation)
_ALPHAS: Final[tuple[float, ...]] = (0.95, 0.975, 0.99, 0.999)
_MF_FACTORS: Final[tuple[str, ...]] = ("SYS_01", "SYS_02", "SYS_03")
_MF_LOADING: Final[float] = 0.3
_SF_RHO: Final[float] = 0.15
_MF_NU: Final[float] = 7.0

_M2_N_OBLIGORS: Final[int] = 100
_WARMUP_N_OBLIGORS: Final[int] = 10
_WARMUP_N_SCENARIOS: Final[int] = 1_000

_MIB: Final[float] = float(2**20)
_GIB: Final[float] = float(2**30)


# =============================================================================
# Shared helpers (parent and child)
# =============================================================================


def _model_params(model: str) -> dict[str, float]:
    if model == "forced_default":
        return {}
    if model == "sf_gaussian":
        return {"rho": _SF_RHO}
    if model == "mf_students_t":
        return {"nu": _MF_NU}
    raise ValueError(f"no benchmark params for model {model!r}")


def _peak_rss_bytes() -> int:
    """Return the peak resident memory of this process in bytes.

    Windows reports ``PeakWorkingSetSize`` (GetProcessMemoryInfo). POSIX reports
    ``ru_maxrss``, which Linux gives in KiB and macOS in bytes.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


# =============================================================================
# Child side
# =============================================================================


def _book_payload(model: str, n_obligors: int, n_scenarios: int) -> dict[str, Any]:
    """Build the deterministic synthetic request payload.

    Every value is a Python int, float or str (``tolist``), so the payload also
    passes the strict request coercion of WS_CREDIT_VAR_API C.18.
    """
    import numpy as np

    rng = np.random.default_rng(_BOOK_SEED)
    pd_values = rng.uniform(_PD_LOW, _PD_HIGH, size=n_obligors).tolist()
    ead_values = rng.lognormal(
        mean=math.log(_EAD_MEDIAN), sigma=_EAD_SIGMA, size=n_obligors
    ).tolist()
    uids = list(range(1, n_obligors + 1))
    payload: dict[str, Any] = {
        "model": model,
        "obligors": [
            {"obligor_uid": uid, "ead": ead, "pd": pd_value, "lgd": _LGD}
            for uid, ead, pd_value in zip(uids, ead_values, pd_values, strict=True)
        ],
        "params": _model_params(model),
        "n_scenarios": n_scenarios,
        "alphas": list(_ALPHAS),
        "seed": _RUN_SEED,
    }
    if model == "mf_students_t":
        payload["loadings"] = [
            {"obligor_uid": uid, "factor": factor, "loading": _MF_LOADING}
            for uid in uids
            for factor in _MF_FACTORS
        ]
    return payload


def _payload_digest(payload: dict[str, Any]) -> str:
    """Return a short SHA-256 of the payload; it proves an equal book per case."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _threading_layer() -> str:
    import numba

    try:
        return str(numba.threading_layer())
    except ValueError as exc:  # raised before the first parallel region
        return f"UNAVAILABLE ({exc})"


def _child_environment() -> dict[str, Any]:
    """Fill the warm cache and record the numerical environment.

    ``numba.threading_layer()`` is defined only after a parallel region ran, so
    one small call per grid model runs first.
    """
    api = importlib.import_module(_API_MODULE)
    for model in _MODELS:
        payload = _book_payload(model, _WARMUP_N_OBLIGORS, _WARMUP_N_SCENARIOS)
        api.run_credit_var(api.CreditVarRequest.model_validate(payload))
        api.simulate_portfolio_losses(api.CreditVarRequest.model_validate(payload))

    import numba
    import numpy
    import pydantic
    import scipy

    riskquant = importlib.import_module("riskquant")
    try:
        dist_version = importlib.metadata.version("riskquant")
    except importlib.metadata.PackageNotFoundError as exc:
        dist_version = f"UNAVAILABLE ({type(exc).__name__}: {exc})"
    return {
        "numpy_version": str(numpy.__version__),
        "scipy_version": str(scipy.__version__),
        "numba_version": str(numba.__version__),
        "pydantic_version": str(pydantic.VERSION),
        "threading_layer": _threading_layer(),
        "numba_config_num_threads": int(numba.config.NUMBA_NUM_THREADS),
        "numba_get_num_threads": int(numba.get_num_threads()),
        "riskquant_file": str(riskquant.__file__),
        "riskquant_dist_version": dist_version,
    }


def _child_first_second(
    model: str, n_obligors: int, n_scenarios: int
) -> dict[str, Any]:
    api = importlib.import_module(_API_MODULE)
    payload = _book_payload(model, n_obligors, n_scenarios)
    request = api.CreditVarRequest.model_validate(payload)
    t0 = time.perf_counter()
    api.run_credit_var(request)
    t1 = time.perf_counter()
    api.run_credit_var(request)
    t2 = time.perf_counter()
    return {
        "first_s": t1 - t0,
        "second_s": t2 - t1,
        "payload_sha16": _payload_digest(payload),
        "threading_layer": _threading_layer(),
    }


def _child_steady_state(
    entry: str, model: str, n_obligors: int, n_scenarios: int
) -> dict[str, Any]:
    if entry not in _ENTRY_POINTS:
        raise ValueError(f"unknown entry point {entry!r}")
    api = importlib.import_module(_API_MODULE)
    call = getattr(api, entry)
    warmup = _book_payload(model, _WARMUP_N_OBLIGORS, _WARMUP_N_SCENARIOS)
    call(api.CreditVarRequest.model_validate(warmup))

    payload = _book_payload(model, n_obligors, n_scenarios)
    digest = _payload_digest(payload)
    request = api.CreditVarRequest.model_validate(payload)
    del payload, warmup
    rss_before = _peak_rss_bytes()
    t0 = time.perf_counter()
    call(request)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_before_call_bytes": rss_before,
        "payload_sha16": digest,
        "threading_layer": _threading_layer(),
    }


def _child_engine_version(target: str) -> dict[str, Any]:
    # Import api first: in a real process the first engine_version computation
    # follows the api import, so module loading is outside this measurement.
    importlib.import_module(_API_MODULE)
    module_name, _, attribute = target.partition(":")
    compute = getattr(importlib.import_module(module_name), attribute)
    t0 = time.perf_counter()
    first = compute()
    t1 = time.perf_counter()
    second = compute()
    t2 = time.perf_counter()
    if not isinstance(first, str) or first != second:
        raise RuntimeError(
            f"{target} returned {first!r} then {second!r}; expected one str value."
        )
    return {"first_s": t1 - t0, "second_s": t2 - t1, "value": first}


def _run_child(spec_json: str) -> int:
    spec = json.loads(spec_json)
    if not isinstance(spec, dict):
        raise TypeError(f"child spec must be a JSON object, got {type(spec).__name__}")
    kind = spec["kind"]
    record: dict[str, Any]
    if kind == "environment":
        record = _child_environment()
    elif kind == "first_second":
        record = _child_first_second(
            str(spec["model"]), int(spec["n_obligors"]), int(spec["n_scenarios"])
        )
    elif kind == "steady_state":
        record = _child_steady_state(
            str(spec["entry"]),
            str(spec["model"]),
            int(spec["n_obligors"]),
            int(spec["n_scenarios"]),
        )
    elif kind == "engine_version":
        record = _child_engine_version(str(spec["target"]))
    else:
        raise ValueError(f"unknown child kind {kind!r}")
    sys.stdout.write(_RESULT_MARKER + json.dumps(record) + "\n")
    sys.stdout.flush()
    return 0


# =============================================================================
# Parent side: process control
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class _Failure:
    case: str
    repetition: int
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _CaseResult:
    case: str
    records: tuple[dict[str, Any], ...]
    failures: tuple[_Failure, ...]


def _script_child_args(spec: dict[str, Any]) -> tuple[str, ...]:
    return (str(_SCRIPT_PATH), "--child", json.dumps(spec))


def _m1_child_args() -> tuple[str, ...]:
    """Return the arguments of the minimal M1 child program.

    The program imports only ``sys`` and ``time`` (both loaded at interpreter
    start) before the timed import. A child that runs this script would first
    import argparse, json, subprocess, importlib.metadata and other modules
    that the api import closure also loads, and so bias M1 low (sandbox:
    0.05 s to 0.11 s of a 1.2 s warm import). ``-P`` keeps the working
    directory off ``sys.path``.
    """
    code = "\n".join([
        "import sys, time",
        f"roots = {_IMPORT_CLOSURE_ROOTS!r}",
        "pre = sorted(m for m in sys.modules if m.partition('.')[0] in roots)",
        "if pre:",
        "    raise RuntimeError('module table not clean: ' + repr(pre[:10]))",
        "t0 = time.perf_counter()",
        f"import {_API_MODULE}",
        "wall = time.perf_counter() - t0",
        "import json",
        f"print({_RESULT_MARKER!r} + json.dumps({{'wall_s': wall}}), flush=True)",
    ])
    return ("-P", "-c", code)


def _run_child_process(child_args: Sequence[str], cache_dir: Path) -> dict[str, Any]:
    """Run one child with its own NUMBA_CACHE_DIR; raise RuntimeError on failure."""
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, *child_args],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = [
        line[len(_RESULT_MARKER) :]
        for line in proc.stdout.splitlines()
        if line.startswith(_RESULT_MARKER)
    ]
    if proc.returncode != 0 or len(lines) != 1:
        tail = "\n".join(proc.stderr.splitlines()[-_STDERR_TAIL_LINES:])
        raise RuntimeError(
            f"child exit code {proc.returncode}, result lines {len(lines)}.\n"
            f"stderr tail:\n{tail}"
        )
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise RuntimeError(f"child result is not a JSON object: {lines[0][:200]}")
    return record


def _fresh_dir(path: Path) -> Path:
    path.mkdir(parents=False, exist_ok=False)
    return path


def _count_index_files(cache_dir: Path) -> int:
    return sum(1 for _ in cache_dir.rglob("*.nbi"))


def _progress(message: str) -> None:
    print(f"[bench] {message}", flush=True)


def _run_case(
    *,
    case: str,
    child_args: Sequence[str],
    repetitions: int,
    cache_dir_for: Callable[[int], Path],
) -> _CaseResult:
    """Run ``repetitions`` children of one case.

    ``cache_dir_for(rep)`` returns the NUMBA_CACHE_DIR of repetition ``rep``.
    The child record gains the index-file count of that directory before and
    after the child, which shows whether the child compiled or loaded.
    """
    records: list[dict[str, Any]] = []
    failures: list[_Failure] = []
    for rep in range(1, repetitions + 1):
        cache_dir = cache_dir_for(rep)
        nbi_before = _count_index_files(cache_dir)
        try:
            record = _run_child_process(child_args, cache_dir)
        except RuntimeError as exc:
            failures.append(_Failure(case=case, repetition=rep, detail=str(exc)))
            _progress(f"{case} rep {rep}/{repetitions}: FAILED")
            continue
        record["nbi_before"] = nbi_before
        record["nbi_after"] = _count_index_files(cache_dir)
        records.append(record)
        _progress(f"{case} rep {rep}/{repetitions}: ok")
    digests = {r["payload_sha16"] for r in records if "payload_sha16" in r}
    if len(digests) > 1:
        failures.append(
            _Failure(
                case=case,
                repetition=0,
                detail=f"request payload differs across repetitions: {sorted(digests)}",
            )
        )
    return _CaseResult(case=case, records=tuple(records), failures=tuple(failures))


# =============================================================================
# Parent side: environment record
# =============================================================================


def _command_output(args: Sequence[str], cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            list(args),
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return f"UNAVAILABLE ({exc})"
    if proc.returncode != 0:
        return f"UNAVAILABLE (exit {proc.returncode}: {proc.stderr.strip()[:200]})"
    return proc.stdout.strip()


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except OSError as exc:
            return f"UNAVAILABLE ({exc})"
        return str(value).strip()
    if sys.platform == "darwin":
        return _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as exc:
        return f"UNAVAILABLE ({exc})"
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model name":
            return value.strip()
    return f"UNAVAILABLE (no model name; platform.processor()={platform.processor()!r})"


def _total_ram() -> str:
    total: int
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = (
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(_MemoryStatusEx),)
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return f"UNAVAILABLE ({ctypes.WinError(ctypes.get_last_error())})"
        total = int(status.ullTotalPhys)
    elif sys.platform == "darwin":
        text = _command_output(["sysctl", "-n", "hw.memsize"])
        if not text.isdigit():
            return text
        total = int(text)
    else:
        total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    return f"{total / _GIB:.1f} GiB ({total} bytes)"


def _exact_command() -> str:
    argv = [sys.executable, *sys.argv]
    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _host_record(package_dir: Path) -> dict[str, str]:
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=package_dir)
    status = _command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=package_dir
    )
    if commit.startswith("UNAVAILABLE"):
        commit_record = commit
    elif status.startswith("UNAVAILABLE"):
        commit_record = f"{commit} dirty=UNKNOWN ({status})"
    else:
        commit_record = f"{commit} dirty={'true' if status else 'false'}"
    thread_vars = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("NUMBA_") or name.endswith("_NUM_THREADS")
    }
    return {
        "Command": _exact_command(),
        "Working directory": str(Path.cwd()),
        "Started (UTC)": datetime.now(UTC).isoformat(timespec="seconds"),
        "CPU model": _cpu_model(),
        "Logical processors (os.cpu_count)": str(os.cpu_count()),
        "RAM": _total_ram(),
        "OS": platform.platform(),
        "Python": sys.version.replace("\n", " "),
        "Python executable": sys.executable,
        "riskquant commit (git, tracked files)": commit_record,
        "riskquant package directory": str(package_dir),
        "NUMBA_NUM_THREADS (environment)": os.environ.get("NUMBA_NUM_THREADS", "unset"),
        "Thread variables in the parent environment": (
            json.dumps(thread_vars, sort_keys=True) if thread_vars else "none"
        ),
    }


# =============================================================================
# Parent side: report
# =============================================================================


def _fmt_stats(values: Sequence[float], *, scale: float = 1.0, digits: int = 4) -> str:
    if not values:
        return "no data"
    scaled = [v / scale for v in values]
    return (
        f"{min(scaled):.{digits}f} / {statistics.median(scaled):.{digits}f} / "
        f"{max(scaled):.{digits}f}"
    )


def _fmt_all(values: Sequence[float], *, scale: float = 1.0, digits: int = 4) -> str:
    return ", ".join(f"{v / scale:.{digits}f}" for v in values) if values else "none"


def _fmt_nbi(result: _CaseResult) -> str:
    pairs = (f"{r['nbi_before']}->{r['nbi_after']}" for r in result.records)
    return ", ".join(pairs) or "none"


def _status(result: _CaseResult, repetitions: int) -> str:
    ok = len(result.records)
    return (
        "ok"
        if ok == repetitions and not result.failures
        else f"FAILED ({ok}/{repetitions} ok)"
    )


def _report(
    *,
    args: argparse.Namespace,
    deviations: list[str],
    host: dict[str, str],
    environment: dict[str, Any],
    work_dir: Path,
    m1: dict[str, _CaseResult],
    m2: dict[tuple[str, str], _CaseResult],
    grid: dict[tuple[str, str, int], _CaseResult],
    m5: _CaseResult | None,
    m5_omitted: str | None,
    all_results: list[_CaseResult],
) -> str:
    reps = int(args.repetitions)
    lines: list[str] = [
        "# credit_var api benchmark (WS_CREDIT_VAR_API S6, C.32)",
        "",
    ]
    if deviations:
        lines += [
            "**NOT S6 EVIDENCE: the parameters differ from the S6 specification.**",
            "",
            *[f"- {d}" for d in deviations],
            "",
        ]
    lines += ["## Environment", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | `{v}` |" for k, v in host.items()]
    lines += [
        f"| NumPy | `{environment['numpy_version']}` |",
        f"| SciPy | `{environment['scipy_version']}` |",
        f"| Numba | `{environment['numba_version']}` |",
        f"| pydantic | `{environment['pydantic_version']}` |",
        f"| numba.threading_layer() | `{environment['threading_layer']}` |",
        f"| numba.config.NUMBA_NUM_THREADS | `{environment['numba_config_num_threads']}` |",
        f"| numba.get_num_threads() | `{environment['numba_get_num_threads']}` |",
        f"| riskquant module file | `{environment['riskquant_file']}` |",
        f"| riskquant distribution version | `{environment['riskquant_dist_version']}` |",
        f"| Work directory | `{work_dir}` |",
    ]
    layers = sorted({
        str(r["threading_layer"])
        for res in all_results
        for r in res.records
        if "threading_layer" in r
    })
    lines.append(
        f"| Threading layers seen in measured children | `{', '.join(layers) or 'none'}` |"
    )
    lines += [
        "",
        "## Method",
        "",
        f"- Each case runs in a fresh child process, {reps} repetitions. Cells give "
        "min / median / max. Wall time is `time.perf_counter` inside the child.",
        "- Book: obligor_uid 1..N; PD ~ U[0.005, 0.05); EAD ~ lognormal(mu = ln(1e6), "
        f"sigma = 1.0); LGD 0.45; generator `numpy.random.default_rng({_BOOK_SEED})`.",
        "- Request: alphas (0.95, 0.975, 0.99, 0.999); seed 12345; sf_gaussian "
        "params {rho: 0.15}; mf_students_t params {nu: 7.0} and loadings 0.3 on "
        "SYS_01, SYS_02, SYS_03 for each obligor; forced_default params {}.",
        "- Numba cache: `NUMBA_CACHE_DIR` per child. Empty: one fresh directory per "
        "repetition. Warm: one shared directory, filled by the environment probe "
        "(import, then one small call per model of each entry point). `.nbi` "
        "columns count cache index files before and after the child.",
        "- M3 and M4: each child first calls the same entry point with the same "
        f"model at N = {_WARMUP_N_OBLIGORS}, S = {_WARMUP_N_SCENARIOS} (warm-up), "
        "then times the full call. Peak RSS is the process peak after the timed "
        "call; the pre-call column is the process peak after import, warm-up and "
        "request construction.",
        "- Peak RSS: Windows `PeakWorkingSetSize` (GetProcessMemoryInfo); POSIX "
        "`ru_maxrss`. 1 MiB = 2**20 bytes.",
        "- Request validation (`CreditVarRequest.model_validate`) is outside every "
        "timed region. A call-time validation inside the entry point is inside it.",
        "- M1 child: `python -P -c <program>`; it loads only `sys` and `time` "
        "before the timed import. The Python bytecode cache (`__pycache__`) is not "
        "controlled.",
        "- After S1, the first `run_credit_var` call of a process also computes "
        "`engine_version` (M5). M2 includes that cost; M3 and M4 do not.",
        "",
    ]

    lines += [
        "## M1: import riskquant.solutions.credit_var.api",
        "",
        "| Numba cache | Status | Wall s (min / median / max) | All repetitions s | .nbi before -> after |",
        "|---|---|---|---|---|",
    ]
    for state, res in m1.items():
        walls = [float(r["wall_s"]) for r in res.records]
        lines.append(
            f"| {state} | {_status(res, reps)} | {_fmt_stats(walls)} | "
            f"{_fmt_all(walls)} | {_fmt_nbi(res)} |"
        )
    lines.append("")

    lines += [
        f"## M2: first and second run_credit_var call in one process (N = {_M2_N_OBLIGORS}, S = {args.n_scenarios})",
        "",
        "| Model | Numba cache | Status | First call s | Second call s | First - second s (median) | .nbi before -> after |",
        "|---|---|---|---|---|---|---|",
    ]
    for (model, state), res in m2.items():
        first = [float(r["first_s"]) for r in res.records]
        second = [float(r["second_s"]) for r in res.records]
        delta = (
            f"{statistics.median(f - s for f, s in zip(first, second, strict=True)):.4f}"
            if first
            else "no data"
        )
        lines.append(
            f"| {model} | {state} | {_status(res, reps)} | {_fmt_stats(first)} | "
            f"{_fmt_stats(second)} | {delta} | {_fmt_nbi(res)} |"
        )
    lines.append("")

    for number, entry in (
        ("M3", "run_credit_var"),
        ("M4", "simulate_portfolio_losses"),
    ):
        suffix = (
            " (default max_samples)" if entry == "simulate_portfolio_losses" else ""
        )
        lines += [
            f"## {number}: {entry}{suffix}, S = {args.n_scenarios}, warm Numba cache",
            "",
            "| Model | N | Status | Wall s | Peak RSS MiB | Peak RSS before call MiB (median) | Payload sha256[:16] |",
            "|---|---|---|---|---|---|---|",
        ]
        for (case_entry, model, n_obligors), res in grid.items():
            if case_entry != entry:
                continue
            walls = [float(r["wall_s"]) for r in res.records]
            peaks = [float(r["peak_rss_bytes"]) for r in res.records]
            before = [float(r["peak_rss_before_call_bytes"]) for r in res.records]
            digests = sorted({str(r["payload_sha16"]) for r in res.records})
            before_text = (
                f"{statistics.median(before) / _MIB:.1f}" if before else "no data"
            )
            lines.append(
                f"| {model} | {n_obligors} | {_status(res, reps)} | {_fmt_stats(walls)} | "
                f"{_fmt_stats(peaks, scale=_MIB, digits=1)} | {before_text} | "
                f"{', '.join(digests) or 'none'} |"
            )
        lines.append("")

    lines += ["## M5: first engine_version computation in a process", ""]
    if m5 is None:
        lines += [f"OMITTED: {m5_omitted}", ""]
    else:
        first = [float(r["first_s"]) for r in m5.records]
        second = [float(r["second_s"]) for r in m5.records]
        values = sorted({str(r["value"]) for r in m5.records})
        lines += [
            f"Callable: `{args.engine_version_callable}`",
            "",
            "| Status | First call s | Second call s | Values |",
            "|---|---|---|---|",
            f"| {_status(m5, reps)} | {_fmt_stats(first, digits=6)} | "
            f"{_fmt_stats(second, digits=6)} | {', '.join(f'`{v}`' for v in values) or 'none'} |",
            "",
        ]

    failures = [f for res in all_results for f in res.failures]
    lines += ["## Failures", ""]
    if not failures:
        lines += ["none", ""]
    for failure in failures:
        lines += [
            f"### {failure.case}, repetition {failure.repetition}",
            "",
            "```text",
            failure.detail,
            "```",
            "",
        ]
    return "\n".join(lines)


# =============================================================================
# Parent side: orchestration
# =============================================================================


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark riskquant.solutions.credit_var.api (WS_CREDIT_VAR_API S6)."
    )
    parser.add_argument("--repetitions", type=int, default=_SPEC_REPETITIONS)
    parser.add_argument("--n-scenarios", type=int, default=_SPEC_N_SCENARIOS)
    parser.add_argument(
        "--n-obligors", type=int, nargs="+", default=list(_SPEC_N_OBLIGORS)
    )
    parser.add_argument(
        "--engine-version-callable",
        default=_DEFAULT_ENGINE_VERSION_CALLABLE,
        help="MODULE:ATTRIBUTE of the zero-argument engine_version function (M5).",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child is None:
        if args.repetitions < 1 or args.n_scenarios < 1:
            parser.error("--repetitions and --n-scenarios must be >= 1.")
        if any(n < 1 for n in args.n_obligors):
            parser.error("--n-obligors values must be >= 1.")
        module_name, sep, attribute = str(args.engine_version_callable).partition(":")
        if not (module_name and sep and attribute):
            parser.error("--engine-version-callable must be MODULE:ATTRIBUTE.")
    return args


def _spec_deviations(args: argparse.Namespace) -> list[str]:
    deviations: list[str] = []
    if args.repetitions != _SPEC_REPETITIONS:
        deviations.append(
            f"repetitions {args.repetitions}, S6 value {_SPEC_REPETITIONS}"
        )
    if args.n_scenarios != _SPEC_N_SCENARIOS:
        deviations.append(
            f"n_scenarios {args.n_scenarios}, S6 value {_SPEC_N_SCENARIOS}"
        )
    if tuple(args.n_obligors) != _SPEC_N_OBLIGORS:
        deviations.append(
            f"N values {tuple(args.n_obligors)}, S6 values {_SPEC_N_OBLIGORS}"
        )
    return deviations


def _main(args: argparse.Namespace) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # A path or CPU name outside the console code page must not stop a run.
        sys.stdout.reconfigure(errors="backslashreplace")
    package_spec = importlib.util.find_spec("riskquant")
    if package_spec is None or package_spec.origin is None:
        print(
            "[bench] PRECONDITION FAILED: package riskquant is not importable.",
            flush=True,
        )
        return 2
    package_dir = Path(package_spec.origin).resolve().parent
    # Record git state before the work directory exists in the working tree.
    host = _host_record(package_dir)
    deviations = _spec_deviations(args)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    work_root = Path.cwd() / _WORK_ROOT_NAME
    work_root.mkdir(exist_ok=True)
    work_dir = _fresh_dir(work_root / stamp)
    warm_dir = _fresh_dir(work_dir / "numba_cache_warm")
    _progress(f"work directory {work_dir}")
    if deviations:
        _progress("NOT S6 EVIDENCE: " + "; ".join(deviations))

    reps = int(args.repetitions)
    n_scenarios = int(args.n_scenarios)
    all_results: list[_CaseResult] = []

    def cold_dir(prefix: str) -> Callable[[int], Path]:
        def fresh(rep: int) -> Path:
            return _fresh_dir(work_dir / f"numba_cache_{prefix}_rep{rep}")

        return fresh

    def warm(_rep: int) -> Path:
        return warm_dir

    env_result = _run_case(
        case="environment probe (warm-cache fill)",
        child_args=_script_child_args({"kind": "environment"}),
        repetitions=1,
        cache_dir_for=warm,
    )
    all_results.append(env_result)
    if not env_result.records:
        print(env_result.failures[0].detail, flush=True)
        print("[bench] PRECONDITION FAILED: the environment probe failed.", flush=True)
        return 2
    environment = env_result.records[0]

    m1 = {
        "empty": _run_case(
            case="M1 import, empty cache",
            child_args=_m1_child_args(),
            repetitions=reps,
            cache_dir_for=cold_dir("m1"),
        ),
        "warm": _run_case(
            case="M1 import, warm cache",
            child_args=_m1_child_args(),
            repetitions=reps,
            cache_dir_for=warm,
        ),
    }
    all_results += m1.values()

    m2: dict[tuple[str, str], _CaseResult] = {}
    for model in _MODELS:
        for state in ("empty", "warm"):
            spec = {
                "kind": "first_second",
                "model": model,
                "n_obligors": _M2_N_OBLIGORS,
                "n_scenarios": n_scenarios,
            }
            m2[(model, state)] = _run_case(
                case=f"M2 {model}, {state} cache",
                child_args=_script_child_args(spec),
                repetitions=reps,
                cache_dir_for=cold_dir(f"m2_{model}") if state == "empty" else warm,
            )
    all_results += m2.values()

    # S1 creates the provenance module. Its absence is the only reason to omit
    # M5; with S1 present, a wrong --engine-version-callable fails loudly.
    m5: _CaseResult | None = None
    m5_omitted: str | None = None
    if importlib.util.find_spec(_PROVENANCE_MODULE) is None:
        m5_omitted = (
            f"module {_PROVENANCE_MODULE} is not importable. S1 creates it, so this "
            "tree predates S1 and has no engine_version computation to measure."
        )
    if m5_omitted is None:
        m5 = _run_case(
            case="M5 engine_version",
            child_args=_script_child_args({
                "kind": "engine_version",
                "target": str(args.engine_version_callable),
            }),
            repetitions=reps,
            cache_dir_for=warm,
        )
        all_results.append(m5)
    else:
        _progress(f"M5 OMITTED: {m5_omitted}")

    grid: dict[tuple[str, str, int], _CaseResult] = {}
    for number, entry in zip(("M3", "M4"), _ENTRY_POINTS, strict=True):
        for model in _MODELS:
            for n_obligors in args.n_obligors:
                spec = {
                    "kind": "steady_state",
                    "entry": entry,
                    "model": model,
                    "n_obligors": int(n_obligors),
                    "n_scenarios": n_scenarios,
                }
                grid[(entry, model, int(n_obligors))] = _run_case(
                    case=f"{number} {entry} {model} N={n_obligors}",
                    child_args=_script_child_args(spec),
                    repetitions=reps,
                    cache_dir_for=warm,
                )
    all_results += grid.values()

    report = _report(
        args=args,
        deviations=deviations,
        host=host,
        environment=environment,
        work_dir=work_dir,
        m1=m1,
        m2=m2,
        grid=grid,
        m5=m5,
        m5_omitted=m5_omitted,
        all_results=all_results,
    )
    report_path = work_dir / "report.md"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report + "\n")
    print(report, flush=True)
    _progress(f"report written to {report_path}")
    failed = any(res.failures for res in all_results)
    _progress("RESULT: " + ("FAILED (see Failures)" if failed else "all cases ok"))
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child is not None:
        return _run_child(str(args.child))
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
