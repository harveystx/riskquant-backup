# scripts/release/forward_node/certify_forward_node_production_scale.py
"""
VANILLA_FORWARD_NODE production-scale release certification.

Runs the official entry point `run_scenario_var_es` on the certification world
`tests/solutions/risk_run/_fwdcertfix.py` (raw frame -> ingest -> compile ->
run axis; 200 forward positions, 250 history rf ids, 140 axis points, mixed
FULL_REVAL and TAYLOR/DERIVED, DiscountingMode.NONE, reporting USD) and
certifies 3 outcomes. Each outcome has one verdict: PASS, FAIL or NOT EXECUTED.

correctness
    For each scenario model, every one of the M + 1 VaR and ES entries (M
    positions, then the portfolio) equals the oracle within the frozen tolerance
    policy of `tests/solutions/risk_run/test_vanilla_forward_node_production.py`
    (revision 8, section 5), built from `_fwdoracle`:

        V0        2^-48 T0_o
        dV FULL   2u|dV_o| + 2^-48 (T_s,o + T0_o) + phi
        dV DERIV  (6u + 2^-47) sum tau_bar + phi       (WS-D polynomial, NONE)
        dW        |r_s| tol_dV + |r_s - r_0| tol_V0 + 2^-48 (...)
        VaR       eps_in = max_s tol_dW;   ES eps_in + 2 gamma_{f+2} (||P_o|| + eps_in)
        portfolio eps_P,s = (1 + 2 gamma_{M-1}) sum tol_dW + 2 gamma_{M-1} sum |dW_o|

    The oracle reads the raw trade record (`ORACLE_ROWS`) and imports no
    valuation, Taylor, PnlGrid, D081 or VaR/ES code. Its P&L input is the (K, S)
    shocked-level matrix. The test captures that matrix by monkeypatching; this
    script does not patch any engine symbol. It composes the same public calls
    that `run_scenario_var_es` composes (proxy compile, lowering, world binding,
    scenario generator, linear-term application). For the historical model the
    levels are also checked against the independent `level_oracle` within the
    chart-specific `tol_level`, so level generation is certified separately.

determinism (D044 vocabulary)
    Per model: a repeat run at N threads and a run at 1 thread must equal the
    first N-thread run bitwise (`float.hex()` of every VaR / ES entry, run
    identity, result digest). Claim A (>= 2 distinct `numba.get_thread_id()`
    inside a real prange at N threads) and claim B (effective CPUs >= 2) are
    separate fields. PASS needs A, B and every comparison. A failed comparison
    is FAIL. Equal results without A or B are NOT EXECUTED, never PASS.

performance
    Wall time of each first run, process peak RSS and host physical memory are
    recorded. A measurement that fails is FAIL. There is no limit.

decisions
    The numeric checks above certify D044, D101.4, D102 and D103. The
    certificate certifies D066 and D101.3 at admission only: ingest admits the
    discount linkage and the absent delivery date, and the run prices the rows.
    DiscountingMode.NONE gives DF = 1 and the oracle has no DF input, so no
    numeric check depends on the linked curve or the asset-leg discount date.
    D103.4 and D103.6 require NONE for TAYLOR/DERIVED. The field
    `admission_only_reason` records this reason in the certificate.

The certificate is written beside this script:

    scripts/release/forward_node/forward_node_production_scale_certificate.json

Exit codes: 0 every verdict PASS; 1 any FAIL, or a verdict other than
determinism NOT EXECUTED; 2 no FAIL and determinism NOT EXECUTED. When 1 and 2
both apply, the exit code is 1.

Usage:
    python scripts/release/forward_node/certify_forward_node_production_scale.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import pathlib
import platform
import sys
import time
import traceback
from typing import Any, Final, TypeAlias

import numba
import numpy as np
import numpy.typing as npt

# Repository-root bootstrap: the world and the oracle live under tests/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(f"no RiskQuant repository root above {__file__!s}")
for _path in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tests.solutions.risk_run._fwdcertfix as fc  # noqa: E402
import tests.solutions.risk_run._fwdoracle as orc  # noqa: E402
from riskquant.config.proxy_model.compile import compile_proxy_model  # noqa: E402
from riskquant.jit import parallel_njit  # noqa: E402
from riskquant.simlab.shocks.axis_operator import lower_axis_operator  # noqa: E402
from riskquant.simlab.shocks.linear_term_apply import (  # noqa: E402
    apply_linear_terms_to_base_levels,
)
from riskquant.solutions.risk_run.api import run_scenario_var_es  # noqa: E402
from riskquant.solutions.risk_run.schemas import RiskRun  # noqa: E402
from riskquant.solutions.scenario_models.api import (  # noqa: E402
    build_scenario_generator,
)
from riskquant.solutions.scenario_models.world_columns import (  # noqa: E402
    bind_operator_to_world,
)

CERTIFICATE_PATH: Final[pathlib.Path] = (
    pathlib
    .Path(__file__)
    .resolve()
    .with_name("forward_node_production_scale_certificate.json")
)
PASS, FAIL, NOT_EXECUTED = "PASS", "FAIL", "NOT EXECUTED"
# Coverage class per decision. DiscountingMode.NONE (required for TAYLOR/DERIVED
# by D103.4 and D103.6) makes every DF equal 1, so the numeric checks cannot
# detect a wrong discount curve (D066) or a wrong asset-leg discount date
# (D101.3). Those decisions are certified by ingest admission only.
DECISIONS: Final[dict[str, Any]] = {
    "numeric_checks": ["D044", "D101.4", "D102", "D103"],
    "admission_only": ["D066", "D101.3"],
    "admission_only_reason": (
        "DiscountingMode.NONE gives DF = 1 on every leg and the oracle has no DF"
        " input (D103.4 and D103.6 require NONE for TAYLOR/DERIVED), so no"
        " numeric check depends on the linked discount curve (D066) or on the"
        " asset-leg discount date (D101.3); ingest admits these rows and the"
        " run prices them"
    ),
}
_PROBE_SIZE: Final[int] = 1 << 16

F64: TypeAlias = npt.NDArray[np.float64]


# ---- shocked levels: the public production composition, no patching ------------------


def shocked_levels(world: fc.CertWorld, run_inputs: dict[str, Any]) -> F64:
    """(K, S) levels by the calls `run_scenario_var_es` makes, in its order.

    Inputs are the run's own keyword arguments, so the scenario world, the
    proxy terms (none), the axis and the batch partition are the run's.
    """
    model = compile_proxy_model(
        authored_terms=run_inputs["proxy_terms"],
        static_config=world.rsc,
        as_of_date=fc.AS_OF,
        run_axis_rf_ids=world.axis.factor_ids,
    )
    operator = lower_axis_operator(
        model=model,
        axis_rf_ids=world.axis.factor_ids,
        axis_shift_type_ids=world.axis.shift_type_ids,
    )
    config, data = run_inputs["scenario_config"], run_inputs["data"]
    binding = bind_operator_to_world(operator, config, data)
    base = np.ascontiguousarray(run_inputs["base_levels"], dtype=np.float64)
    chunks: list[F64] = []
    for batch in build_scenario_generator(config, data=data).iter_batches(
        driver_rf_ids=binding.request_driver_rf_ids,
        chart_ids=binding.request_chart_ids,
        batch_size=run_inputs["batch_size"],
    ):
        out = np.empty((base.shape[0], batch.shocks.shape[0]), dtype=np.float64)
        apply_linear_terms_to_base_levels(
            base_levels=base,
            row_term_offsets=operator.row_term_offsets,
            term_column_uix=binding.term_column_uix,
            term_loadings=operator.term_loadings,
            target_shift_types=operator.target_shift_type_ids,
            shocks=batch.shocks,
            out=out,
        )
        chunks.append(out)
    return np.ascontiguousarray(np.hstack(chunks), dtype=np.float64)


# ---- independent oracle (raw trade record + _fwdoracle) --------------------------------


def _has_inverted_leg(row: fc.OracleRow) -> bool:
    return fc.INVERTED in (row.fx_asset, row.fx_strike)


def _ratio(error: float, tolerance: float) -> float:
    if tolerance > 0.0:
        return error / tolerance
    return 0.0 if error == 0.0 else math.inf


def _rate(x: F64, orientation: str) -> F64:
    r = orc.effective_rate(x, orientation)
    return np.ones_like(x) if r is None else r


def oracle_pnl(world: fc.CertWorld, levels: F64) -> dict[str, F64]:
    """Per UID (rows = sorted U): V0_o, dW_o, tol_dW, portfolio P_o and eps_P."""
    base = dict(world.base_by_name)
    lvs = {
        n: np.asarray(levels[i], dtype=np.float64) for i, n in enumerate(world.names)
    }
    delta = {n: lvs[n] - base[n] for n in world.names}
    x0 = np.array([base[fc.FX_NAME]])
    xs = lvs[fc.FX_NAME]
    v0, t0, dv, tdv, r0, rs = [], [], [], [], [], []
    for uid in world.uids.tolist():
        row = world.oracle_rows[uid]
        f0 = np.array([base[row.underlying]])
        a0, b0 = orc.forward_value(
            row.quantity,
            f0,
            row.strike,
            orc.effective_rate(x0, row.fx_asset),
            orc.effective_rate(x0, row.fx_strike),
        )
        if row.derived:
            if _has_inverted_leg(row):
                raise AssertionError(f"uid {uid}: inverted leg in U_D (D103.3)")
            value, tau = orc.forward_taylor_pnl(
                row.quantity,
                row.strike,
                row.underlying,
                None if row.fx_asset == fc.IDENTITY else fc.FX_NAME,
                None if row.fx_strike == fc.IDENTITY else fc.FX_NAME,
                base,
                delta,
            )
            dv.append(value)
            tdv.append(orc.tol_dv_derived(tau))
        else:
            a_s, b_s = orc.forward_value(
                row.quantity,
                lvs[row.underlying],
                row.strike,
                orc.effective_rate(xs, row.fx_asset),
                orc.effective_rate(xs, row.fx_strike),
            )
            dv.append(a_s - a0[0])
            tdv.append(orc.tol_dv_full(a_s - a0[0], b_s, b0))
        v0.append(float(a0[0]))
        t0.append(float(b0[0]))
        r0.append(float(_rate(x0, row.fx_report)[0]))
        rs.append(_rate(xs, row.fx_report))
    v0_a, t0_a, dv_a, tdv_a = np.array(v0), np.array(t0), np.array(dv), np.array(tdv)
    r0_a, rs_a = np.array(r0)[:, None], np.array(rs)
    dw = orc.reporting_dw(dv_a, v0_a[:, None], rs_a, r0_a)
    tol_dw = orc.tol_dw(
        rs_a, r0_a, tdv_a, orc.tol_v0(t0_a)[:, None], dv_a, v0_a[:, None]
    )
    return {
        "dw": dw,
        "tol_dw": tol_dw,
        "portfolio": dw.sum(axis=0),
        "eps_p": orc.portfolio_eps(tol_dw, dw),
    }


def reconcile_tail(world: fc.CertWorld, run: RiskRun, levels: F64) -> dict[str, Any]:
    """VaR and ES of all M + 1 entries against the oracle."""
    o = oracle_pnl(world, levels)
    scenarios = int(levels.shape[1])
    geometry = orc.tail_geometry(fc.CONFIDENCE_LEVEL, scenarios)
    f = geometry[1]
    series = [*o["dw"], o["portfolio"]]
    eps_in = [float(t.max()) for t in o["tol_dw"]] + [float(o["eps_p"].max())]
    var_ratio, es_ratio, failed = 0.0, 0.0, []
    for i, (p, eps) in enumerate(zip(series, eps_in, strict=True)):
        var_o, es_o = orc.var_es(p, geometry)
        tol_es = orc.tol_es(eps, float(np.abs(p).max()), f)
        e_var, e_es = abs(float(run.var[i]) - var_o), abs(float(run.es[i]) - es_o)
        var_ratio = max(var_ratio, _ratio(e_var, eps))
        es_ratio = max(es_ratio, _ratio(e_es, tol_es))
        if not (e_var <= eps and e_es <= tol_es):
            failed.append(i)
    return {
        "scenarios": scenarios,
        "entries": len(series),
        "entries_outside_tolerance": len(failed),
        "first_failed_entries": failed[:10],
        "max_var_error_over_tolerance": var_ratio,
        "max_es_error_over_tolerance": es_ratio,
        "portfolio_var": float(run.var[-1]),
        "portfolio_es": float(run.es[-1]),
        "ok": not failed and len(series) == run.var.shape[0] == run.es.shape[0],
    }


def reconcile_historical_levels(world: fc.CertWorld, levels: F64) -> dict[str, Any]:
    """Independent level oracle (horizon 1, beta 1.0) on every axis row."""
    worst, failed = 0.0, []
    for i, name in enumerate(world.names):
        _, _, x_hat, tol = orc.level_oracle(
            world.history_by_name[name],
            float(world.base[i]),
            1.0,
            world.chart_by_name[name],
        )
        err = np.abs(levels[i] - x_hat)
        worst = max(worst, float((err / tol).max()))
        if not bool((err <= tol).all()):
            failed.append(name)
    return {
        "axis_rows": len(world.names),
        "rows_outside_tolerance": len(failed),
        "first_failed_rows": failed[:10],
        "max_level_error_over_tolerance": worst,
        "ok": not failed,
    }


# ---- hardware and process measurements ---------------------------------------------


def effective_cpus() -> tuple[int, str, bool]:
    """(count, source, claim B usable) for THIS process (D044: affinity first)."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            n = len(affinity(0))
            return n, "sched_getaffinity", True
        except OSError:
            pass
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_mask, system_mask = ctypes.c_size_t(0), ctypes.c_size_t(0)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ok = kernel32.GetProcessAffinityMask(
            ctypes.c_void_p(kernel32.GetCurrentProcess()),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        )
        if ok and process_mask.value != 0:  # a zero mask counts as not available
            return bin(process_mask.value).count("1"), "GetProcessAffinityMask", True
    return int(os.cpu_count() or 0), "cpu_count", False


def peak_rss_bytes() -> int:
    """Process-lifetime peak resident memory in bytes."""
    if sys.platform == "win32":

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if not psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(kernel32.GetCurrentProcess()),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def host_physical_memory_bytes() -> int:
    """Installed physical memory of the host in bytes."""
    if sys.platform == "win32":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(status.ullTotalPhys)
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def _thread_ids_py(out: npt.NDArray[np.int64]) -> None:
    for i in numba.prange(out.shape[0]):
        out[i] = numba.get_thread_id()


# Applied as a call: the canonical decorator is untyped (pyproject numba override).
_thread_ids = parallel_njit(_thread_ids_py)


def observed_workers() -> int:
    """Distinct Numba thread ids inside a real prange at the current setting."""
    out: npt.NDArray[np.int64] = np.empty(_PROBE_SIZE, dtype=np.int64)
    _thread_ids(out)
    return int(np.unique(out).size)


# ---- sections --------------------------------------------------------------------------


def _signature(run: RiskRun) -> tuple[object, ...]:
    return (
        run.identity,
        run.result_digest,
        tuple(v.hex() for v in run.var.tolist()),
        tuple(v.hex() for v in run.es.tolist()),
    )


def _timed_run(
    world: fc.CertWorld, model: str
) -> tuple[RiskRun, dict[str, Any], float]:
    run_inputs = fc.inputs(world, model)
    t0 = time.perf_counter()
    run = run_scenario_var_es(**run_inputs)
    return run, run_inputs, time.perf_counter() - t0


def _error(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def world_record(world: fc.CertWorld) -> dict[str, Any]:
    rows = world.oracle_rows.values()
    relative = sum(r.underlying in fc.RELATIVE_NAMES for r in rows)

    def count(attr: str) -> dict[str, int]:
        return {
            o: sum(getattr(r, attr) == o for r in rows)
            for o in (fc.IDENTITY, fc.DIRECT, fc.INVERTED)
        }

    k_val = world.axis.num_valuation_factors
    return {
        "positions": int(world.uids.size),
        "spot_release_positions": fc.SPOT_RELEASE_M,
        "history_rf_ids": int(world.history.factor_ids.shape[0]),
        "axis_points": len(world.names),
        "valuation_prefix": int(k_val),
        "observations": fc.T,
        "stochastic_scenarios": fc.S,
        "full_reval_positions": int(world.full_uids.size),
        "taylor_derived_positions": int(world.derived_uids.size),
        "exact_node_positions": len(rows) - relative,
        "relative_node_positions": relative,
        "exact_without_delivery": sum(r["delivery_date"] is None for r in world.rows),
        "fx_asset_leg": count("fx_asset"),
        "fx_strike_leg": count("fx_strike"),
        "axis_charts": {
            c: sum(world.chart_by_name[n] == c for n in world.names)
            for c in sorted(set(world.chart_by_name.values()))
        },
        "confidence_level": fc.CONFIDENCE_LEVEL,
        "batch_size": fc.BATCH_SIZE,
        "discounting": fc.DISCOUNTING.name,
        "reporting_ccy": fc.REPORTING_CCY,
    }


def certify(world: fc.CertWorld) -> dict[str, Any]:
    n_threads = int(numba.config.NUMBA_NUM_THREADS)
    cpus, cpu_source, cpu_measured = effective_cpus()
    numba.set_num_threads(n_threads)
    workers_n = observed_workers()
    correctness: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    walls: dict[str, Any] = {}
    for model in fc.MODEL_CONFIGS:
        try:
            first, run_inputs, wall = _timed_run(world, model)
            walls[model] = wall if math.isfinite(wall) and wall > 0.0 else None
        except Exception as exc:  # fail loud: record and continue
            correctness[model] = {"ok": False, "error": _error(exc)}
            determinism[model] = {"ok": False, "error": _error(exc)}
            walls[model] = None
            continue
        try:
            levels = shocked_levels(world, run_inputs)
            rec = reconcile_tail(world, first, levels)
            if model == "historical":
                rec["levels"] = reconcile_historical_levels(world, levels)
                rec["ok"] = rec["ok"] and rec["levels"]["ok"]
            correctness[model] = rec
        except Exception as exc:
            correctness[model] = {"ok": False, "error": _error(exc)}
        try:
            repeat = run_scenario_var_es(**fc.inputs(world, model))
            numba.set_num_threads(1)
            try:
                single = run_scenario_var_es(**fc.inputs(world, model))
            finally:
                numba.set_num_threads(n_threads)
            determinism[model] = {
                "repeat_bitwise_equal": _signature(repeat) == _signature(first),
                "one_vs_n_threads_bitwise_equal": _signature(single)
                == _signature(first),
            }
            determinism[model]["ok"] = all(determinism[model].values())
        except Exception as exc:
            determinism[model] = {"ok": False, "error": _error(exc)}
    numba.set_num_threads(1)
    workers_1 = observed_workers()
    numba.set_num_threads(n_threads)

    claim_a = workers_n >= 2
    claim_b = cpu_measured and cpus >= 2
    compared_ok = all(d["ok"] for d in determinism.values())
    det_verdict = (
        FAIL
        if not compared_ok or workers_1 != 1
        else (PASS if claim_a and claim_b and n_threads >= 2 else NOT_EXECUTED)
    )
    corr_verdict = PASS if all(c["ok"] for c in correctness.values()) else FAIL

    performance: dict[str, Any] = {"wall_s": walls, "errors": []}
    for key, fn in (
        ("peak_rss_bytes", peak_rss_bytes),
        ("host_physical_memory_bytes", host_physical_memory_bytes),
    ):
        try:
            value = fn()
            performance[key] = value if value > 0 else None
        except Exception as exc:
            performance[key] = None
            performance["errors"].append(f"{key}: {_error(exc)}")
    measured = [*walls.values(), performance["peak_rss_bytes"]]
    measured.append(performance["host_physical_memory_bytes"])
    perf_verdict = PASS if all(v is not None for v in measured) else FAIL

    determinism_record = {
        "requested_threads": n_threads,
        "observed_workers_at_n_threads": workers_n,
        "observed_workers_at_1_thread": workers_1,
        "threading_layer": str(numba.threading_layer()),
        "claim_a_numba_multi_thread_execution": claim_a,
        "claim_b_multi_core_hardware": claim_b,
        "effective_cpus": cpus,
        "effective_cpus_source": cpu_source,
        "models": determinism,
    }
    return {
        "correctness": correctness,
        "determinism": determinism_record,
        "performance": performance,
        "verdict": {
            "correctness": corr_verdict,
            "determinism": det_verdict,
            "performance": perf_verdict,
        },
    }


def _json_safe(value: Any) -> Any:
    """Non-finite floats as strings, so a failed ratio never breaks the JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def exit_code(verdict: dict[str, str]) -> int:
    if FAIL in verdict.values():
        return 1
    if any(v == NOT_EXECUTED for k, v in verdict.items() if k != "determinism"):
        return 1
    return 2 if verdict["determinism"] == NOT_EXECUTED else 0


def environment() -> dict[str, Any]:
    import polars

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "polars": polars.__version__,
        "os_cpu_count": os.cpu_count(),
    }


def main() -> int:
    certificate: dict[str, Any] = {
        "certificate": "VANILLA_FORWARD_NODE production scale",
        "decisions": DECISIONS,
        "environment": environment(),
    }
    try:
        world = fc.build_world()
    except Exception as exc:
        certificate["world"] = {"error": _error(exc)}
        certificate["verdict"] = {
            "correctness": FAIL,
            "determinism": NOT_EXECUTED,
            "performance": FAIL,
        }
    else:
        certificate["world"] = world_record(world)
        certificate.update(certify(world))
    verdict = certificate["verdict"]
    verdict["exit_code"] = exit_code({
        k: verdict[k] for k in ("correctness", "determinism", "performance")
    })
    text = json.dumps(
        _json_safe(certificate), indent=2, sort_keys=True, allow_nan=False
    )
    CERTIFICATE_PATH.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict}, indent=2, sort_keys=True))
    print(f"certificate: {CERTIFICATE_PATH}")
    return int(verdict["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
