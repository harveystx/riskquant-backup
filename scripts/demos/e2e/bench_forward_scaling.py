# scripts/demos/e2e/bench_forward_scaling.py
"""
RiskQuant VANILLA_FORWARD_NODE -- benchmark: build and run time versus book
size M and scenario count S, for FULL_REVAL and TAYLOR/DERIVED SEPARATELY.

Not a regression gate. Timings are informational and never asserted.
Correctness is asserted: the CORRECTNESS GATE runs first, on the chapter-10
book, and a failure aborts the benchmark (a fast wrong run is not a result):

    FULL_REVAL gate      production V0 == hand formula V = Q (F X_a - K X_k)
    TAYLOR/DERIVED gate  certified Taylor P&L == full-revaluation P&L on every
                         historical scenario, within 8 u * scale (chapter 11)

Two paths, never one combined number
------------------------------------
    FULL_REVAL      build = ingest + compile + run axis + history
                    run   = run_scenario_var_es, every position FULL_REVAL
    TAYLOR/DERIVED  build = the FULL_REVAL build + base valuation (V0)
                            + run_sensitivities + certification + book compile
                    run   = run_scenario_var_es, every position TAYLOR/DERIVED
`total` = build + run (median run). The run timer covers the whole production
call: identity, base valuation, scenario generation, shocked levels, P&L
(pricer or Taylor kernel), D081 conversion and VaR / ES. No private function
is wrapped: stage timings inside the run are not reported.

Sweeps
------
    M   historical, S fixed, M forward positions varied
    S   historical, M fixed, S = number of 1-day scenarios varied
Book of M forwards: unique sparse UIDs; underlyings cycle over four NODE rf
ids (two exact DEMO_GAS_FWD pillars, BRENT_CRUDE_FWD[M+0], WTI_CRUDE_FWD[M+0]);
every third forward settles in EUR (direct USDEUR legs); quantities and
strikes from a local seeded RNG. No repeated UIDs, no inverted legs.

Process model (as `bench_spot_scaling.py`)
------------------------------------------
One subprocess per cell (clean process-lifetime peak RSS), a warm-up run on a
small book first (Numba load, never timed), `--repeats` timed runs. A cell
whose ESTIMATED memory exceeds `--mem-budget-gib` is reported as SKIP with
the estimate; it is never run and never extrapolated.

Run:
    python scripts/demos/e2e/bench_forward_scaling.py --quick
    python scripts/demos/e2e/bench_forward_scaling.py --path full --sweep M
    python scripts/demos/e2e/bench_forward_scaling.py --json-out bench.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _forward_world as fw

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import MethodologyDeclaration, ReportingSpec
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

U32 = 2.0**-24  # float32 unit roundoff: the P&L grid stores float32
HISTORICAL = {"model": "historical", "horizon": 1}

BYTES_PER_CELL_ESTIMATE = 64.0  # per (position, scenario); an ESTIMATE, labelled so
NODES = (
    ("DEMO_GAS_FWD[1M]", None),
    ("DEMO_GAS_FWD[3M]", dt.date(2026, 4, 15)),
    ("BRENT_CRUDE_FWD[M+0]", fw.D_FEB),
    ("WTI_CRUDE_FWD[M+0]", fw.D_FEB),
)


class BenchCheckError(AssertionError):
    """A correctness check failed: the benchmark stops."""


@dataclasses.dataclass(frozen=True)
class Cell:
    path: str  # "full" or "derived"
    sweep: str
    num_positions: int
    num_scenarios: int


# ---- book --------------------------------------------------------------------------


def book(num_positions: int, seed: int = 20260115) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_positions):
        node, delivery = NODES[i % len(NODES)]
        settle = "EUR" if i % 3 == 2 else "USD"
        qty = float(np.round(rng.uniform(-200.0, 200.0), 2)) or 1.0
        strike = float(np.round(fw.LEVELS[node] * (1.0 + rng.normal(0.0, 0.05)), 2))
        rows.append(
            fw.forward(
                1_000_001 + 7 * i,
                node,
                qty,
                strike,
                settle_ccy=settle,
                delivery=delivery,
                settlement=(delivery or fw.D_MAR) + dt.timedelta(days=2),
            )
        )
    return rows


# ---- correctness gate (runs before any timing) ------------------------------------------


def correctness_gate() -> None:
    """The chapter-10 book: V0 vs hand formula, then Taylor vs FULL P&L on the
    SAME production scenarios (the checks chapters 10 and 11 print in full)."""
    rsc = fw.static_config()
    world = fw.build_world(fw.MAIN_BOOK, reporting_ccy="EUR", rsc=rsc)
    v0 = fw.base_value_by_uid(world)
    base_levels = {n: np.array([fw.LEVELS[n]]) for n in fw.LEVELS}
    for row in fw.MAIN_BOOK:
        hand = fw.hand_value(row, fw.LEVELS)
        if abs(v0[row["uid"]] - hand) > 2.0**-48 * float(
            fw.leg_scale(row, base_levels)[0]
        ):
            raise BenchCheckError(
                f"FULL_REVAL gate: V0 of uid {row['uid']} != hand formula"
            )
    print(
        "  correctness gate FULL_REVAL      : V0 == hand formula for every row   PASS"
    )
    block = fw.certified_taylor_block(world, fw.base_sensitivities(world), v0)
    shocked = fw.production_shocked_levels(
        world, scenario_config=parse_scenario_model_config(HISTORICAL)
    )
    full, taylor = fw.scenario_pnl_full_and_taylor(world, block, shocked)
    by_name = {n: shocked[i] for i, n in enumerate(world.names)}
    for row in fw.MAIN_BOOK:
        uid = row["uid"]
        tol = 8.0 * U32 * (np.abs(full[uid]) + fw.leg_scale(row, by_name))
        if not np.all(np.abs(full[uid] - taylor[uid]) <= tol):
            raise BenchCheckError(f"TAYLOR gate: uid {uid} Taylor P&L != FULL P&L")
    print(
        "  correctness gate TAYLOR/DERIVED  : Taylor P&L == FULL P&L, every scenario PASS"
    )


# ---- one cell (worker process) --------------------------------------------------------


def build(cell: Cell, rsc: Any) -> tuple[fw.ForwardWorld, Any]:
    world = fw.build_world(
        book(cell.num_positions),
        reporting_ccy="USD",
        rsc=rsc,
        num_observations=cell.num_scenarios + 1,
    )
    if cell.path == "full":
        return world, None
    v0 = fw.base_value_by_uid(world)
    return world, fw.certified_taylor_block(world, fw.base_sensitivities(world), v0)


def execute(cell: Cell, world: fw.ForwardWorld, block: Any) -> Any:
    """One production call; every position in ONE methodology partition."""
    uids = world.uids
    none = np.empty(0, dtype=np.int64)
    full = cell.path == "full"
    return run_scenario_var_es(
        scenario_config=parse_scenario_model_config(HISTORICAL),
        data=world.data,
        required_factor_ids=world.axis.factor_ids.copy(),
        required_shift_type_id=world.axis.shift_type_ids.copy(),
        proxy_terms=(),
        base_levels=world.base.copy(),
        methodology=MethodologyDeclaration(
            production_position_uids=uids,
            full_reval_uids=uids if full else none,
            taylor_supplied_uids=none,
            taylor_derived_uids=none if full else uids,
        ),
        valuation_context=world.context,
        reporting=ReportingSpec(reporting_ccy=world.reporting_ccy),
        confidence_level=0.95,
        batch_size=64,
        taylor_derived=None if full else block,
    )


def check_run(result: Any, num_positions: int) -> None:
    if result.var.shape != (num_positions + 1,) or not (
        np.isfinite(result.var).all() and np.isfinite(result.es).all()
    ):
        raise BenchCheckError("run output malformed (shape or non-finite VaR / ES)")


if sys.platform == "win32":  # the exact working setup of bench_spot_scaling.py
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
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
        ]

    # Typed signatures are required: without them ctypes passes the 64-bit
    # pseudo-handle of GetCurrentProcess as a C int, the call fails, and an
    # unchecked failure leaves every counter at 0 (the "0.0 MiB" defect).
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    _KERNEL32.GetCurrentProcess.argtypes = []
    _KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
    _PSAPI.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    _PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def peak_rss_mib() -> float | None:
    """
    Process-lifetime peak resident memory in MiB, or None when the platform
    offers no measurement (reported as UNAVAILABLE, never as a number).
    Windows: `PeakWorkingSetSize` (bytes). Linux: `ru_maxrss` (KiB). macOS: bytes.
    """
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = _PSAPI.GetProcessMemoryInfo(
            _KERNEL32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return float(counters.PeakWorkingSetSize) / 2**20
    try:
        import resource
    except ImportError:
        return None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024.0


def run_worker(cell: Cell, repeats: int) -> dict[str, Any]:
    rsc = fw.static_config()
    warm = Cell(cell.path, "warm-up", 8, 16)
    world, block = build(warm, rsc)
    check_run(execute(warm, world, block), 8)  # Numba load; never timed
    start = time.perf_counter()
    world, block = build(cell, rsc)
    build_s = time.perf_counter() - start
    # Untimed validation run of THIS cell: the production run verifies the
    # Taylor base binding of this cell's own block (bitwise V0 digest) and the
    # output is checked before any timing is recorded.
    check_run(execute(cell, world, block), cell.num_positions)
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = execute(cell, world, block)
        runs.append(time.perf_counter() - start)
        check_run(result, cell.num_positions)
    run_s = statistics.median(runs)
    return {
        "status": "ok",
        "build_s": build_s,
        "run_s": run_s,
        "runs_s": runs,
        "total_s": build_s + run_s,
        "peak_rss_mib": peak_rss_mib(),
        "num_factors": int(world.axis.factor_ids.size),
    }


def run_cell_subprocess(cell: Cell, repeats: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "cell.json"
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--cell-json",
                json.dumps(dataclasses.asdict(cell)),
                "--repeats",
                str(repeats),
                "--result-path",
                str(result_path),
            ],
            env=env,
            check=False,
        )
        if result_path.is_file():
            return dict(json.loads(result_path.read_text(encoding="utf-8")))
    return {"status": "fail", "returncode": proc.returncode}


# ---- parent --------------------------------------------------------------------------


def _mib(value: float | None) -> str:
    """Measured peak RSS in MiB, or UNAVAILABLE (never a substitute number)."""
    return f"{value:9.1f}" if value is not None else "UNAVAILABLE"


def cells(args: argparse.Namespace) -> list[Cell]:
    if args.quick:
        m_grid, s_fixed, s_grid, m_fixed = (50, 400), 64, (64, 256), 100
    else:
        m_grid, s_fixed, s_grid, m_fixed = (
            (100, 1_000, 10_000, 50_000),
            260,
            (130, 520, 2_080),
            2_000,
        )
    paths = ("full", "derived") if args.path == "both" else (args.path,)
    out = []
    for path in paths:
        if args.sweep in ("M", "all"):
            out += [Cell(path, "M", m, s_fixed) for m in m_grid]
        if args.sweep in ("S", "all"):
            out += [Cell(path, "S", m_fixed, s) for s in s_grid]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", default="both", choices=("full", "derived", "both"))
    parser.add_argument("--sweep", default="all", choices=("M", "S", "all"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mem-budget-gib", type=float, default=8.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cell-json", default="")
    parser.add_argument("--result-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        cell = Cell(**json.loads(args.cell_json))
        record = run_worker(cell, args.repeats)
        args.result_path.write_text(json.dumps(record), encoding="utf-8")
        return 0
    print(
        "VANILLA_FORWARD_NODE scaling benchmark (timings informational, not asserted)"
    )
    correctness_gate()
    repeats = 1 if args.quick else args.repeats
    rows = []
    header = (
        f"  {'path':8s} {'sweep':5s} {'M':>7s} {'S':>6s} {'K':>3s} {'build s':>9s} "
        f"{'run s':>9s} {'total s':>9s} {'peak MiB':>9s}  status"
    )
    print(header)
    for cell in cells(args):
        estimate_gib = (
            cell.num_positions * cell.num_scenarios * BYTES_PER_CELL_ESTIMATE / 2**30
        )
        if estimate_gib > args.mem_budget_gib:
            record: dict[str, Any] = {
                "status": f"SKIP (estimated {estimate_gib:.1f} GiB > budget)"
            }
        else:
            record = run_cell_subprocess(cell, repeats)
        rows.append({**dataclasses.asdict(cell), **record})
        if record.get("status") == "ok":
            print(
                f"  {cell.path:8s} {cell.sweep:5s} {cell.num_positions:7d} "
                f"{cell.num_scenarios:6d} {record['num_factors']:3d} "
                f"{record['build_s']:9.3f} {record['run_s']:9.3f} {record['total_s']:9.3f} "
                f"{_mib(record['peak_rss_mib'])}  ok"
            )
        else:
            print(
                f"  {cell.path:8s} {cell.sweep:5s} {cell.num_positions:7d} "
                f"{cell.num_scenarios:6d}   -  {record.get('status')}"
            )
            if record.get("status") == "fail":
                return 1
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps({"cells": rows}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
