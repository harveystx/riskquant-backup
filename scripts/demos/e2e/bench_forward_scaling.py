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
import importlib
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

ch11 = importlib.import_module("11_forward_node_taylor_derived")

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
    rsc = fw.static_config()
    world = fw.build_world(fw.MAIN_BOOK, reporting_ccy="EUR", rsc=rsc)
    v0 = fw.base_value_by_uid(world)
    for row in fw.MAIN_BOOK:
        hand = fw.hand_value(row, fw.LEVELS)
        scale = float(
            ch11._leg_scale(row, {n: np.array([fw.LEVELS[n]]) for n in fw.LEVELS})[0]
        )
        if abs(v0[row["uid"]] - hand) > 2.0**-48 * scale:
            raise BenchCheckError(
                f"FULL_REVAL gate: V0 of uid {row['uid']} != hand formula"
            )
    print(
        "  correctness gate FULL_REVAL      : V0 == hand formula for every row   PASS"
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    sens = ch11.run_sensitivities(
        portfolio=world.context.portfolio,
        uix_mapper=world.context.uix_mapper,
        market_data=ch11._market_at(
            world, {n: np.array([world.level(n)]) for n in world.names}
        ),
    )
    block = ch11.certified_block(world, sens, v0)
    hist, ids = world.data.values, list(world.data.factor_ids)
    ret = np.log(hist[1:] / hist[:-1])
    shocked = {
        names[int(f)]: world.level(names[int(f)]) * np.exp(ret[:, ids.index(f)])
        for f in world.axis.factor_ids
    }
    full = {u: v - v0[u] for u, v in fw.value_by_uid(world, shocked).items()}
    prefix = [names[int(f)] for f in world.valuation_factor_ids]
    result = block.calculator.run(
        block.calculator.axis.levels(
            base_levels=np.array([world.level(n) for n in prefix]),
            shocked_levels=np.ascontiguousarray(np.stack([shocked[n] for n in prefix])),
        )
    )
    uids = result.pnl_matrix.trade_uids.tolist()
    for row in fw.MAIN_BOOK:
        t = np.asarray(result.pnl_matrix.pnl, dtype=np.float64)[uids.index(row["uid"])]
        tol = (
            8.0 * ch11.U32 * (np.abs(full[row["uid"]]) + ch11._leg_scale(row, shocked))
        )
        if not np.all(np.abs(full[row["uid"]] - t) <= tol):
            raise BenchCheckError(
                f"TAYLOR gate: uid {row['uid']} Taylor P&L != FULL P&L"
            )
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
    sens = ch11.run_sensitivities(
        portfolio=world.context.portfolio,
        uix_mapper=world.context.uix_mapper,
        market_data=ch11._market_at(
            world, {n: np.array([world.level(n)]) for n in world.names}
        ),
    )
    return world, ch11.certified_block(world, sens, v0)


def execute(cell: Cell, world: fw.ForwardWorld, block: Any) -> Any:
    uids = world.uids.tolist()
    if cell.path == "full":
        return ch11.run(world, full=uids)
    return ch11.run(world, derived=uids, block=block)


def check_run(result: Any, num_positions: int) -> None:
    if result.var.shape != (num_positions + 1,) or not (
        np.isfinite(result.var).all() and np.isfinite(result.es).all()
    ):
        raise BenchCheckError("run output malformed (shape or non-finite VaR / ES)")


def peak_rss_mib() -> float:
    if sys.platform == "win32":  # pragma: no cover - same API as bench_spot_scaling
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
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

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        ctypes.WinDLL("psapi").GetProcessMemoryInfo(
            ctypes.WinDLL("kernel32").GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return float(counters.PeakWorkingSetSize) / 2**20
    import resource

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
                f"{record['peak_rss_mib']:9.1f}  ok"
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
