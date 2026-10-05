# scripts/demos/e2e/bench_spot_scaling.py
"""
RiskQuant SPOT -- benchmark: wall time of a full run versus book size and
scenario count. FULL_REVAL only.

Not a regression gate. The timings are informational and never asserted
(like chapter 06). Correctness checks run inside every cell: a failed check
aborts the benchmark loudly, because a fast wrong run is not a result.

What one cell measures
----------------------
One call of `riskquant.solutions.risk_run.api.run_scenario_var_es` on a book
of M SPOT positions, ALL declared FULL_REVAL (`taylor_derived=None`), with S
scenarios. The timer covers the whole call: identity, base valuation,
scenario generation, shocked levels, FULL_REVAL valuation, the float32 P&L
grid, the D081 reporting conversion and VaR/ES. Fixture build (ingest,
compile, V0) is timed separately as `build s` and is NOT part of `run s`.

Stage split
-----------
The worker wraps three functions of the run module and sums their wall time:
base and scenario valuation (`_value_positions`), the D081 reporting
conversion and VaR/ES (`compute_var_es`). The remainder is "other":
identity, scenario generation, shocked levels and the grid scatter. This is
the technique of `scripts/release/spot/certify_spot_production_scale.py`. If
a wrapped name no longer exists, the split is reported as unavailable and
the benchmark still runs.

Sweeps (RAM grid, the default)
------------------------------
    A    historical, S = 520 (T = 521 observations), M = 100 .. 250,000
    B    SBB, S = 100,000 paths,                     M = 10 .. 2,000
    B2   SBB, S = 10,000 paths,                      M = 100 .. 20,000
    C    SBB, M = 50,000,                            S = 500 .. 4,000

`--requested-grid` selects the original request (M up to 1,000,000; S up
to 1,000,000). Cells that cannot fit the memory budget are reported as
SKIP with the memory estimate and a LABELLED linear extrapolation. They are
never run.

Why the grids stop where they do
--------------------------------
`run_scenario_var_es` materialises the whole (M, S) grid. Peak memory is
about 50 bytes per (position, scenario) cell, so M x S is bounded by the RAM
of the machine. Lifting this bound needs a position-chunked run path; see
`notes/HANDOFF_position_chunked_run.md`.

Process model
-------------
The parent process runs one subprocess per cell. This gives a clean
process-lifetime peak RSS per cell, frees multi-GiB arrays between cells and
contains an out-of-memory failure. The worker warms up on a small world
first (Numba kernel load), then builds the fixture, then times the run.

Run:
    python scripts/demos/e2e/bench_spot_scaling.py
    python scripts/demos/e2e/bench_spot_scaling.py --quick
    python scripts/demos/e2e/bench_spot_scaling.py --sweep A --repeats 5
    python scripts/demos/e2e/bench_spot_scaling.py --json-out bench.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import _spot_world as sw
import numpy as np

import riskquant.solutions.risk_run.api as risk_run_api
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RiskRun,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import build_scenario_generator
from riskquant.solutions.scenario_models.schemas import (
    HistoricalGeneratorConfig,
    SbbGeneratorConfig,
    ScenarioModelConfig,
)

# =============================================================================
# 1. CONSTANTS
# =============================================================================

CONFIDENCE: float = 0.99
SEED: int = 11
HORIZON: int = 1
SBB_BLOCK_SIZE: float = 5.0
#: Observations behind every SBB world (same as chapter 06).
SBB_OBSERVATIONS: int = 1_001
MAX_BATCH_SIZE: int = 1_024

#: Memory model. Derived by reading `run_scenario_var_es`, not measured.
#: Live at the peak of the D081 conversion, per (position, scenario) cell:
#: float32 grid 4 B, float64 settlement P&L 8 B, and inside the conversion
#: rates 8 B, translation 8 B, one product temporary 8 B, converted 8 B,
#: plus the FX gather. The parent replaces this constant with a measured
#: value as soon as a cell of at least `CALIBRATION_MIN_CELLS` has run.
BYTES_PER_CELL: float = 52.0
#: Hard floor under any calibrated value: grid 4 B + settlement P&L 8 B +
#: `groups` copy 8 B + `compute_var_es` scratchpad 8 B are all live when
#: VaR/ES runs. A measurement below this is a measurement error.
MIN_BYTES_PER_CELL: float = 28.0
#: FULL_REVAL batch temporaries per (position, batch column): `rows`,
#: `rows[row_order]`, the `reduceat` output, `vs[full_positions]` and the
#: `vs - v0` difference, 8 B each.
TEMP_BYTES_PER_POSITION_COLUMN: int = 40
#: Parent process (imports) plus worker baseline (imports, history, world).
FIXED_GIB: float = 2.0
#: Frame, ingest batch, compiled blocks, uid maps, per position.
PER_POSITION_BYTES: int = 2_048
CALIBRATION_MIN_CELLS: int = 10_000_000
CALIBRATION_MIN_GROWTH_MIB: float = 64.0
CALIBRATION_SAFETY: float = 1.15

#: Warm-up world: small, same code path, never timed.
WARMUP_POSITIONS: int = 64
WARMUP_HISTORY_OBSERVATIONS: int = 201
WARMUP_SBB_SCENARIOS: int = 256


class BenchCheckError(AssertionError):
    """A correctness check of one benchmark cell failed."""


# =============================================================================
# 2. SWEEP DEFINITIONS
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class Sweep:
    """One scaling experiment: one dimension varies, the other is held."""

    name: str
    title: str
    model: str  # "historical" | "sbb"
    axis: str  # "positions" | "scenarios": the varying dimension
    fixed: int  # the held dimension (S for axis="positions", M otherwise)
    values: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    """One timed run."""

    sweep: str
    model: str
    num_positions: int
    num_scenarios: int

    @property
    def cells(self) -> int:
        return self.num_positions * self.num_scenarios


RAM_SWEEPS: tuple[Sweep, ...] = (
    Sweep(
        name="A",
        title="historical, S = 520, vary positions",
        model="historical",
        axis="positions",
        fixed=520,
        values=(100, 1_000, 10_000, 100_000, 250_000),
    ),
    Sweep(
        name="B",
        title="SBB, S = 100,000 paths, vary positions",
        model="sbb",
        axis="positions",
        fixed=100_000,
        values=(10, 100, 500, 1_000, 2_000),
    ),
    Sweep(
        name="B2",
        title="SBB, S = 10,000 paths, vary positions",
        model="sbb",
        axis="positions",
        fixed=10_000,
        values=(100, 1_000, 10_000, 20_000),
    ),
    Sweep(
        name="C",
        title="SBB, M = 50,000 positions, vary scenarios",
        model="sbb",
        axis="scenarios",
        fixed=50_000,
        values=(500, 1_000, 2_000, 4_000),
    ),
)

REQUESTED_SWEEPS: tuple[Sweep, ...] = (
    Sweep(
        name="A",
        title="historical, S = 520, vary positions (requested grid)",
        model="historical",
        axis="positions",
        fixed=520,
        values=(100, 1_000, 10_000, 100_000, 1_000_000),
    ),
    Sweep(
        name="B",
        title="SBB, S = 100,000 paths, vary positions (requested grid)",
        model="sbb",
        axis="positions",
        fixed=100_000,
        values=(100, 1_000, 10_000, 100_000, 1_000_000),
    ),
    Sweep(
        name="C",
        title="SBB, M = 50,000 positions, vary scenarios (requested grid)",
        model="sbb",
        axis="scenarios",
        fixed=50_000,
        values=(500, 1_000, 10_000, 100_000, 1_000_000),
    ),
)

QUICK_VALUES: dict[str, tuple[int, ...]] = {
    "A": (100, 1_000, 10_000),
    "B": (10, 100),
    "B2": (100, 1_000),
    "C": (500, 1_000),
}


def expand(sweep: Sweep) -> list[Cell]:
    """Cells of one sweep in ascending order of the varying dimension."""
    if sweep.axis == "positions":
        return [
            Cell(
                sweep=sweep.name,
                model=sweep.model,
                num_positions=v,
                num_scenarios=sweep.fixed,
            )
            for v in sweep.values
        ]
    return [
        Cell(
            sweep=sweep.name,
            model=sweep.model,
            num_positions=sweep.fixed,
            num_scenarios=v,
        )
        for v in sweep.values
    ]


# =============================================================================
# 3. MEMORY MODEL AND BATCH POLICY
# =============================================================================


def auto_batch_size(
    num_positions: int, num_scenarios: int, temp_budget_mib: float
) -> int:
    """
    Largest batch whose FULL_REVAL temporaries fit `temp_budget_mib`.

    The batch size is operational only (D041): it changes time, never the
    numbers. It is printed for every cell because it can bias the scaling.
    """
    by_memory = int(
        temp_budget_mib * 2**20 // (TEMP_BYTES_PER_POSITION_COLUMN * num_positions)
    )
    return max(1, min(MAX_BATCH_SIZE, by_memory, num_scenarios))


def estimate_gib(cell: Cell, bytes_per_cell: float) -> float:
    """Estimated process-lifetime peak memory of one cell (parent + worker)."""
    total = (
        FIXED_GIB * 2**30
        + PER_POSITION_BYTES * cell.num_positions
        + bytes_per_cell * cell.cells
    )
    return total / 2**30


def observed_bytes_per_cell(result: dict[str, Any]) -> float | None:
    """Measured peak growth per cell, or None when the cell is too small."""
    cells = int(result["num_positions"]) * int(result["num_scenarios"])
    if cells < CALIBRATION_MIN_CELLS:
        return None
    growth_mib = float(result["peak_rss_mib"]) - float(
        result["peak_rss_after_build_mib"]
    )
    if growth_mib < CALIBRATION_MIN_GROWTH_MIB:
        return None
    return growth_mib * 2**20 / cells


def repeats_for(requested: int, cells: int) -> int:
    """Fewer repeats for the largest cells: they are slow and heavy."""
    if cells > 100_000_000:
        return 1
    if cells > 50_000_000:
        return min(requested, 2)
    return requested


# =============================================================================
# 4. PEAK RESIDENT MEMORY (process lifetime)
# =============================================================================

if sys.platform == "win32":
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


def peak_rss_mib() -> float:
    """
    Process-lifetime peak resident memory in MiB.

    Windows: `PeakWorkingSetSize`. Linux: `ru_maxrss` in KiB. macOS: bytes.
    The peak never decreases, which is why each cell runs in its own process.
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
    import resource

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024.0


# =============================================================================
# 5. WORKER: ONE CELL IN ONE PROCESS
# =============================================================================


def scenario_config(model: str, num_scenarios: int) -> ScenarioModelConfig:
    """The scenario model of one cell (horizon 1)."""
    if model == "historical":
        return HistoricalGeneratorConfig(model="historical", horizon=HORIZON)
    if model == "sbb":
        return SbbGeneratorConfig(
            model="sbb",
            horizon=HORIZON,
            total_scenarios=num_scenarios,
            expected_block_size=SBB_BLOCK_SIZE,
            seed=SEED,
        )
    raise ValueError(f"unknown scenario model {model!r}")


def history_observations(model: str, num_scenarios: int) -> int:
    """Historical simulation yields T - horizon scenarios (one per window)."""
    return num_scenarios + HORIZON if model == "historical" else SBB_OBSERVATIONS


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInputs:
    """Inputs that are not part of the timed call, built once per cell."""

    methodology: MethodologyDeclaration
    context: ValuationContext
    reporting: ReportingSpec


def prepare(feed: sw.RealisticFixture) -> RunInputs:
    """Declare every position FULL_REVAL. Kept outside the timer."""
    world = feed.world
    return RunInputs(
        methodology=sw.methodology(full=world.uids.tolist()),
        context=sw.valuation_context(world),
        reporting=sw.reporting_spec(world, reporting_ccy="USD"),
    )


def execute(
    feed: sw.RealisticFixture,
    inputs: RunInputs,
    config: ScenarioModelConfig,
    *,
    batch_size: int,
) -> RiskRun:
    """One production call: every position FULL_REVAL, no Taylor block."""
    world = feed.world
    return run_scenario_var_es(
        scenario_config=config,
        data=feed.history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        proxy_terms=world.proxy_terms,
        base_levels=world.base_levels,
        methodology=inputs.methodology,
        valuation_context=inputs.context,
        taylor_derived=None,
        reporting=inputs.reporting,
        confidence_level=CONFIDENCE,
        batch_size=batch_size,
    )


def check_run(run: RiskRun, *, num_positions: int) -> None:
    """Cheap structural checks. A failure aborts the benchmark (FAIL LOUD)."""
    if run.var.shape != (num_positions + 1,) or run.es.shape != run.var.shape:
        raise BenchCheckError(
            f"expected {num_positions + 1} VaR/ES values, got {run.var.shape}."
        )
    if not (bool(np.isfinite(run.var).all()) and bool(np.isfinite(run.es).all())):
        raise BenchCheckError("VaR/ES contains a non-finite value.")
    if not bool((run.es <= run.var).all()):
        raise BenchCheckError("ES is less severe than VaR for at least one row.")
    if not run.reproducible:
        raise BenchCheckError("the run is not marked reproducible.")


def warm_up(model: str) -> None:
    """Load every Numba kernel on a small world. Never timed."""
    if model == "historical":
        observations, scenarios = WARMUP_HISTORY_OBSERVATIONS, 0
    else:
        observations, scenarios = SBB_OBSERVATIONS, WARMUP_SBB_SCENARIOS
    feed = sw.realistic_fixture(
        num_positions=WARMUP_POSITIONS, num_observations=observations
    )
    config = scenario_config(model, scenarios)
    check_run(
        execute(feed, prepare(feed), config, batch_size=WARMUP_POSITIONS),
        num_positions=WARMUP_POSITIONS,
    )


class StageTimer:
    """
    Sum the wall time of three run-module functions inside one `with` block.

    Wrapping is by module attribute, which is how `run_scenario_var_es`
    resolves these names at call time. A missing attribute is skipped, never
    an error: the stage split is a diagnostic, the total is the benchmark.
    """

    TARGETS: ClassVar[dict[str, str]] = {
        "valuation": "_value_positions",
        "d081": "convert_scenario_pnl_to_reporting_currency",
        "var_es": "compute_var_es",
    }

    def __init__(self) -> None:
        self.seconds: dict[str, float] = dict.fromkeys(self.TARGETS, 0.0)
        self.available: list[str] = []
        self._real: dict[str, Any] = {}

    def _wrap(self, key: str, fn: Any) -> Any:
        def timed(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds[key] += time.perf_counter() - t0

        return timed

    def __enter__(self) -> StageTimer:
        for key, attr in self.TARGETS.items():
            fn = getattr(risk_run_api, attr, None)
            if fn is None:
                continue
            self._real[attr] = fn
            setattr(risk_run_api, attr, self._wrap(key, fn))
            self.available.append(key)
        return self

    def __exit__(self, *exc: object) -> None:
        for attr, fn in self._real.items():
            setattr(risk_run_api, attr, fn)

    def snapshot(self) -> dict[str, float]:
        return {key: self.seconds[key] for key in self.available}


def run_worker(cell: Cell, *, repeats: int, batch_size: int) -> dict[str, Any]:
    """Time one cell. Called in a fresh process."""
    warm_up(cell.model)

    t_build = time.perf_counter()
    feed = sw.realistic_fixture(
        num_positions=1,  # replaced by `positions`; history is independent of it
        num_observations=history_observations(cell.model, cell.num_scenarios),
        positions=sw.realistic_positions(num_positions=cell.num_positions),
    )
    config = scenario_config(cell.model, cell.num_scenarios)
    declared = int(build_scenario_generator(config, data=feed.history).total_scenarios)
    if declared != cell.num_scenarios:
        raise BenchCheckError(
            f"the model yields {declared} scenarios, the cell asks for "
            f"{cell.num_scenarios}."
        )
    inputs = prepare(feed)
    build_s = time.perf_counter() - t_build
    peak_after_build = peak_rss_mib()

    seconds: list[float] = []
    stage_runs: list[dict[str, float]] = []
    first: RiskRun | None = None
    try:
        for _ in range(repeats):
            with StageTimer() as timer:
                t_run = time.perf_counter()
                run = execute(feed, inputs, config, batch_size=batch_size)
                seconds.append(time.perf_counter() - t_run)
            stage_runs.append(timer.snapshot())
            check_run(run, num_positions=cell.num_positions)
            if first is None:
                first = run
            elif not (
                run.identity == first.identity
                and np.array_equal(run.var, first.var)
                and np.array_equal(run.es, first.es)
            ):
                raise BenchCheckError("a repeated run differs from the first run.")
    except MemoryError:
        return {
            **dataclasses.asdict(cell),
            "status": "oom",
            "batch_size": batch_size,
            "peak_rss_after_build_mib": peak_after_build,
            "peak_rss_mib": peak_rss_mib(),
        }
    assert first is not None
    return {
        **dataclasses.asdict(cell),
        "status": "ok",
        "batch_size": batch_size,
        "repeats": repeats,
        "build_s": build_s,
        "run_s": seconds,
        "stages_s": stage_runs[seconds.index(min(seconds))],
        "peak_rss_after_build_mib": peak_after_build,
        "peak_rss_mib": peak_rss_mib(),
        "identity": first.identity,
        "portfolio_var_hex": float(first.portfolio_var).hex(),
        "environment": environment(),
    }


def environment() -> dict[str, Any]:
    """Facts needed to compare two benchmark files."""
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threads": int(numba.get_num_threads()),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


# =============================================================================
# 6. PARENT: ORCHESTRATION
# =============================================================================


@dataclass(slots=True, kw_only=True)
class Row:
    """One line of the result table."""

    cell: Cell
    batch_size: int
    status: str  # "ok" | "SKIP" | "OOM" | "FAIL"
    estimate_gib: float
    build_s: float | None = None
    run_min_s: float | None = None
    run_median_s: float | None = None
    peak_gib: float | None = None
    extrapolated_s: float | None = None
    stages_s: dict[str, float] | None = None
    detail: str = ""

    @property
    def ns_per_cell(self) -> float | None:
        if self.run_min_s is None:
            return None
        return self.run_min_s * 1e9 / self.cell.cells


def run_cell_subprocess(cell: Cell, *, repeats: int, batch_size: int) -> dict[str, Any]:
    """Run one cell in a fresh interpreter and return its result record."""
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "cell.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--cell-json",
            json.dumps(dataclasses.asdict(cell)),
            "--repeats",
            str(repeats),
            "--batch-size",
            str(batch_size),
            "--result-path",
            str(result_path),
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.run(command, env=env, check=False)
        if result_path.is_file():
            record: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            return record
    return {"status": "fail", "returncode": proc.returncode}


def fmt_seconds(value: float | None, *, approx: bool = False) -> str:
    if value is None:
        return "-"
    text = f"{value:,.2f}" if value < 1_000 else f"{value:,.0f}"
    return f"~{text}" if approx else text


def run_sweep(
    sweep: Sweep,
    *,
    mem_budget_gib: float,
    temp_budget_mib: float,
    fixed_batch: int,
    repeats: int,
    bytes_per_cell: float,
) -> tuple[list[Row], float, list[dict[str, Any]]]:
    """Run the cells of one sweep in ascending order, guarding memory."""
    rows: list[Row] = []
    records: list[dict[str, Any]] = []
    blocked = False
    for cell in expand(sweep):
        batch = fixed_batch or auto_batch_size(
            cell.num_positions, cell.num_scenarios, temp_budget_mib
        )
        estimate = estimate_gib(cell, bytes_per_cell)
        if blocked or estimate > mem_budget_gib:
            rows.append(
                Row(cell=cell, batch_size=batch, status="SKIP", estimate_gib=estimate)
            )
            print(
                f"  [{sweep.name}] M={cell.num_positions:,} S={cell.num_scenarios:,}"
                f" SKIP (estimate {estimate:.1f} GiB, budget {mem_budget_gib:.1f})",
                flush=True,
            )
            continue
        n_repeats = repeats_for(repeats, cell.cells)
        print(
            f"  [{sweep.name}] M={cell.num_positions:,} S={cell.num_scenarios:,}"
            f" batch={batch:,} repeats={n_repeats} estimate={estimate:.1f} GiB",
            flush=True,
        )
        record = run_cell_subprocess(cell, repeats=n_repeats, batch_size=batch)
        records.append(record)
        status = str(record["status"])
        if status == "ok":
            seconds = [float(s) for s in record["run_s"]]
            rows.append(
                Row(
                    cell=cell,
                    batch_size=batch,
                    status="ok",
                    estimate_gib=estimate,
                    build_s=float(record["build_s"]),
                    run_min_s=min(seconds),
                    run_median_s=statistics.median(seconds),
                    peak_gib=float(record["peak_rss_mib"]) / 1024.0,
                    stages_s={k: float(v) for k, v in record["stages_s"].items()},
                )
            )
            observed = observed_bytes_per_cell(record)
            if observed is not None:
                bytes_per_cell = max(MIN_BYTES_PER_CELL, observed * CALIBRATION_SAFETY)
        else:
            rows.append(
                Row(
                    cell=cell,
                    batch_size=batch,
                    status="OOM" if status == "oom" else "FAIL",
                    estimate_gib=estimate,
                    detail=f"returncode={record.get('returncode', '-')}",
                )
            )
            blocked = True  # larger cells of this sweep cannot succeed
    add_extrapolation(rows)
    return rows, bytes_per_cell, records


def add_extrapolation(rows: list[Row]) -> None:
    """Linear projection of SKIP cells from the largest measured cell."""
    measured = [r for r in rows if r.status == "ok" and r.ns_per_cell is not None]
    if not measured:
        return
    reference = max(measured, key=lambda r: r.cell.cells)
    ns = reference.ns_per_cell
    assert ns is not None
    for row in rows:
        if row.status == "SKIP":
            row.extrapolated_s = ns * row.cell.cells / 1e9


def exponents(sweep: Sweep, rows: list[Row]) -> tuple[list[float], float | None]:
    """Local log-log slopes between measured cells and one least-squares fit."""
    points: list[tuple[float, float]] = []
    for r in rows:
        if r.status != "ok" or r.run_min_s is None or r.run_min_s <= 0.0:
            continue
        x = r.cell.num_positions if sweep.axis == "positions" else r.cell.num_scenarios
        points.append((float(x), float(r.run_min_s)))
    local = [
        math.log(y1 / y0) / math.log(x1 / x0)
        for (x0, y0), (x1, y1) in itertools.pairwise(points)
    ]
    if len(points) < 2:
        return local, None
    lx = np.log([p[0] for p in points])
    ly = np.log([p[1] for p in points])
    return local, float(np.polyfit(lx, ly, 1)[0])


def print_sweep_table(sweep: Sweep, rows: list[Row]) -> None:
    table: list[list[str]] = []
    for r in rows:
        skipped = r.status == "SKIP"
        table.append([
            sweep.name,
            f"{r.cell.num_positions:,}",
            f"{r.cell.num_scenarios:,}",
            f"{r.batch_size:,}",
            fmt_seconds(r.build_s),
            fmt_seconds(r.extrapolated_s if skipped else r.run_min_s, approx=skipped),
            "-" if r.ns_per_cell is None else f"{r.ns_per_cell:,.0f}",
            f"~{r.estimate_gib:.1f}" if r.peak_gib is None else f"{r.peak_gib:.1f}",
            r.status,
        ])
    sw.table(
        ["sw", "M", "S", "B", "build s", "run s", "ns/cell", "GiB", "status"],
        table,
        align="lrrrrrrrl",
    )


def print_stage_table(sweep: Sweep, rows: list[Row]) -> None:
    """Stage split of the fastest repeat of every measured cell."""
    table: list[list[str]] = []
    for r in rows:
        if r.status != "ok" or r.run_min_s is None:
            continue
        stages = r.stages_s or {}
        known = sum(stages.values())
        table.append([
            sweep.name,
            f"{r.cell.num_positions:,}",
            f"{r.cell.num_scenarios:,}",
            fmt_seconds(stages["valuation"]) if "valuation" in stages else "n/a",
            fmt_seconds(stages["d081"]) if "d081" in stages else "n/a",
            fmt_seconds(stages["var_es"]) if "var_es" in stages else "n/a",
            fmt_seconds(max(0.0, r.run_min_s - known)) if stages else "n/a",
            fmt_seconds(r.run_min_s),
        ])
    if table:
        sw.table(
            ["sw", "M", "S", "valuation", "d081", "var_es", "other", "total"],
            table,
            align="lrrrrrrr",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", default="all", choices=("A", "B", "B2", "C", "all"))
    parser.add_argument("--requested-grid", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mem-budget-gib", type=float, default=14.0)
    parser.add_argument("--temp-budget-mib", type=float, default=512.0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    # internal: one cell in one process
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cell-json", default="")
    parser.add_argument("--result-path", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.worker:
        cell = Cell(**json.loads(args.cell_json))
        result = run_worker(cell, repeats=args.repeats, batch_size=args.batch_size)
        assert args.result_path is not None
        args.result_path.write_text(json.dumps(result), encoding="utf-8")
        return

    sweeps = REQUESTED_SWEEPS if args.requested_grid else RAM_SWEEPS
    if args.sweep != "all":
        sweeps = tuple(s for s in sweeps if s.name == args.sweep)
        if not sweeps:
            raise SystemExit(f"sweep {args.sweep!r} is not in the selected grid.")
    if args.quick:
        sweeps = tuple(
            dataclasses.replace(s, values=QUICK_VALUES[s.name]) for s in sweeps
        )
    repeats = min(args.repeats, 2) if args.quick else args.repeats

    sw.title("RiskQuant SPOT benchmark - full-run scaling, FULL_REVAL only")
    sw.step("1", "Plan")
    sw.kv([
        ("grid", "requested" if args.requested_grid else "RAM"),
        ("memory budget", f"{args.mem_budget_gib:.1f} GiB (parent + worker)"),
        ("temp budget", f"{args.temp_budget_mib:.0f} MiB per batch"),
        ("batch size", args.batch_size or "auto (memory bound, max 1,024)"),
        ("repeats", f"{repeats} (1 or 2 for the largest cells)"),
        ("cells", sum(len(expand(s)) for s in sweeps)),
        ("bytes per cell", f"{BYTES_PER_CELL:.0f} (static, calibrated at run)"),
    ])

    sw.step("2", "Run (one subprocess per cell)")
    bytes_per_cell = BYTES_PER_CELL
    results: list[tuple[Sweep, list[Row]]] = []
    all_records: list[dict[str, Any]] = []
    for sweep in sweeps:
        print(f"\nSweep {sweep.name}: {sweep.title}", flush=True)
        rows, bytes_per_cell, records = run_sweep(
            sweep,
            mem_budget_gib=args.mem_budget_gib,
            temp_budget_mib=args.temp_budget_mib,
            fixed_batch=args.batch_size,
            repeats=repeats,
            bytes_per_cell=bytes_per_cell,
        )
        results.append((sweep, rows))
        all_records.extend(records)

    sw.step("3", "Results")
    for sweep, rows in results:
        sw.para(f"Sweep {sweep.name}: {sweep.title}.")
        print_sweep_table(sweep, rows)
    sw.para("Stage split (seconds) of the fastest repeat of each measured cell.")
    for sweep, rows in results:
        print_stage_table(sweep, rows)
    sw.para(
        """
        run s is the minimum over the repeats of one run_scenario_var_es call.
        build s is fixture build (ingest, compile, V0) and is not in run s.
        GiB is the measured process-lifetime peak; a leading ~ marks an
        estimate. A run s with a leading ~ on a SKIP row is a linear
        projection from the largest measured cell of the sweep. It is not a
        measurement. B is the batch size; it is operational (D041), but it
        changes time, so read the scaling together with the B column.

        Status ok: measured. SKIP: not run, because the memory estimate is
        above the budget or a smaller cell of the sweep failed. OOM: the
        worker raised MemoryError. FAIL: the worker exited without a result;
        read the traceback printed in step 2.
        """
    )

    sw.step("4", "Scaling exponents (run s versus the varying dimension)")
    exponent_rows: list[list[str]] = []
    for sweep, rows in results:
        local, fit = exponents(sweep, rows)
        exponent_rows.append([
            sweep.name,
            sweep.axis,
            ", ".join(f"{e:.2f}" for e in local) or "-",
            "-" if fit is None else f"{fit:.2f}",
        ])
    sw.table(
        ["sw", "varies", "local exponents", "fit"],
        exponent_rows,
        align="llll",
    )
    sw.para(
        """
        An exponent of 1.00 means time grows in proportion to the varying
        dimension. Fixed costs lower the exponent at the small end.
        """
    )
    sw.kv([
        ("measured bytes per cell", f"{bytes_per_cell:.1f}"),
        ("static model", f"{BYTES_PER_CELL:.1f}"),
    ])

    if args.json_out is not None:
        payload = {
            "grid": "requested" if args.requested_grid else "ram",
            "mem_budget_gib": args.mem_budget_gib,
            "records": all_records,
            "rows": [
                {
                    **dataclasses.asdict(r.cell),
                    "batch_size": r.batch_size,
                    "status": r.status,
                    "estimate_gib": r.estimate_gib,
                    "build_s": r.build_s,
                    "run_min_s": r.run_min_s,
                    "run_median_s": r.run_median_s,
                    "peak_gib": r.peak_gib,
                    "extrapolated_s": r.extrapolated_s,
                    "stages_s": r.stages_s,
                }
                for _, rows in results
                for r in rows
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    if any(r.status == "FAIL" for _, rows in results for r in rows):
        print("\nBENCHMARK FAILED: at least one cell exited without a result.")
        sys.exit(1)


if __name__ == "__main__":
    main()
