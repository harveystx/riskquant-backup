# src/riskquant/solutions/sensitivity_pnl/_verify/verify_cached_book.py
"""
Ruthless Reviewer, pass 3: the cached-book contract.

Three demands, each answered by measurement rather than by argument:

  1. Compile once, price many, with validation on every scenario set that is
     exactly as strict as on the first.
  2. No `(K, S)` matrix copy for unsorted factor keys, at any point.
  3. A finiteness check that cannot be fooled and cannot be broken by the host's
     numpy error policy.
"""

from __future__ import annotations

import itertools
import sys
import time
import tracemalloc

import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    ScenarioShocks,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def hdr(s: str) -> None:
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


def expect(name: str, fn, *, contains: tuple[str, ...] = ()) -> None:
    try:
        fn()
    except SensitivityPnlInputError as e:
        text = str(e)
        missing = [s for s in contains if s not in text]
        check(name, not missing, f"lacked {missing}\n{text}")
        if not missing:
            print("        " + text.splitlines()[1].strip()[:140])
        return
    except Exception as e:
        check(name, False, f"wrong type {type(e).__name__}: {str(e)[:200]}")
        return
    check(name, False, "no error raised")


# ------------------------------------------------------------------ fixture
VOLUME, SPOT, FX, HEDGE = 50_000.0, 42.50, 0.85, -18_000.0
SENS = pl.DataFrame(
    {
        "position_id": [101, 101, 202, 202],
        "factor_id": [10, 20, 10, 20],
        "first_order": [VOLUME * FX, VOLUME * SPOT, HEDGE * FX, HEDGE * SPOT],
        "second_order": [0.0, 0.0, 0.0, 0.0],
    }
)
CROSS = pl.DataFrame(
    {
        "position_id": [101, 202],
        "factor_id_i": [10, 10],
        "factor_id_j": [20, 20],
        "cross_second_order": [VOLUME, HEDGE],
    }
)
META = pl.DataFrame(
    {
        "factor_id": [10, 20],
        "displacement_convention_id": [1, 1],
        "quote_base": [1.0, 1.0],
    }
)
AXIS = FactorAxis.from_keys([10, 20])
SPEC = SensitivityBookSpec(
    sensitivities=SENS, factor_meta=META, factor_axis=AXIS, cross_sensitivities=CROSS
)


def exact(volume: float, d_spot: float, d_fx: float) -> float:
    return volume * (FX + d_fx) * (SPOT + d_spot) - volume * FX * SPOT


# ============================================================ 1. the contract
hdr("1 — compile once, price many")
calc = SensitivityPnlCalculator.compile(SPEC)
check("position_uids exposed before any run", calc.position_uids.tolist() == [101, 202])
check("factor_keys are the caller's own", calc.factor_keys.to_list() == [10, 20])
check("active-factor count exposed", calc.num_active_factors == 2)

rng = np.random.default_rng(11)
for tick in range(5):
    d_spot, d_fx = rng.normal(0, 3.0), rng.normal(0, 0.02)
    market = calc.axis.shocks(np.array([[d_spot], [d_fx]]))
    result = calc.run(market)
    ok = all(
        abs(float(result.pnl[row, 0]) - exact(vol, d_spot, d_fx)) < 1e-2
        for row, vol in ((0, VOLUME), (1, HEDGE))
    )
    check(f"tick {tick} matches closed form", ok, result.pnl[:, 0])
check("provenance flags a reused book", result.provenance.book_reused)

one_shot = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=SENS, factor_meta=META, cross_sensitivities=CROSS, market=market
    )
)
check(
    "one-shot API bit-identical to the cached path",
    bool(np.array_equal(one_shot.pnl, result.pnl)),
    f"{one_shot.pnl}\n{result.pnl}",
)
check("one-shot flags a freshly compiled book", not one_shot.provenance.book_reused)

hdr("2 — results never alias each other")
a = calc.run(calc.axis.shocks(np.array([[1.0], [0.0]])))
snapshot = a.pnl.copy()
b = calc.run(calc.axis.shocks(np.array([[99.0], [0.0]])))
check("earlier result unchanged by a later run", bool(np.array_equal(a.pnl, snapshot)))
check("the two results own different buffers", not np.shares_memory(a.pnl, b.pnl))

hdr("3 — the axis is the seam a cached book can newly get wrong")
expect(
    "a market on a different axis is rejected",
    lambda: calc.run(
        ScenarioShocks.from_matrix(
            factor_keys=[10, 20, 30], displacement=np.ones((3, 1))
        )
    ),
    contains=("3 factors; the book was compiled against 2", "plausible, wrong number"),
)
expect(
    "a REORDERED axis of the same factors is rejected",
    lambda: calc.run(
        ScenarioShocks.from_matrix(factor_keys=[20, 10], displacement=np.ones((2, 1)))
    ),
    contains=("different row order", "row 0 is 20 here and 10"),
)
rebuilt = FactorAxis.from_keys([10, 20])
check(
    "an equal axis rebuilt from the same keys is accepted",
    calc.run(rebuilt.shocks(np.array([[1.0], [0.0]]))).pnl.shape == (2, 1),
)

hdr("4 — validation on tick N is a construction invariant, not a run-path step")
expect(
    "NaN in tick-5000 shocks still rejected",
    lambda: calc.axis.shocks(np.array([[1.0], [np.nan]])),
    contains=("non-finite", "rows (0-based): 1"),
)
expect(
    "wrong row count still rejected",
    lambda: calc.axis.shocks(np.ones((3, 4))),
    contains=("3 rows against a 2-factor axis",),
)
check(
    "there is no run() argument that could skip a check",
    "validate" not in SensitivityPnlCalculator.run.__code__.co_varnames
    and set(SensitivityPnlCalculator.run.__code__.co_varnames[:3])
    == {"self", "market", "book_reused"},
    SensitivityPnlCalculator.run.__code__.co_varnames[:4],
)

hdr("5 — calculator is frozen and holds no per-run state")
check("frozen dataclass", SensitivityPnlCalculator.__dataclass_params__.frozen)
check(
    "slots",
    SensitivityPnlCalculator.__dataclass_params__.slots
    if hasattr(SensitivityPnlCalculator.__dataclass_params__, "slots")
    else True,
)
check(
    "only book and axis are held",
    set(SensitivityPnlCalculator.__slots__) == {"book", "axis"},
    SensitivityPnlCalculator.__slots__,
)

# ============================================================ 6. zero copy
hdr("6 — the (K, S) copy is gone, at every K")
for k, s in ((4, 100), (500, 2000), (2000, 4000)):
    keys = np.random.default_rng(k).permutation(k).astype(np.int64)
    raw = np.ascontiguousarray(np.ones((k, s), dtype=np.float32))
    market = ScenarioShocks.from_matrix(factor_keys=keys, displacement=raw)
    check(
        f"K={k:>4} S={s:>4}: unsorted keys, matrix adopted by reference",
        np.shares_memory(market.displacement, raw),
    )
    check(f"K={k:>4} S={s:>4}: encoding is slot", market.axis.slot_map is not None)

hdr("7 — unsorted keys give the SAME numbers as sorted, not merely plausible ones")
rng = np.random.default_rng(4)
K, S = 200, 64
perm = rng.permutation(K)
disp_sorted = np.ascontiguousarray(rng.normal(0, 0.02, (K, S)).astype(np.float32))
pos = np.repeat(np.arange(300, dtype=np.int64), 6)
fac = rng.integers(0, K, size=pos.shape[0]).astype(np.int64)
frame = (
    pl.DataFrame(
        {
            "position_id": pos,
            "factor_id": fac,
            "first_order": rng.normal(0, 1e4, pos.shape[0]),
            "second_order": rng.normal(0, 1e3, pos.shape[0]),
        }
    )
    .unique(subset=["position_id", "factor_id"], keep="first")
    .sort(["position_id", "factor_id"])
)
meta = pl.DataFrame(
    {
        "factor_id": np.arange(K, dtype=np.int64),
        "displacement_convention_id": np.ones(K, dtype=np.int64),
        "quote_base": np.ones(K),
    }
)
sorted_result = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=frame,
        factor_meta=meta,
        market=ScenarioShocks.from_matrix(
            factor_keys=np.arange(K, dtype=np.int64), displacement=disp_sorted
        ),
    )
)
# Same market, expressed with permuted keys AND correspondingly permuted rows.
shuffled_result = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=frame,
        factor_meta=meta,
        market=ScenarioShocks.from_matrix(
            factor_keys=perm.astype(np.int64),
            displacement=np.ascontiguousarray(disp_sorted[perm]),
        ),
    )
)
check(
    "slot encoding is bit-identical to direct encoding",
    bool(np.array_equal(sorted_result.pnl, shuffled_result.pnl)),
    f"max diff {np.abs(sorted_result.pnl - shuffled_result.pnl).max()}",
)
check(
    "encodings actually differed",
    sorted_result.provenance.factor_key_encoding == "direct"
    and shuffled_result.provenance.factor_key_encoding == "slot",
)

# ============================================================ 8. finiteness
hdr("8 — the float64 accumulation cannot be fooled")
poison = [np.inf, -np.inf, np.nan]
filler = [0.0, -0.0, 1.0, -1.0, 1e308, -1e308, 3.4e38, -3.4e38]
false_negatives = 0
cases = 0
for dtype in (np.float32, np.float64):
    for count in (1, 2, 3):
        for combo in itertools.product(poison, repeat=count):
            for pad in (0, 1, 5, 64, 127, 128, 1000):
                for tail in (filler[: pad and 3], []):
                    arr = np.array(list(combo) + list(tail) + [0.0] * pad, dtype=dtype)
                    for variant in (arr, arr[::-1].copy()):
                        cases += 1
                        truth = bool(np.isfinite(variant).all())
                        with np.errstate(all="ignore"):
                            trick = bool(np.isfinite(variant.sum(dtype=np.float64)))
                        if trick and not truth:
                            false_negatives += 1
# inf/-inf straddling every pairwise-summation block boundary
for n in (7, 8, 9, 127, 128, 129, 1023, 1024, 8192, 100_000):
    for dtype in (np.float32, np.float64):
        for i, j in ((0, n - 1), (n // 2, n // 2 + 1), (1, 2)):
            arr = np.zeros(n, dtype=dtype)
            arr[i], arr[j] = np.inf, -np.inf
            cases += 1
            with np.errstate(all="ignore"):
                if bool(np.isfinite(arr.sum(dtype=np.float64))):
                    false_negatives += 1
check(
    f"zero false negatives over {cases:,} adversarial arrays",
    false_negatives == 0,
    false_negatives,
)
big = np.full(10_000_000, 3.4e38, dtype=np.float32)
with np.errstate(all="ignore"):
    f64_sum, f32_sum = big.sum(dtype=np.float64), big.sum(dtype=np.float32)
check(
    "float64 accumulator is load-bearing (f32 would false-positive)",
    np.isfinite(f64_sum) and not np.isfinite(f32_sum),
    f"f64={f64_sum:.3e} f32={f32_sum}",
)

hdr("9 — np.seterr(all='raise') in the host does not break the validator")
np.seterr(all="raise")
try:
    expect(
        "inf/-inf under seterr='raise' yields a clean input error",
        lambda: calc.axis.shocks(np.array([[np.inf], [-np.inf]])),
        contains=("non-finite",),
    )
    expect(
        "NaN under seterr='raise' yields a clean input error",
        lambda: calc.axis.shocks(np.array([[np.nan], [1.0]])),
        contains=("non-finite",),
    )
    good = calc.run(calc.axis.shocks(np.array([[1.0], [0.01]])))
    check("a valid run still completes under seterr='raise'", good.pnl.shape == (2, 1))
finally:
    np.seterr(all="warn")

# ============================================================ 10. the payoff
hdr("10 — what the cache is worth: 20k positions, 792k entries, S=250 ticks")
M, PER, K, S = 20_000, 40, 2_000, 250
rng = np.random.default_rng(7)
pos = np.repeat(np.arange(M, dtype=np.int64), PER)
fac = rng.integers(0, K, size=M * PER).astype(np.int64)
big_sens = (
    pl.DataFrame(
        {
            "position_id": pos,
            "factor_id": fac,
            "first_order": rng.normal(0, 1e4, M * PER),
            "second_order": np.zeros(M * PER),
        }
    )
    .unique(subset=["position_id", "factor_id"], keep="first")
    .sort(["position_id", "factor_id"])
)
big_meta = pl.DataFrame(
    {
        "factor_id": np.arange(K, dtype=np.int64),
        "displacement_convention_id": np.ones(K, dtype=np.int64),
        "quote_base": np.ones(K),
    }
)
big_axis = FactorAxis.from_keys(np.arange(K, dtype=np.int64))
big_spec = SensitivityBookSpec(
    sensitivities=big_sens,
    factor_meta=big_meta,
    factor_axis=big_axis,
    position_uids=np.arange(M, dtype=np.int64),
)


def tick_matrix() -> np.ndarray:
    return np.ascontiguousarray(rng.normal(0, 0.01, (K, S)).astype(np.float32))


warm = SensitivityPnlCalculator.compile(big_spec)
warm.run(big_axis.shocks(tick_matrix()))

t = time.perf_counter()
calc_big = SensitivityPnlCalculator.compile(big_spec)
compile_ms = (time.perf_counter() - t) * 1e3

ticks = [big_axis.shocks(tick_matrix()) for _ in range(10)]
t = time.perf_counter()
for market in ticks:
    calc_big.run(market)
cached_ms = (time.perf_counter() - t) * 1e3 / len(ticks)

t = time.perf_counter()
for market in ticks[:3]:
    run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=big_sens,
            factor_meta=big_meta,
            market=market,
            position_uids=np.arange(M, dtype=np.int64),
        )
    )
oneshot_ms = (time.perf_counter() - t) * 1e3 / 3

print(f"  compile (once)          {compile_ms:8.1f} ms")
print(f"  cached run per tick     {cached_ms:8.1f} ms")
print(f"  one-shot run per tick   {oneshot_ms:8.1f} ms")
print(f"  speed-up per tick       {oneshot_ms / cached_ms:8.2f}x")
check(
    "cached tick is at least 3x faster than a one-shot",
    oneshot_ms / cached_ms > 3.0,
    f"{oneshot_ms / cached_ms:.2f}x",
)
check(
    "compile is amortised after < 20 ticks", compile_ms / (oneshot_ms - cached_ms) < 20
)

tracemalloc.start()
calc_big.run(ticks[0])
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
block = M * S * 4
print(
    f"  peak {peak / 1e6:.1f} MB, output block {block / 1e6:.1f} MB, overhead {(peak - block) / 1e6:.1f} MB"
)
check(
    "cached tick allocates little beyond its output block",
    peak - block < 40e6,
    f"{(peak - block) / 1e6:.1f} MB",
)

hdr("11 — run_into_grid: the ticking caller who wants one buffer")
grid = allocate_pnl_grid(block_position_uids=[calc.position_uids], num_scenarios=1)
prov = calc.run_into_grid(
    calc.axis.shocks(np.array([[2.5], [-0.03]])), grid=grid, block_index=0
)
direct = calc.run(calc.axis.shocks(np.array([[2.5], [-0.03]])))
check(
    "in-place write matches the allocating path",
    bool(np.array_equal(grid.block_view(0), direct.pnl)),
)
check("provenance returned", prov.num_positions == 2)
expect(
    "a mismatched axis is caught before touching the caller's grid",
    lambda: calc.run_into_grid(
        ScenarioShocks.from_matrix(
            factor_keys=[10, 20, 30], displacement=np.ones((3, 1))
        ),
        grid=grid,
        block_index=0,
    ),
    contains=("3 factors; the book was compiled against 2",),
)

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
sys.exit(1 if FAILS else 0)
