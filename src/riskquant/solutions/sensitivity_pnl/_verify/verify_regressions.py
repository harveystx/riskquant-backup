# src/riskquant/solutions/sensitivity_pnl/_verify/verify_regressions.py

"""Ruthless Reviewer, pass 2: regressions on pass-1 findings, then harder attacks."""

from __future__ import annotations

import sys
import time
import tracemalloc

import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    ScenarioLevels,
    ScenarioShocks,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    SensitivityPnlResult,
    run_sensitivity_pnl,
    write_sensitivity_pnl,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def hdr(s: str) -> None:
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


def expect(
    name: str, fn, *, contains: tuple[str, ...] = (), exc=SensitivityPnlInputError
):
    try:
        fn()
    except exc as e:
        text = str(e)
        missing = [s for s in contains if s not in text]
        check(name, not missing, f"lacked {missing}\n{text}")
        if not missing and text:
            print(
                "        "
                + text.splitlines()[min(1, len(text.splitlines()) - 1)].strip()[:150]
            )
        return
    except Exception as e:
        check(name, False, f"wrong type {type(e).__name__}: {str(e)[:200]}")
        return
    check(name, False, "no error raised")


# ================================================== REGRESSION: pass-1 finds
hdr(
    "R1 — factor_meta superset under STRING keys (was: raw Polars InvalidOperationError)"
)
sens = pl.DataFrame(
    {
        "position_id": [101],
        "factor_id": ["EQ.ACME"],
        "first_order": [3.0],
        "second_order": [0.0],
    }
)
meta_house = pl.DataFrame(
    {
        "factor_id": [f"IR.NODE.{i:04d}" for i in range(2000)]
        + ["EQ.ACME", "FX.EURGBP"],
        "displacement_convention_id": [1] * 2002,
        "quote_base": [1.0] * 2002,
    }
)
mkt = ScenarioShocks.from_matrix(
    factor_keys=["EQ.ACME"], displacement=np.array([[2.0, -1.0]])
)
res = run_sensitivity_pnl(
    SensitivityPnlRequest(sensitivities=sens, factor_meta=meta_house, market=mkt)
)
check("house-wide factor_meta accepted", res.pnl.tolist() == [[6.0, -3.0]], res.pnl)
check(
    "provenance reports the market size, not the table size",
    res.provenance.num_market_factors == 1,
)

hdr("R2 — factor missing from factor_meta is named as a LABEL, not a slot index")
meta_short = pl.DataFrame(
    {"factor_id": ["EQ.ACME"], "displacement_convention_id": [1], "quote_base": [1.0]}
)
sens2 = pl.DataFrame(
    {
        "position_id": [101, 101],
        "factor_id": ["EQ.ACME", "FX.EURGBP"],
        "first_order": [1.0, 1.0],
        "second_order": [0.0, 0.0],
    }
)
mkt2 = ScenarioShocks.from_matrix(
    factor_keys=["EQ.ACME", "FX.EURGBP"], displacement=np.ones((2, 2))
)
expect(
    "missing metadata names 'FX.EURGBP'",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=sens2, factor_meta=meta_short, market=mkt2)
    ),
    contains=("FX.EURGBP", "no row in factor_meta"),
)

hdr("R3 — bad quote_base names the factor")
meta_bad = pl.DataFrame(
    {
        "factor_id": ["EQ.ACME", "FX.EURGBP"],
        "displacement_convention_id": [1, 1],
        "quote_base": [1.0, 0.0],
    }
)
expect(
    "quote_base defect names 'FX.EURGBP'",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=sens2, factor_meta=meta_bad, market=mkt2)
    ),
    contains=("FX.EURGBP", "quote_base"),
)

hdr("R4 — missing-factor diagnostic is DETERMINISTIC (was: 30 answers in 30 calls)")
keys = [f"F{i:04d}" for i in range(500)]
meta_many = pl.DataFrame(
    {
        "factor_id": keys,
        "displacement_convention_id": [1] * 500,
        "quote_base": [1.0] * 500,
    }
)
wide_sens = pl.DataFrame(
    {
        "position_id": np.arange(20_000) % 50,
        "factor_id": [f"F{i % 600:04d}" for i in range(20_000)],
        "first_order": np.ones(20_000),
        "second_order": np.zeros(20_000),
    }
)
mkt_many = ScenarioShocks.from_matrix(factor_keys=keys, displacement=np.ones((500, 2)))
messages = set()
for _ in range(30):
    try:
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=wide_sens, factor_meta=meta_many, market=mkt_many
            )
        )
    except SensitivityPnlInputError as e:
        messages.add(str(e))
check(
    "one message over 30 identical runs",
    len(messages) == 1,
    f"{len(messages)} distinct",
)
print("       ", sorted(messages)[0].splitlines()[1].strip()[:150])

# ================================================== conventions
hdr("R5 — RELATIVE and LOG conventions, against closed form")
# V = v * S. Delta w.r.t. RELATIVE displacement r = S/S0 - 1 is v*S0 = PV.
V, S0 = 1_000.0, 50.0
levels_1d = np.array([[55.0, 45.0, 50.0]])
for convention, name in ((2, "RELATIVE"), (3, "LOG")):
    s = pl.DataFrame(
        {
            "position_id": [1],
            "factor_id": [10],
            "first_order": [V * S0],
            "second_order": [0.0],
        }
    )
    m = pl.DataFrame(
        {
            "factor_id": [10],
            "displacement_convention_id": [convention],
            "quote_base": [1.0],
        }
    )
    lv = ScenarioLevels.from_matrix(
        factor_keys=[10], base_levels=np.array([S0]), shocked_levels=levels_1d
    )
    got = (
        run_sensitivity_pnl(
            SensitivityPnlRequest(sensitivities=s, factor_meta=m, market=lv)
        )
        .pnl[0]
        .astype(np.float64)
    )
    if convention == 2:
        want = V * S0 * (levels_1d[0] / S0 - 1.0)
    else:
        want = V * S0 * np.log(levels_1d[0] / S0)
    check(
        f"{name} displacement matches closed form",
        bool(np.allclose(got, want, rtol=1e-5)),
        f"{got} vs {want}",
    )

hdr("R6 — quote_base folding: per-basis-point DV01")
# dV/dz in natural units = -1e6. Quoted per bp: -100. quote_base = 1e4.
dv01_quoted = -100.0
s = pl.DataFrame(
    {
        "position_id": [7],
        "factor_id": [10],
        "first_order": [dv01_quoted],
        "second_order": [0.0],
    }
)
m = pl.DataFrame(
    {"factor_id": [10], "displacement_convention_id": [1], "quote_base": [1.0e4]}
)
shock_bp = np.array([[0.0025]])  # +25 bp in natural units
got = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=s,
        factor_meta=m,
        market=ScenarioShocks.from_matrix(factor_keys=[10], displacement=shock_bp),
    )
).pnl[0, 0]
check(
    "quote_base folded (-100/bp x 25bp = -2500)", abs(float(got) + 2500.0) < 1e-3, got
)

hdr("R7 — non-positive base level under RELATIVE is left to the engine")
lv_bad = ScenarioLevels.from_matrix(
    factor_keys=[10], base_levels=np.array([0.0]), shocked_levels=np.array([[1.0]])
)
expect(
    "engine rejects a zero base under a ratio convention",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id": [10],
                    "first_order": [1.0],
                    "second_order": [0.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [10],
                    "displacement_convention_id": [2],
                    "quote_base": [1.0],
                }
            ),
            market=lv_bad,
        )
    ),
    contains=("strictly positive",),
    exc=ValueError,
)

hdr("R8 — cross term with i == j is left to the compiler")
expect(
    "i == j rejected",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id": [10],
                    "first_order": [1.0],
                    "second_order": [0.0],
                }
            ),
            cross_sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id_i": [10],
                    "factor_id_j": [10],
                    "cross_second_order": [1.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [10],
                    "displacement_convention_id": [1],
                    "quote_base": [1.0],
                }
            ),
            market=ScenarioShocks.from_matrix(
                factor_keys=[10], displacement=np.ones((1, 1))
            ),
        )
    ),
    contains=("second_order",),
    exc=ValueError,
)

# ================================================== zero copy
hdr("R9 — zero-copy audit")
raw = np.ascontiguousarray(np.ones((4, 100), dtype=np.float32))
sh = ScenarioShocks.from_matrix(factor_keys=[1, 2, 3, 4], displacement=raw)
check(
    "f32 C-contiguous input adopted without a copy",
    sh.displacement is raw or np.shares_memory(sh.displacement, raw),
)
raw64 = np.ones((4, 100), dtype=np.float64)
sh64 = ScenarioShocks.from_matrix(factor_keys=[1, 2, 3, 4], displacement=raw64)
check(
    "f64 input converted exactly once", not np.shares_memory(sh64.displacement, raw64)
)
check(
    "converted result is C-contiguous f32",
    sh64.displacement.dtype == np.float32 and sh64.displacement.flags.c_contiguous,
)

hdr("R10 — result surface: no matrix-materialising convenience")
check(
    "no to_dataframe on the result", not hasattr(SensitivityPnlResult, "to_dataframe")
)
check("no to_polars on the result", not hasattr(SensitivityPnlResult, "to_polars"))
r = run_sensitivity_pnl(
    SensitivityPnlRequest(sensitivities=sens, factor_meta=meta_house, market=mkt)
)
check("row_for returns a view, not a copy", np.shares_memory(r.row_for(101), r.pnl))
try:
    r.row_for(999)
    check("row_for raises KeyError for an absent uid", False)
except KeyError as e:
    check("row_for raises KeyError for an absent uid", "999" in str(e))

# ================================================== scale
hdr("R11 — scale: 20k positions x 40 factors, 2000-factor market, 4000 scenarios")
rng = np.random.default_rng(7)
M, PER, K, S = 20_000, 40, 2_000, 4_000
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
big_market = ScenarioShocks.from_matrix(
    factor_keys=np.arange(K, dtype=np.int64),
    displacement=np.ascontiguousarray(rng.normal(0, 0.01, (K, S)).astype(np.float32)),
)
request = SensitivityPnlRequest(
    sensitivities=big_sens,
    factor_meta=big_meta,
    market=big_market,
    position_uids=np.arange(M, dtype=np.int64),
)
run_sensitivity_pnl(  # warm the JIT
    SensitivityPnlRequest(
        sensitivities=big_sens.head(100),
        factor_meta=big_meta,
        market=ScenarioShocks.from_matrix(
            factor_keys=np.arange(K, dtype=np.int64),
            displacement=np.ascontiguousarray(np.zeros((K, 8), dtype=np.float32)),
        ),
    )
)
tracemalloc.start()
t0 = time.perf_counter()
big = run_sensitivity_pnl(request)
elapsed = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(
    f"  nnz={big.provenance.num_first_order_entries:,}  Ka={big.provenance.num_active_factors}"
)
print(f"  wall {elapsed:.3f}s   python-tracked peak {peak / 1e6:.1f} MB")
print(
    f"  output block {big.pnl.nbytes / 1e6:.1f} MB   batch_size {big.provenance.batch_size}"
)
check("output shape", big.pnl.shape == (M, S))
check(
    "scratch stayed inside the budget",
    big.provenance.batch_size * big.provenance.num_active_factors * 4
    <= 16 * 1024 * 1024,
)
overhead = peak - big.pnl.nbytes
print(f"  overhead beyond the output block: {overhead / 1e6:.1f} MB")
check(
    "everything except the output block fits in 64 MB",
    overhead < 64e6,
    f"{overhead / 1e6:.1f} MB",
)

# spot-check one position against a hand computation
row_uid = 12_345
entries = big_sens.filter(pl.col("position_id") == row_uid)
coef = entries.get_column("first_order").to_numpy()
fidx = entries.get_column("factor_id").to_numpy()
want = coef @ big_market.displacement[fidx].astype(np.float64)
got = big.row_for(row_uid).astype(np.float64)
check(
    "hand-computed position matches the engine",
    bool(np.allclose(got, want, rtol=2e-4, atol=1e-2)),
    f"max err {np.abs(got - want).max():.4g}, scale {np.abs(want).max():.4g}",
)

# ================================================== handoff edges
hdr("R12 — handoff scenario-count mismatch")
grid = allocate_pnl_grid(
    block_position_uids=[np.array([101], dtype=np.int64)], num_scenarios=5
)
expect(
    "scenario mismatch rejected before compiling",
    lambda: write_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens, factor_meta=meta_house, market=mkt, position_uids=[101]
        ),
        grid=grid,
        block_index=0,
    ),
    contains=("scenario", "grid of 5"),
)
expect(
    "missing position_uids rejected",
    lambda: write_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=sens, factor_meta=meta_house, market=mkt),
        grid=allocate_pnl_grid(
            block_position_uids=[np.array([101], dtype=np.int64)], num_scenarios=2
        ),
        block_index=0,
    ),
    contains=("allocate_pnl_grid",),
)

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
sys.exit(1 if FAILS else 0)
