# src/riskquant/solutions/sensitivity_pnl/_verify/verify_contract.py
"""Ruthless Reviewer, pass 1: attack the sensitivity_pnl solutions facade."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    ScenarioLevels,
    ScenarioShocks,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
    write_sensitivity_pnl,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def expect_error(name: str, fn, *, must_contain: tuple[str, ...] = ()) -> None:
    try:
        fn()
    except SensitivityPnlInputError as e:
        text = str(e)
        missing = [s for s in must_contain if s not in text]
        check(name, not missing, f"message lacked {missing}\n---\n{text}\n---")
        if not missing:
            print(f"        {text.splitlines()[0]}")
        return
    except Exception as e:
        check(name, False, f"wrong type {type(e).__name__}: {e}")
        return
    check(name, False, "no error raised")


def hdr(s: str) -> None:
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


# ------------------------------------------------------------------ fixture
VOLUME, SPOT, FX = 50_000.0, 42.50, 0.85
HEDGE = -18_000.0


def base_frames(factor_keys=(10, 20)):
    f_s, f_x = factor_keys
    sens = pl.DataFrame(
        {
            "position_id": [101, 101, 202, 202],
            "factor_id": [f_s, f_x, f_s, f_x],
            "first_order": [VOLUME * FX, VOLUME * SPOT, HEDGE * FX, HEDGE * SPOT],
            "second_order": [0.0, 0.0, 0.0, 0.0],
        }
    )
    cross = pl.DataFrame(
        {
            "position_id": [101, 202],
            "factor_id_i": [f_s, f_s],
            "factor_id_j": [f_x, f_x],
            "cross_second_order": [VOLUME, HEDGE],
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": [f_s, f_x],
            "displacement_convention_id": [1, 1],
            "quote_base": [1.0, 1.0],
        }
    )
    return sens, cross, meta


def pv(volume: float, spot: float, fx: float) -> float:
    return volume * fx * spot


D_SPOT = np.array([2.50, -5.00, 0.0], dtype=np.float64)
D_FX = np.array([-0.03, 0.04, 0.0], dtype=np.float64)


def exact_pnl(volume: float) -> np.ndarray:
    return np.array(
        [
            pv(volume, SPOT + ds, FX + dx) - pv(volume, SPOT, FX)
            for ds, dx in zip(D_SPOT, D_FX, strict=True)
        ]
    )


# ------------------------------------------------------------------ 1
hdr("1 — numeric contract: exact reproduction of a bilinear payoff")
sens, cross, meta = base_frames()
shocks = ScenarioShocks.from_matrix(
    factor_keys=[10, 20],
    displacement=np.vstack([D_SPOT, D_FX]),
)
result = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens, factor_meta=meta, cross_sensitivities=cross, market=shocks
    )
)
print("  pnl =\n", result.pnl)
check(
    "row order is ascending position_uid", result.position_uids.tolist() == [101, 202]
)
check("dtype float32", result.pnl.dtype == np.float32)
check("C-contiguous", result.pnl.flags.c_contiguous)
check("shape (M, S)", result.pnl.shape == (2, 3))
for row, volume in ((0, VOLUME), (1, HEDGE)):
    got = result.pnl[row].astype(np.float64)
    want = exact_pnl(volume)
    check(
        f"row {row} matches closed form",
        bool(np.allclose(got, want, rtol=5e-6, atol=1e-6)),
        f"{got} vs {want}",
    )
print("  provenance:", result.provenance.model_dump())

# ------------------------------------------------------------------ 2
hdr("2 — string factor labels give bit-identical results")
sens_s, cross_s, meta_s = base_frames(("EQ.ACME", "FX.EURGBP"))
shocks_s = ScenarioShocks.from_matrix(
    factor_keys=["EQ.ACME", "FX.EURGBP"], displacement=np.vstack([D_SPOT, D_FX])
)
res_s = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens_s,
        factor_meta=meta_s,
        cross_sensitivities=cross_s,
        market=shocks_s,
    )
)
check(
    "string-keyed result is bit-identical to integer-keyed",
    bool(np.array_equal(res_s.pnl, result.pnl)),
    f"{res_s.pnl}\n{result.pnl}",
)

# ------------------------------------------------------------------ 3
hdr("3 — labels in REVERSE market order still resolve correctly")
shocks_r = ScenarioShocks.from_matrix(
    factor_keys=["FX.EURGBP", "EQ.ACME"], displacement=np.vstack([D_FX, D_SPOT])
)
res_r = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens_s,
        factor_meta=meta_s,
        cross_sensitivities=cross_s,
        market=shocks_r,
    )
)
check(
    "permuting BOTH keys and rows changes nothing",
    bool(np.array_equal(res_r.pnl, result.pnl)),
    f"{res_r.pnl}",
)

# ------------------------------------------------------------------ 4
hdr("4 — unsorted integer keys are accepted with ZERO matrix copy")
raw_unsorted = np.ascontiguousarray(np.vstack([D_FX, D_SPOT]).astype(np.float32))
opted = ScenarioShocks.from_matrix(factor_keys=[20, 10], displacement=raw_unsorted)
check(
    "matrix adopted by reference despite unsorted keys",
    np.shares_memory(opted.displacement, raw_unsorted),
)
check("permutation lives in the index, not the data", opted.axis.slot_map is not None)
check(
    "matrix rows untouched",
    bool(np.array_equal(opted.displacement, raw_unsorted)),
)
res_p = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens, factor_meta=meta, cross_sensitivities=cross, market=opted
    )
)
check(
    "unsorted keys yield the correct answer",
    bool(np.array_equal(res_p.pnl, result.pnl)),
    f"{res_p.pnl}\n{result.pnl}",
)
check(
    "slot encoding recorded in provenance",
    res_p.provenance.factor_key_encoding == "slot",
    res_p.provenance.factor_key_encoding,
)
check(
    "sorted integer keys still take the direct path",
    result.provenance.factor_key_encoding == "direct",
    result.provenance.factor_key_encoding,
)

# ------------------------------------------------------------------ 5
hdr("5 — null / float / string identifier columns")
bad = sens.with_columns(pl.Series("position_id", [101, None, 202, 202], dtype=pl.Int64))
expect_error(
    "null position_id rejected",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=bad, factor_meta=meta, market=shocks)
    ),
    must_contain=("null", "-9223372036854775808", "rows (0-based): 1"),
)
bad_f = sens.with_columns(pl.col("position_id").cast(pl.Float64))
expect_error(
    "float position_id rejected",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=bad_f, factor_meta=meta, market=shocks)
    ),
    must_contain=("Float64", "2.7 becomes position 2"),
)

# ------------------------------------------------------------------ 6
hdr("6 — NaN coefficient and NaN shock")
bad_v = sens.with_columns(pl.Series("first_order", [1.0, float("nan"), 3.0, 4.0]))
expect_error(
    "NaN coefficient rejected and localised",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=bad_v, factor_meta=meta, market=shocks)
    ),
    must_contain=("non-finite", "rows (0-based): 1"),
)
nan_shock = np.vstack([D_SPOT, D_FX]).copy()
nan_shock[1, 2] = np.nan
expect_error(
    "NaN shock rejected at market construction",
    lambda: ScenarioShocks.from_matrix(factor_keys=[10, 20], displacement=nan_shock),
    must_contain=("non-finite", "rows (0-based): 1"),
)

# ------------------------------------------------------------------ 7
hdr("7 — coverage: a sensitivity to a factor with no market row")
sens_x = pl.concat(
    [
        sens,
        pl.DataFrame(
            {
                "position_id": [101],
                "factor_id": [99],
                "first_order": [1.0],
                "second_order": [0.0],
            }
        ),
    ]
)
expect_error(
    "uncovered factor named in the message",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=sens_x, factor_meta=meta, market=shocks)
    ),
    must_contain=("99", "zero risk in every scenario"),
)

# ------------------------------------------------------------------ 8
hdr("8 — declared row set: hedged position holds its row")
res_d = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens,
        factor_meta=meta,
        cross_sensitivities=cross,
        market=shocks,
        position_uids=[101, 202, 303],
    )
)
check("declared row set honoured", res_d.position_uids.tolist() == [101, 202, 303])
check(
    "silent position emits exact zeros", bool(np.all(res_d.pnl[2] == 0.0)), res_d.pnl[2]
)
check(
    "provenance counts the silent position",
    res_d.provenance.num_positions_without_sensitivities == 1,
    res_d.provenance.num_positions_without_sensitivities,
)
expect_error(
    "undeclared position rejected",
    lambda: run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens, factor_meta=meta, market=shocks, position_uids=[101]
        )
    ),
    must_contain=("202", "absent from position_uids"),
)

# ------------------------------------------------------------------ 9
hdr("9 — levels path agrees with displacement path")
levels = ScenarioLevels.from_matrix(
    factor_keys=[10, 20],
    base_levels=np.array([SPOT, FX]),
    shocked_levels=np.vstack([SPOT + D_SPOT, FX + D_FX]),
)
res_l = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens, factor_meta=meta, cross_sensitivities=cross, market=levels
    )
)
check(
    "ABSOLUTE levels == displacements",
    bool(np.allclose(res_l.pnl, result.pnl, rtol=5e-6, atol=1e-6)),
    f"{res_l.pnl}\n{result.pnl}",
)
check("provenance records the contract", res_l.provenance.market_contract == "levels")

# ------------------------------------------------------------------ 10
hdr("10 — multiple defects reported together")
multi = pl.DataFrame(
    {
        "position_id": [101.0, None, 202.0],
        "factor_id": [10, 20, 77],
        "first_order": [1.0, float("inf"), 3.0],
        "second_order": [0.0, 0.0, 0.0],
    }
)
try:
    run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=multi, factor_meta=meta, market=shocks)
    )
    check("multi-defect report", False, "no error")
except SensitivityPnlInputError as e:
    print(str(e))
    check("reports more than one defect at once", len(e.defects) >= 2, len(e.defects))

# ------------------------------------------------------------------ 11
hdr("11 — batching: S not a multiple of the batch")
big_shocks = ScenarioShocks.from_matrix(
    factor_keys=[10, 20],
    displacement=np.vstack(
        [
            np.random.default_rng(0).normal(0, 1, 9973),
            np.random.default_rng(1).normal(0, 0.01, 9973),
        ]
    ),
)
res_b = run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens,
        factor_meta=meta,
        cross_sensitivities=cross,
        market=big_shocks,
    )
)
ds = big_shocks.displacement[0].astype(np.float64)
dx = big_shocks.displacement[1].astype(np.float64)
want = VOLUME * FX * ds + VOLUME * SPOT * dx + VOLUME * ds * dx
got = res_b.pnl[0].astype(np.float64)
check("9973 scenarios all written", res_b.pnl.shape == (2, 9973))
check(
    "every scenario correct across batch boundaries",
    bool(np.allclose(got, want, rtol=2e-5, atol=1e-3)),
    f"max abs err {np.abs(got - want).max()}",
)
print("  batch_size =", res_b.provenance.batch_size)

# ------------------------------------------------------------------ 12
hdr("12 — grid co-allocation handoff")
grid = allocate_pnl_grid(
    block_position_uids=[
        np.array([1, 2], dtype=np.int64),
        np.array([101, 202], dtype=np.int64),
    ],
    num_scenarios=3,
)
grid.block_view(0)[:] = 0.0
prov = write_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens,
        factor_meta=meta,
        cross_sensitivities=cross,
        market=shocks,
        position_uids=[101, 202],
    ),
    grid=grid,
    block_index=1,
)
check(
    "block written in place",
    bool(np.array_equal(grid.block_view(1), result.pnl)),
    grid.block_view(1),
)
expect_error(
    "row-set mismatch against the registered block rejected",
    lambda: write_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens,
            factor_meta=meta,
            market=shocks,
            position_uids=[101, 202, 303],
        ),
        grid=grid,
        block_index=1,
    ),
    must_contain=("np.empty",),
)

# ------------------------------------------------------------------ 13
hdr("13 — duplicate policy passes through")
dup = pl.concat([sens, sens.head(1)])
try:
    run_sensitivity_pnl(
        SensitivityPnlRequest(sensitivities=dup, factor_meta=meta, market=shocks)
    )
    check("duplicate key raises by default", False, "no error")
except ValueError as e:
    check(
        "duplicate key raises a ValueError from the compiler", "duplicate" in str(e), e
    )
    print("       ", str(e)[:110])

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
sys.exit(1 if FAILS else 0)
