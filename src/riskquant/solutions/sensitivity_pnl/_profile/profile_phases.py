# src/riskquant/solutions/sensitivity_pnl/_profile/profile_phases.py
"""Where does the wall time go? The facade must not tax the kernel."""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    ScenarioShocks,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
)
from riskquant.solutions.sensitivity_pnl.pipeline import (
    build_provenance,
    compile_book,
    make_engine,
    resolve_batch_size,
    scope_factor_meta,
    stream_scenarios,
)
from riskquant.solutions.sensitivity_pnl.validate import validate_book_spec

rng = np.random.default_rng(7)
M, PER, K, S = 20_000, 40, 2_000, 4_000
pos = np.repeat(np.arange(M, dtype=np.int64), PER)
fac = rng.integers(0, K, size=M * PER).astype(np.int64)
sens = (
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
meta = pl.DataFrame(
    {
        "factor_id": np.arange(K, dtype=np.int64),
        "displacement_convention_id": np.ones(K, dtype=np.int64),
        "quote_base": np.ones(K),
    }
)
market = ScenarioShocks.from_matrix(
    factor_keys=np.arange(K, dtype=np.int64),
    displacement=np.ascontiguousarray(rng.normal(0, 0.01, (K, S)).astype(np.float32)),
)
req = SensitivityPnlRequest(
    sensitivities=sens,
    factor_meta=meta,
    market=market,
    position_uids=np.arange(M, dtype=np.int64),
)
run_sensitivity_pnl(
    SensitivityPnlRequest(
        sensitivities=sens.head(50),
        factor_meta=meta,
        market=ScenarioShocks.from_matrix(
            factor_keys=np.arange(K, dtype=np.int64),
            displacement=np.ascontiguousarray(np.zeros((K, 8), dtype=np.float32)),
        ),
    )
)


def clock(label, fn):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    times[label] = dt
    return out


times: dict[str, float] = {}
spec = req.book_spec()
scoped = clock("scope_factor_meta", lambda: scope_factor_meta(spec))
clock("validate_book_spec", lambda: validate_book_spec(spec, factor_meta=scoped))
book = clock("compile_book", lambda: compile_book(spec, factor_meta=scoped))
bs = resolve_batch_size(num_active_factors=book.num_active_factors, num_scenarios=S)
grid = clock(
    "allocate_grid",
    lambda: allocate_pnl_grid(
        block_position_uids=[book.position_uids], num_scenarios=S
    ),
)
eng = clock(
    "make_engine",
    lambda: make_engine(
        book=book, market=market, out=grid.block_view(0), batch_size=bs
    ),
)
clock(
    "stream (KERNEL)",
    lambda: stream_scenarios(market=market, engine=eng, num_scenarios=S),
)
clock(
    "build_provenance",
    lambda: build_provenance(
        market=market,
        axis=spec.factor_axis,
        book=book,
        batch_size=bs,
        book_reused=False,
    ),
)

total = sum(times.values())
print(
    f"nnz={book.nnz:,}  Ka={book.num_active_factors}  M={book.num_positions}  S={S}  batch={bs}\n"
)
for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<20} {v * 1e3:9.1f} ms  {v / total:6.1%}")
print(f"  {'TOTAL':<20} {total * 1e3:9.1f} ms")
overhead = total - times["stream (KERNEL)"]
print(
    f"\n  facade overhead outside the kernel: {overhead * 1e3:.1f} ms ({overhead / total:.1%})"
)
