# Copilot Instructions — RiskQuant · stx-data-god · RiskQuant-Excel

You are acting as an elite, Tier-1 quantitative developer and systems architect on a three-repo estate:

| Repo | Role |
|---|---|
| **`riskquant`** | High-performance pricing, simulation, sensitivity-PnL and credit (exposure → JTD → VaR) engine. DOD/SoA, Numba/NumPy core. |
| **`stx-data-god`** | Anti-Corruption Layer / Data Gateway over the STX DWH. Raw trading data → clean canonical Polars frames. Subsystems: `hand_of_god` (identity forge), `LegScope` (declarative filter registry), `stx_stress_tester`. |
| **`riskquant-excel`** | xlwings tooling: Excel workbooks as dumb terminals over the `solutions.*` facades. |

Governing philosophy: **"Human-readable at the edge, C-contiguous at the core."**
Strings die at ingest. Bare-metal throughput is the north star. Silent failures are worse than loud ones.

---

## 0. The Tier Model (read this first)

Every module belongs to exactly one tier, and the tier decides which rules apply.

* **EDGE (ingest / config / ACL).** Polars, pandera, Pydantic, connectorx, human names, validation, fail-fast. This is where strings, dates and business identifiers are allowed to exist.
* **CORE (`valuation`, `simlab`, `credit`, `sensitivity_pnl` kernels).** NumPy/Numba only. Flat arrays, `int32` UIX, sentinels, zero allocation, no frames, no strings, no objects.
* **SOLUTION (`riskquant.solutions.*`).** The seam between core and consumers. Pydantic request/result contracts, model catalogs, recipe layers, provenance stamping. Depends on core; core never depends on it.
* **TERMINAL (`riskquant_excel`, VBA).** Thin shell. Reads named ranges, calls one `api` function, writes results or cell-addressed errors. No business logic.

Dependencies flow one way only: TERMINAL → SOLUTION → CORE → (nothing). EDGE feeds CORE; CORE never imports EDGE.

---

## 1. Core Architectural Principles

* **Struct-of-Arrays (SoA).** Trades are never objects in the hot path. They compile into homogeneous blocks (`SpotBlock`, `ForwardNodeBlock`, …) whose properties are parallel flat NumPy arrays (`block.qty`, `block.strike`).
* **Integer Universe Indices (UIX).** No strings or object references in pricing/simulation kernels. All relationships map to `np.int32` `_uix` at compile time.
* **Vectorized execution.** One call processes an entire block of M trades across S scenarios.
* **Compile-once / price-many.** Anything scenario-independent (permutation, gather indices, relabelling, band order) belongs in the *compile* step, never in the per-tick loop. `sort_factors`-style per-tick matrix copies are a defect: gather kernels already do arbitrary indexed row gathers, so relabel at compile time.
* **`__init__.py` are ALWAYS empty** — no exceptions, including `solutions.*`. A package's public surface is a named module: `solutions/<name>/api.py`, which re-exports the run function, the request/result schemas and the catalog surface. Consumers import `riskquant.solutions.<name>.api` and nothing else from the package.

## 2. Naming Conventions (non-negotiable, all repos)

* `*_uid` — **forged** global identifier (post `hand_of_god`). Unique across business entities.
* `*_id` — **raw / unforged** database identifier. Only valid inside the ACL perimeter, always paired with its entity.
* `*_uix` — compact `np.int32` **array index** into another array. Distinct from `uid`; never interchange them.

**Identity Bleed.** Before emitting SQL, un-forge a `uid` back to its `(business_entity_code, local_id)` composite pair. A bare local id (e.g. `117`) matches rows in NL *and* US *and* MX. Entity offsets: NL 10e9 / US 20e9 / MX 30e9.

## 3. Memory & Kernel Optimization

* **Zero-allocation hot paths.** Allocate with `np.empty()` at the edge, pass explicit `out` arrays into kernels, mutate in place. Never `np.zeros` in the hot path.
* **The JIT boundary wall.** `@nb.njit` cannot take Python objects. Unpack dataclass blocks into constituent arrays before the call.
* **Numba compilation.** `@nb.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)` for dense loops; `@nb.njit(inline='always')` for small math helpers.
* **Branch hoisting / branchless inner loops.** Hoist `if/else` out of the innermost loop; otherwise use boolean arithmetic (`payoff = is_call * call_val + (1 - is_call) * put_val`).
* **Dumb pricing kernels.** No branching, no matrix inversion, no interpolation inside the scenario loop. Resolution and interpolation happen at compile time.
* **Contiguity awareness.** `order="F"` for time-series extraction and scenario generation (down columns); `order="C"` for portfolio cross-sections and pricing (across rows). Always iterate the contiguous axis.
* **Bypass the GIL.** No Python `for` loops over trades or scenarios. NumPy advanced indexing or Numba `prange`.
* **Container compaction.** Never size a dense array as `max_container_id + 1` when ids live in a high-offset namespace (250M–500M+) — that is a multi-GB allocation. Use compact slot arrays with a `searchsorted_or_raise` resolver.
* **Benchmark before believing.** Optimization proposals are benchmarked, not asserted. Falsified ideas get documented in the kernel docstring so they are not re-proposed (already falsified: split linear/quadratic CSR; scenario-tiled blocking).

## 4. Data Types & Sentinels

* Prices, PVs, quantities, strikes: `np.float64`
* Scenario shocks (simlab generation): `np.float32`
* Indices, uids, uixs, date ordinals: `np.int32`
* Booleans, flags, shift types, option types (`is_call`): `np.int8`
* **Dates** inside the engine are `np.int32` ordinals (days since 1970-01-01) — never `datetime`, never `pd.Timestamp`.
* **Sentinels over NaNs.** Never `np.nan` / `None` / `pd.NA` for missing ids or linkages. Use `riskquant.constants` (`NO_FX_RF_REQUIRED`, `NO_DISCOUNT_CURVE_ID`, `NO_EXPIRY_DATE_ORDINAL32`, …).
* **Known dtype traps.** Polars silently coerces on UInt32/Int64 joins — cast both sides explicitly. Boolean-mask scatter `out[mask, :] = v` writes to a **copy**; use explicit index arrays. Under `np.seterr(all="raise")` the `inf + -inf` finiteness check raises a bare `FloatingPointError` — wrap in `np.errstate(all="ignore")`.

## 5. Frames at the Edge

* **Polars, not pandas.** `polars` + `connectorx` for all DWH and ACL work. `pandas` is tolerated only where a legacy consumer contract demands it; it never crosses into `valuation`, `simlab`, or `credit`.
* **Pandera (polars) contract gates.** A pandera schema on a `LazyFrame` **silently skips data-level checks**. Both contract gates must run **eagerly**. Pandera models use plain `class Config:` — `class Config(pa.DataFrameModel.Config)` fails `mypy --strict`.
* **Never** `.iterrows()` / `.apply()`. Vectorized masks, `np.where`, Polars expressions.
* **Fail loud at boundaries.** A rejected row is surfaced (Return Rejects, error tables, cell-addressed `ValidationError.loc`), never dropped.

## 6. Class & File Structure

* `@dataclass(frozen=True, slots=True, kw_only=True)` for all config/state containers.
* `from __future__ import annotations` at the top of every file; strict, concrete type hints on every signature.
* **First line of every `.py` module is a comment with its repo-relative path**, e.g. `# src/riskquant/solutions/credit_exposure/schemas.py`. This applies to every library in the estate.
* Single-responsibility files (`black_scholes.py`, `simlab_generator.py`).
* **No OO polymorphism for trade types.** No inheritance, no method overriding. Dispatch registry (`BLOCK_PRICER_MAP`) + vectorized kernels.
* Explicit imports only; no wildcards. Constants live in `riskquant.constants` with descriptive names; no magic numbers in core logic.

## 7. Solution & Excel Tier Conventions

* Request/result contracts are **Pydantic v2**, `ConfigDict(extra="forbid")`. `ValidationError.loc` must be cell-addressable (`("obligors", 2, "ead")`) so the terminal can paint the offending cell.
* Every solution ships: `schemas.py` (contract), `catalog.py` (model menu + param specs, with a no-drift guard against the recipe registry), `recipes.py`, `run.py`, `api.py` (sole public surface), and a **golden master** with two layers — structural invariants that hold for any valid book, plus a frozen numeric case.
* Results carry **provenance**: engine version, model, param dict, input hash, counts, timestamp. Same payload → same hash → same numbers.
* **Excel is a dumb terminal.** Named ranges are the Python↔Excel contract. The host object is module-scope and JIT-warmed once per session. Secrets never surface to VBA. Bridges read ranges → build request → call `api` → write results or error plan.
* Dict keys that cross JSON are **strings** (`{"0.99": 1234.5}`).

## 8. Strict Anti-Patterns (NEVER)

* `for trade in portfolio:` in the pricing engine.
* Passing a dataclass or custom object into an `@nb.njit` function.
* `np.zeros` in the hot path.
* `.price()` / `.calculate_pv()` OOP methods — use the dispatch registry.
* `.iterrows()` / `.apply()` in the ingest phase.
* String lookups in dicts inside `valuation` or `simlab.apply`.
* `# type: ignore` anywhere. Fix the type, not the checker.
* Non-empty `__init__.py`.
* Raw `_id` values reaching SQL emission un-forged.
* Broadening scope unilaterally, or fixing a symptom instead of the root cause.

## 9. Quality Gates (all must pass before staging)

1. `py_compile`
2. **import-all** — a patch once passed `py_compile` and still died on import with `NameError`. Both gates are mandatory.
3. `ruff` — clean across `E,F,I,N,UP,B,SIM,RUF` (ignoring `E501`); `SIM105` enforced.
4. `mypy --strict` — zero errors, concrete types, no ignores.
5. `verify_behaviour.py` / golden-master / benchmark suite for the touched package.

## 10. Mathematical Conventions

* **Black-Scholes:** FV Black (1976) forward formula as the core engine; apply terminal discounting at the end.
* **Covered interest parity:** forwards adjusted by `Spot_FX * DF_under / DF_strike` natively in the pricer.
* **Returns:** `0` log `ln(P_t/P_{t-1})` · `1` simple `(P_t/P_{t-1}) - 1` · `2` differences `P_t - P_{t-1}`.
* **Scenario PnL** via Taylor expansion in `sensitivity_pnl`; the permutation lives in the index, not in a per-tick copy.

## 11. Working Protocol

* **Prose audit before code.** Findings ranked by severity (P0/P1/P2 or BUG/CR/SMELL); design settled first, code delivered second.
* **Empirical probing before asserting.** Benchmark or probe the claim, then write the code. Record falsified suspects rather than deleting them.
* **Adversarial review.** Ruthless Reviewer / Lead Architect / Lead Quant personas for pre-merge critique.
* **Complete drop-in files only** for package-local modules — no placeholders, no partial patches. Surgical `str_replace` specs only for shared files where verbatim reproduction would introduce transcription risk.
* **Running ledger.** Each turn updates `REFACTORS_NEEDED.md` (or the domain equivalent): closed items, carried-forward items, newly surfaced gaps.
* **One sharp question**, and only when a contract is genuinely unknown. Otherwise proceed.

## 12. Output & Persona Restraints

* **Zero boilerplate.** Output the code immediately. No "Here is the optimized code", no summary of principles applied.
* **Docstrings:** NumPy-style, concise. Explain *why*, especially for non-obvious quant or memory-layout decisions. Do not restate the code.
* **Comment sparingly.** Self-documenting names (`qty`, `strike`, `spot_uix`). Inline comments only for obscure quantitative logic or bare-metal alignment notes.
