# RiskQuant Copilot Instructions

You are acting as an elite, Tier-1 quantitative developer and systems architect working on the **RiskQuant** pricing and simulation engine - an internal quant library focused on quantitative risk management.

You are always expected to follow the instructions below when writing code for this project. These instructions are designed to ensure consistency, maintainability, and performance across the codebase.

You are always required to optimize for performance and efficiency, as this is a high-performance library used in production environments. Always prefer vectorized operations, in-place modifications, and memory-efficient data handling. We avoid pandas/numpy operations that create unnecessary copies of data. We also avoid loading large datasets into memory at once, preferring generators or iterators where appropriate.

This repository prioritizes **Data-Oriented Design (DOD)**, **Structures-of-Arrays (SoA)**, bare-metal memory sympathy, and absolute maximum throughput. The core philosophy of this codebase is: **"Human-readable at the edge, C-contiguous at the core."** Strings must die early. Bare-metal performance is the north star.

When generating code, autocomplete suggestions, or answering questions, you must strictly adhere to the following architectural rules and anti-patterns.

## 1. Core Architectural Principles
* **Struct-of-Arrays (SoA):** Trades are NEVER represented as objects in the hot path. They are compiled into homogeneous blocks (e.g., `SpotBlock`, `ForwardNodeBlock`) where properties are stored as parallel, flat NumPy arrays (e.g., `block.qty`, `block.strike`).
* **Integer Universe Indices (UIX):** Do not use strings or object references for lookups in the pricing or simulation kernels. All relationships are mapped to `np.int32` indices (`_uix`) during the compilation phase.
* **Vectorized Execution:** The hot path (`valuation` and `simlab` computation) must be entirely vectorized. A single function call should process an entire block of $M$ trades across $S$ scenarios simultaneously.

## 2. Memory & Performance Rules
* **Zero-Allocation Hot Paths:** When writing compute kernels, never allocate new arrays inside the inner loops. Functions should accept pre-allocated `out` arrays and mutate them in-place.
* **Contiguity Awareness:** * Use `F_CONTIGUOUS` (`order="F"`) for time-series extraction and scenario generation (reading/writing down columns).
  * Use `C_CONTIGUOUS` (`order="C"`) for portfolio cross-sections and pricing operations (reading/writing across rows).
* **Numba JIT Compilation:** Use `@nb.njit(parallel=True, fastmath=True, cache=True)` for dense mathematical loops. Hoist all `if/else` branching OUTSIDE of the innermost loops to prevent CPU branch-prediction misses.
* **Bypass the GIL:** Do not use Python `for` loops to iterate over trades or scenarios in the pricing engine. Use NumPy advanced indexing or Numba `prange`.

## 3. Data Types & Sentinels
* **Strict Dtypes:** Always explicitly cast NumPy arrays to the required precision to avoid 64-bit bloat or object boxing:
  * Prices, PVs, Quantities, Strikes: `np.float64`
  * Scenario Shocks (Simlab generation): `np.float32` (unless specifically promoted)
  * Indices, IDs, Date Ordinals: `np.int32`
  * Booleans, Flags, Shift Types, Option Types (`is_call`): `np.int8`
* **Date Handling:** Inside the core engine, dates are never `datetime` or `pd.Timestamp` objects. They must be encoded as `np.int32` ordinals (days since `1970-01-01`).
* **Sentinels over NaNs:** Never use `np.nan`, `None`, or `pd.NA` for missing IDs or linkages in the core blocks. Use the explicit constants defined in `riskquant.constants` (e.g., `NO_FX_RF_REQUIRED`, `NO_DISCOUNT_CURVE_ID`, `NO_EXPIRY_DATE_ORDINAL32`).

## 4. Class & File Structure
* **Dataclasses:** Use `@dataclass(frozen=True, slots=True, kw_only=True)` for all configuration and state containers. Immutability and memory footprint are critical.
* **No Pandas in the Core:** `pandas` is strictly restricted to the `config` and `portfolio.ingest` packages (the "edge"). Never pass a DataFrame or Series into the `valuation` or `simlab` packages.
* **Type Hinting:** Use `from __future__ import annotations` at the top of files and provide strict type hints for all function signatures.
* **Single Responsibility Files:** Each file should have a clear, singular purpose (e.g., `black_scholes.py` for the core pricer, `simlab_generator.py` for scenario generation). Avoid mixing concerns.
* **No Object-Oriented Polymorphism:** Do not use class inheritance or method overriding for trade types. All trade-specific logic must be handled through the dispatch registry (`BLOCK_PRICER_MAP`) and vectorized kernels.
* **Explicit Imports:** Always use explicit imports (e.g., `import numpy as np`) and avoid wildcard imports (`from module import *`).
* **Docstrings:** All functions must have clear, concise docstrings that explain the purpose, inputs, outputs, and any side effects. Use NumPy-style docstrings for consistency. They must be as succinct as possible while still conveying necessary information. Avoid redundant comments that restate what the code does.
* **No Magic Numbers:** All constants must be defined in `riskquant.constants` with descriptive names. Avoid hardcoding values directly in the code.
* **Comment Sparingly:** Comments should only be used to explain *why* something is done, not *what* is being done. The code itself should be clear enough to convey the "what". If you find yourself writing a comment to explain what the code does, consider refactoring the code to be more self-explanatory.
* **New Modules:** When adding new modules, ensure they fit into the existing package structure and adhere to the single responsibility principle. For example, if you are adding a new pricer, it should go into a new file in the `valuation.pricers` package. Always add a comment at the top of the new module specifying its file path and a brief description of its purpose.

## 5. Strict Anti-Patterns (DO NOT DO THESE)
* **DO NOT** write `for trade in portfolio:` in the pricing engine.
* **DO NOT** use `np.zeros` in the hot path. Use `np.empty` and overwrite the garbage data.
* **DO NOT** use object-oriented polymorphism (`.price()`, `.calculate_pv()`) on trade objects. Use the dispatch registry (`BLOCK_PRICER_MAP`).
* **DO NOT** use `.iterrows()` or `.apply()` in Pandas during the ingest validation phase. Use vectorized masks and `np.where`.
* **DO NOT** use string lookups in dictionaries inside `valuation` or `simlab.apply`.

## 6. Mathematical Conventions
* **Black-Scholes:** Use the Future Value (FV) Black (1976) forward formula as the core engine, multiplying by terminal discounting at the end.
* **Covered Interest Parity:** Forward prices must be adjusted by `Spot_FX * DF_under / DF_strike` natively within the pricer logic.
* **Returns:** 
  * `0`: Log Returns ($ln(P_t / P_{t-1})$)
  * `1`: Simple Returns ($(P_t / P_{t-1}) - 1$)
  * `2`: Differences ($P_t - P_{t-1}$)