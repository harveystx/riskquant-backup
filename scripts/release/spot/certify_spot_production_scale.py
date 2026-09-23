# # scripts/release/spot/certify_spot_production_scale.py
# """
# SPOT production-scale certification gate.

# WINDOWS REFACTOR:
#     This release certification script was explicitly refactored to run on
#     native Windows Python.

#     The Unix-only `resource` module is imported lazily and is only used on
#     Unix-like platforms. On Windows, peak process resident memory is measured
#     through the native Win32 `GetProcessMemoryInfo` API using
#     `PeakWorkingSetSize`.

#     The script also bootstraps the RiskQuant repository root when executed
#     directly so that the canonical certification fixture under
#     `tests.solutions.risk_run._certfix` is importable regardless of the
#     caller's current working directory.

# Executes the ADMITTED SPOT production path at the release dimensions through
# the real entry point (`run_scenario_var_es`) on the canonical-ingest
# certification world (`tests/solutions/risk_run/_certfix.py`:
# raw frame -> IngestBatch -> run-owned compile), and reconciles it against an
# INDEPENDENT NumPy oracle that calls no production scenario generator,
# valuation, Taylor, PnlGrid, D081 or VaR/ES code.

# Levels reported:
#     - functional correctness
#     - production-scale execution
#     - multi-core determinism
#     - production-data readiness

# The machine-readable certificate is written automatically to:

#     scripts/release/spot/spot_production_scale_certifcate.json

# The output path can be overridden with:

#     --json-out <path>

# Usage:
#     python scripts/release/spot/certify_spot_production_scale.py
# """

# from __future__ import annotations

# import argparse
# import ctypes
# import json
# import math
# import os
# import pathlib
# import sys
# import time
# from ctypes import wintypes

# import numpy as np

# # ---------------------------------------------------------------------------
# # Repository-root bootstrap
# # ---------------------------------------------------------------------------
# # This release script imports the canonical certification fixture from
# # `tests.solutions.risk_run._certfix`. When the script is executed directly,
# # Python places only this script's directory on sys.path. Explicitly add the
# # repository root so the certification world is importable regardless of the
# # caller's current working directory.
# _REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# if not (_REPO_ROOT / "pyproject.toml").is_file():
#     raise RuntimeError(
#         f"Could not locate RiskQuant repository root from {__file__!s}: "
#         f"expected {_REPO_ROOT / 'pyproject.toml'}"
#     )

# if str(_REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(_REPO_ROOT))


# import riskquant.solutions.risk_run.api as api
# import tests.solutions.risk_run._certfix as cf
# from riskquant import constants
# from riskquant.solutions.risk_run.api import run_scenario_var_es
# from riskquant.solutions.risk_run.schemas import (
#     MethodologyDeclaration,
#     ValuationContext,
# )
# from riskquant.solutions.sensitivity_pnl.api import (
#     SensitivityPnlCalculator,
# )

# FX_ID = cf.K_TOTAL  # USDEUR node id (EUR per USD)
# ROW = {int(f): i for i, f in enumerate(cf.FIDS.tolist())}

# ATOL_PNL = 1e-3  # float32 PnlGrid
# ATOL_V0 = 1e-8
# ATOL_LEVEL = 1e-4  # float32 shocks on levels of order 100

# # Additional production-path stress sizes.
# #
# # Stress is specifically intended to increase the number of scenarios (S).
# # Batch size is increased with S, but remains materially smaller than S so
# # that the batching path is exercised rather than collapsing to one batch.
# #
# # The canonical certification world remains unchanged:
# #     S = 2,000
# #     batch = 256
# #
# # Stress worlds:
# #     S = 20,000   batch = 2,048
# #     S = 100,000  batch = 4,096
# STRESS_CASES = (
#     (20_000, 2_048),
#     (100_000, 4_096),
# )


# # ---------------------------------------------------------------------------
# # Windows peak-working-set support
# # ---------------------------------------------------------------------------
# # Explicit Windows refactor:
# #   Windows -> Win32 GetProcessMemoryInfo().PeakWorkingSetSize
# #   Unix    -> resource.getrusage().ru_maxrss
# #
# # This preserves the intent of the original metric: peak process resident
# # memory rather than current RSS at a single point in time.
# if sys.platform == "win32":

#     class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
#         """
#         Win32 PROCESS_MEMORY_COUNTERS structure.

#         `PeakWorkingSetSize` is the Windows measurement used for this
#         certification script's peak resident-memory metric.
#         """

#         _fields_ = [
#             ("cb", wintypes.DWORD),
#             ("PageFaultCount", wintypes.DWORD),
#             ("PeakWorkingSetSize", ctypes.c_size_t),
#             ("WorkingSetSize", ctypes.c_size_t),
#             ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
#             ("QuotaPagedPoolUsage", ctypes.c_size_t),
#             ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
#             ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
#             ("PagefileUsage", ctypes.c_size_t),
#             ("PeakPagefileUsage", ctypes.c_size_t),
#         ]

#     _KERNEL32 = ctypes.WinDLL(
#         "kernel32",
#         use_last_error=True,
#     )
#     _PSAPI = ctypes.WinDLL(
#         "psapi",
#         use_last_error=True,
#     )

#     _GET_CURRENT_PROCESS = _KERNEL32.GetCurrentProcess
#     _GET_CURRENT_PROCESS.argtypes = []
#     _GET_CURRENT_PROCESS.restype = wintypes.HANDLE

#     _GET_PROCESS_MEMORY_INFO = _PSAPI.GetProcessMemoryInfo
#     _GET_PROCESS_MEMORY_INFO.argtypes = [
#         wintypes.HANDLE,
#         ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
#         wintypes.DWORD,
#     ]
#     _GET_PROCESS_MEMORY_INFO.restype = wintypes.BOOL


# def _peak_rss_mib_windows() -> float:
#     """Return peak process working-set size in MiB on native Windows."""
#     counters = _PROCESS_MEMORY_COUNTERS()
#     counters.cb = ctypes.sizeof(counters)

#     process = _GET_CURRENT_PROCESS()

#     ok = _GET_PROCESS_MEMORY_INFO(
#         process,
#         ctypes.byref(counters),
#         counters.cb,
#     )

#     if not ok:
#         raise ctypes.WinError(ctypes.get_last_error())

#     return counters.PeakWorkingSetSize / (1024.0 * 1024.0)


# def _peak_rss_mib_unix() -> float:
#     """
#     Return peak resident set size in MiB on Unix-like systems.

#     Linux reports `ru_maxrss` in KiB.
#     macOS reports `ru_maxrss` in bytes.
#     """
#     import resource

#     rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#     if sys.platform == "darwin":
#         return float(rss) / (1024.0 * 1024.0)

#     return float(rss) / 1024.0


# def peak_rss_mib() -> float:
#     """
#     Return peak process resident memory in MiB.

#     The implementation is deliberately platform-specific because the Python
#     standard-library `resource` module is unavailable on native Windows.
#     """
#     if sys.platform == "win32":
#         return _peak_rss_mib_windows()

#     return _peak_rss_mib_unix()


# # ---------------------------------------------------------------------------
# # Independent oracle (plain NumPy; raw sources only)
# # ---------------------------------------------------------------------------
# def book():
#     """Per-position economics read from the RAW positions frame, ordered by U."""
#     f = cf.POSITIONS

#     by_uid = {
#         int(u): (float(q), str(n), str(c))
#         for u, q, n, c in zip(
#             f["uid"],
#             f["quantity"],
#             f["underlying"],
#             f["settlement_ccy"],
#             strict=True,
#         )
#     }

#     uids = cf.UIDS.tolist()

#     qty = np.array([by_uid[u][0] for u in uids])

#     u_row = np.array([ROW[cf.NAMES.index(by_uid[u][1]) + 1] for u in uids])

#     eur = np.array([by_uid[u][2] == "EUR" for u in uids])

#     return uids, qty, u_row, eur


# def oracle_rows(x0, xs, derived_mask):
#     """
#     V_0, dV (FULL exact / DERIVED SECOND_ORDER_FULL Taylor) and dW per UID.
#     """
#     _, qty, u_row, eur = book()

#     fx = ROW[FX_ID]

#     p0 = x0[u_row]
#     ps = xs[u_row]

#     x0fx = float(x0[fx])
#     xsfx = xs[fx]

#     fx0 = np.where(
#         eur,
#         x0fx,
#         1.0,
#     )

#     fxs = np.where(
#         eur[:, None],
#         xsfx[None, :],
#         1.0,
#     )

#     v0 = qty * p0 * fx0

#     dv_full = qty[:, None] * ps * fxs - v0[:, None]

#     dp = ps - p0[:, None]

#     dx = (xsfx - x0fx)[None, :] * eur[:, None]

#     dv_taylor = qty[:, None] * (fx0[:, None] * dp + p0[:, None] * dx + dp * dx)

#     dv = np.where(
#         derived_mask[:, None],
#         dv_taylor,
#         dv_full,
#     )

#     q0 = np.where(
#         eur,
#         1.0 / x0fx,
#         1.0,
#     )

#     qs = np.where(
#         eur[:, None],
#         1.0 / xsfx[None, :],
#         1.0,
#     )

#     dw = dv * qs + v0[:, None] * (qs - q0[:, None])

#     check = (v0[:, None] + dv) * qs - v0[:, None] * q0[:, None]

#     assert np.abs(dw - check).max() < 1e-7, "D081 forms disagree in the oracle"

#     return v0, dv, dw


# def oracle_historical_levels(x0):
#     """
#     Historical, horizon 1, LOG_RETURN, no proxy:

#         x_s = x_0 exp(ln(H_t / H_t-1)).
#     """
#     vals = np.asarray(
#         cf.VALS,
#         dtype=np.float64,
#     )[:, cf.FIDS - 1]

#     r = np.log(vals[1:] / vals[:-1]).T

#     return x0[:, None] * np.exp(r)


# def oracle_var_es(p, c):
#     """
#     Re-implement the production convention:

#         alpha = 1 - c
#         m = alpha * S
#         m is snapped to the nearest integer within 16 eps
#         VaR = ceil(m)-th lower order statistic
#         ES = (
#             sum of the floor(m) worst observations
#             + frac * the next observation
#         ) / m
#     """
#     s = np.sort(
#         np.asarray(
#             p,
#             dtype=np.float64,
#         )
#     )

#     m = (1.0 - c) * s.shape[0]

#     near = np.rint(m)

#     if abs(m - near) <= (16.0 * np.finfo(np.float64).eps * max(1.0, abs(m))):
#         m = float(near)

#     f = int(math.floor(m))
#     w = m - f

#     var = s[max(0, f - 1)] if w == 0.0 else s[f]

#     es = (s[:f].sum() + (w * s[f] if w > 0.0 else 0.0)) / m

#     return float(var), float(es)


# # ---------------------------------------------------------------------------
# # Production execution with capture and coarse stage timing
# # ---------------------------------------------------------------------------
# class Capture:
#     def __init__(self):
#         self.t = {
#             "valuation": 0.0,
#             "taylor": 0.0,
#             "d081": 0.0,
#             "var_es": 0.0,
#         }

#         self.seen = {}
#         self._real = {}

#     def __enter__(self):
#         self._real = {
#             "value": api._value_positions,
#             "convert": api.convert_scenario_pnl_to_reporting_currency,
#             "var": api.compute_var_es,
#             "taylor": SensitivityPnlCalculator.run_into_grid,
#             "write": api._write_full_reval_block,
#         }

#         def timed(key, fn):
#             def wrapper(*a, **kw):
#                 t0 = time.perf_counter()

#                 out = fn(*a, **kw)

#                 self.t[key] += time.perf_counter() - t0

#                 return out

#             return wrapper

#         api._value_positions = timed(
#             "valuation",
#             self._real["value"],
#         )

#         SensitivityPnlCalculator.run_into_grid = timed(
#             "taylor",
#             self._real["taylor"],
#         )

#         def convert(**kw):
#             self.seen["v0"] = np.array(
#                 kw["base_settlement_values"],
#                 copy=True,
#             )

#             self.seen["settlement"] = np.array(
#                 kw["settlement_pnl"],
#                 copy=True,
#             )

#             t0 = time.perf_counter()

#             out = self._real["convert"](**kw)

#             self.t["d081"] += time.perf_counter() - t0

#             self.seen["reported"] = np.array(
#                 out.converted_pnl,
#                 dtype=np.float64,
#                 copy=True,
#             )

#             self.seen["portfolio"] = np.array(
#                 out.portfolio_pnl,
#                 dtype=np.float64,
#                 copy=True,
#             )

#             return out

#         def write(**kw):
#             self.seen["shocked"] = np.array(
#                 kw["shocked_levels"],
#                 copy=True,
#             )

#             return self._real["write"](**kw)

#         api.convert_scenario_pnl_to_reporting_currency = convert
#         api.compute_var_es = timed(
#             "var_es",
#             self._real["var"],
#         )
#         api._write_full_reval_block = write

#         return self

#     def __exit__(self, *exc):
#         api._value_positions = self._real["value"]
#         api.convert_scenario_pnl_to_reporting_currency = self._real["convert"]
#         api.compute_var_es = self._real["var"]
#         SensitivityPnlCalculator.run_into_grid = self._real["taylor"]
#         api._write_full_reval_block = self._real["write"]


# def run(
#     model,
#     *,
#     batch_size=256,
#     seed=7,
#     mixed=True,
#     scenario_count=None,
# ):
#     inp = cf.inputs(
#         model,
#         batch_size=batch_size,
#         seed=seed,
#         mixed=mixed,
#         scenario_count=scenario_count,
#     )

#     with Capture() as cap:
#         t0 = time.perf_counter()

#         result = run_scenario_var_es(**inp)

#         wall = time.perf_counter() - t0

#     return inp, result, cap, wall


# def reconcile(model, inp, result, cap):
#     """Per-UID and tail reconciliation against the independent oracle."""
#     x0 = np.asarray(
#         inp["base_levels"],
#         dtype=np.float64,
#     )

#     xs = np.asarray(
#         cap.seen["shocked"],
#         dtype=np.float64,
#     )

#     derived = np.isin(
#         cf.UIDS,
#         inp["methodology"].taylor_derived_uids,
#     )

#     v0, dv, dw = oracle_rows(
#         x0,
#         xs,
#         derived,
#     )

#     out = {
#         "scenarios": int(xs.shape[1]),
#         "max_abs_v0_error": float(np.abs(cap.seen["v0"] - v0).max()),
#         "max_abs_dv_error": float(
#             np.abs(np.asarray(cap.seen["settlement"]) - dv).max()
#         ),
#         "max_abs_dw_error": float(np.abs(cap.seen["reported"] - dw).max()),
#         "max_abs_portfolio_error": float(
#             np.abs(cap.seen["portfolio"] - dw.sum(axis=0)).max()
#         ),
#     }

#     var, es = oracle_var_es(
#         dw.sum(axis=0),
#         inp["confidence_level"],
#     )

#     out["var_production"] = float(result.var[-1])
#     out["var_oracle"] = var

#     out["es_production"] = float(result.es[-1])
#     out["es_oracle"] = es

#     out["var_error"] = abs(float(result.var[-1]) - var)

#     out["es_error"] = abs(float(result.es[-1]) - es)

#     if model == "historical":
#         # Exact independent scenario engine.
#         out["max_abs_level_error"] = float(
#             np.abs(xs - oracle_historical_levels(x0)).max()
#         )

#     out["ok"] = (
#         out["max_abs_v0_error"] < ATOL_V0
#         and out["max_abs_dv_error"] < ATOL_PNL
#         and out["max_abs_dw_error"] < ATOL_PNL
#         and out["var_error"] < 10 * ATOL_PNL
#         and out["es_error"] < 10 * ATOL_PNL
#         and out.get(
#             "max_abs_level_error",
#             0.0,
#         )
#         < ATOL_LEVEL
#     )

#     return out


# # ---------------------------------------------------------------------------
# # Adversarial sections
# # ---------------------------------------------------------------------------
# def identity_mutations():
#     base_inp, base_run, _, _ = run("sbb")
#     results = {}

#     def outcome(name, inp):
#         try:
#             r = run_scenario_var_es(**inp)
#         except Exception as exc:
#             # Any refusal is a pass.
#             results[name] = f"refused:{type(exc).__name__}"
#             return

#         results[name] = (
#             "identity_changed" if r.identity != base_run.identity else "SAME IDENTITY"
#         )

#     mapper2 = cf.FX.build_uix_mapper(
#         [cf._BATCH],
#         as_of_ord32=cf.AS_OF + 1,
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     inp["valuation_context"] = ValuationContext(
#         static_config=cf.RSC,
#         ingest_batch=cf._BATCH,
#         uix_mapper=mapper2,
#         discounting_mode=cf.NONE,
#     )

#     outcome(
#         "as_of",
#         inp,
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     inp["valuation_context"] = ValuationContext(
#         static_config=cf._static_config(first_equity_shift="SIMPLE_RETURN"),
#         ingest_batch=cf._BATCH,
#         uix_mapper=cf.MAPPER,
#         discounting_mode=cf.NONE,
#     )

#     outcome(
#         "static_config",
#         inp,
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     inp["valuation_context"] = ValuationContext(
#         static_config=cf._static_config(),
#         ingest_batch=cf._BATCH,
#         uix_mapper=cf.MAPPER,
#         discounting_mode=cf.NONE,
#     )

#     results["static_config_same_content_new_object"] = (
#         "same_identity_as_expected"
#         if run_scenario_var_es(**inp).identity == base_run.identity
#         else "IDENTITY DRIFT"
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     inp["valuation_context"] = ValuationContext(
#         static_config=cf.RSC,
#         ingest_batch=cf._BATCH,
#         uix_mapper=cf.MAPPER,
#         discounting_mode=cf.FX.DiscountingMode.CURVE,
#     )

#     outcome(
#         "discounting_mode",
#         inp,
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     order = np.r_[
#         1,
#         0,
#         np.arange(2, cf.K),
#     ]

#     for k in (
#         "required_factor_ids",
#         "required_shift_type_id",
#         "source_factor_ids",
#         "shock_scale",
#         "base_levels",
#     ):
#         inp[k] = np.ascontiguousarray(np.asarray(inp[k])[order])

#     outcome(
#         "factor_axis_order",
#         inp,
#     )

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     keep = cf.UIDS[:-1]

#     inp["methodology"] = MethodologyDeclaration(
#         production_position_uids=keep,
#         full_reval_uids=np.intersect1d(
#             keep,
#             cf.partition(True)[0],
#         ),
#         taylor_supplied_uids=[],
#         taylor_derived_uids=np.intersect1d(
#             keep,
#             cf.partition(True)[1],
#         ),
#     )

#     outcome(
#         "economic_universe",
#         inp,
#     )

#     u_full, u_derived = cf.partition(True)

#     inp = cf.inputs(
#         "sbb",
#         mixed=True,
#     )

#     moved_f = np.sort(
#         np.append(
#             u_full,
#             u_derived[0],
#         )
#     )

#     inp["methodology"] = MethodologyDeclaration(
#         production_position_uids=cf.UIDS,
#         full_reval_uids=moved_f,
#         taylor_supplied_uids=[],
#         taylor_derived_uids=u_derived[1:],
#     )

#     outcome(
#         "methodology_partition",
#         inp,
#     )

#     outcome(
#         "scenario_seed",
#         cf.inputs(
#             "sbb",
#             seed=11,
#             mixed=True,
#         ),
#     )

#     fp = constants.TTE_FLOOR

#     try:
#         constants.TTE_FLOOR = fp * 2.0

#         outcome(
#             "valuation_contract",
#             cf.inputs(
#                 "sbb",
#                 mixed=True,
#             ),
#         )
#     finally:
#         constants.TTE_FLOOR = fp

#     return results


# def determinism_and_batching():
#     out = {}

#     runs = {
#         b: run(
#             "sbb",
#             batch_size=b,
#         )
#         for b in (
#             64,
#             256,
#             2000,
#         )
#     }

#     ref = runs[256][1]

#     out["batch_identity_equal"] = all(
#         r[1].identity == ref.identity for r in runs.values()
#     )

#     out["batch_var_bitwise_equal"] = all(
#         [v.hex() for v in r[1].var.tolist()] == [v.hex() for v in ref.var.tolist()]
#         for r in runs.values()
#     )

#     out["batch_es_bitwise_equal"] = all(
#         [v.hex() for v in r[1].es.tolist()] == [v.hex() for v in ref.es.tolist()]
#         for r in runs.values()
#     )

#     again = run(
#         "sbb",
#         batch_size=256,
#     )[1]

#     out["repeat_bitwise_equal"] = [v.hex() for v in again.var.tolist()] == [
#         v.hex() for v in ref.var.tolist()
#     ] and again.identity == ref.identity

#     other = run(
#         "filtered_bootstrap",
#         batch_size=256,
#     )[1]  # different scenarios in between

#     third = run(
#         "sbb",
#         batch_size=256,
#     )[1]

#     out["no_cross_run_leakage"] = [v.hex() for v in third.var.tolist()] == [
#         v.hex() for v in ref.var.tolist()
#     ] and other.identity != ref.identity

#     return out


# def stress_execution():
#     """
#     Stress the official production path at materially larger batch sizes.

#     This does not alter the canonical certification world. Each stress run
#     uses the same fixture, same seed, same model and same independent oracle;
#     only the production batch size is increased.

#     The stress gate checks:
#         - independent oracle reconciliation
#         - stable identity
#         - bitwise-identical VaR/ES versus the baseline batch
#         - wall-clock execution
#     """
#     out = {}

#     _, baseline, _, _ = run(
#         "sbb",
#         batch_size=256,
#         seed=7,
#         mixed=True,
#     )

#     for scenario_count, batch_size in STRESS_CASES:
#         rss_before = peak_rss_mib()

#         inp = cf.inputs(
#             "sbb",
#             batch_size=batch_size,
#             seed=7,
#             mixed=True,
#             scenario_count=scenario_count,
#         )

#         with Capture() as cap:
#             t0 = time.perf_counter()

#             result = run_scenario_var_es(**inp)

#             wall = time.perf_counter() - t0

#         rss_after = peak_rss_mib()

#         rec = reconcile(
#             "sbb",
#             inp,
#             result,
#             cap,
#         )

#         rec["S_target"] = int(scenario_count)
#         rec["batch_size"] = int(batch_size)
#         rec["wall_s"] = round(wall, 3)

#         rec["stages_s"] = {k: round(v, 3) for k, v in cap.t.items()}

#         rec["stages_s"]["scenario_and_overhead"] = round(
#             wall - sum(cap.t.values()),
#             3,
#         )

#         rec["peak_rss_mib_before"] = round(
#             rss_before,
#             1,
#         )

#         rec["peak_rss_mib_after"] = round(
#             rss_after,
#             1,
#         )

#         rec["identity_equal_baseline"] = result.identity == baseline.identity

#         rec["var_bitwise_equal_baseline"] = [v.hex() for v in result.var.tolist()] == [
#             v.hex() for v in baseline.var.tolist()
#         ]

#         rec["es_bitwise_equal_baseline"] = [v.hex() for v in result.es.tolist()] == [
#             v.hex() for v in baseline.es.tolist()
#         ]

#         rec["ok"] = (
#             rec["ok"]
#             and rec["identity_equal_baseline"]
#             and rec["var_bitwise_equal_baseline"]
#             and rec["es_bitwise_equal_baseline"]
#         )

#         out[f"S{scenario_count}_B{batch_size}"] = rec

#     return out


# # ---------------------------------------------------------------------------
# # Release gate
# # ---------------------------------------------------------------------------
# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description=__doc__,
#     )

#     parser.add_argument(
#         "--json-out",
#         default=str(
#             pathlib
#             .Path(__file__)
#             .resolve()
#             .with_name("spot_production_scale_certifcate.json")
#         ),
#         help=(
#             "Path for the machine-readable certification JSON. "
#             "Defaults to spot_production_scale_certifcate.json "
#             "beside this script."
#         ),
#     )

#     args = parser.parse_args()

#     cores = os.cpu_count() or 1

#     report = {
#         "dimensions": {
#             "M": int(cf.M),
#             "K_axis": int(cf.K),
#             "K_nodes": int(cf.K_TOTAL),
#             "T": int(cf.T),
#             "S_target": int(cf.S),
#             "batch_size": 256,
#             "confidence_level": 0.99,
#             "discounting": "NONE",
#             "cores": cores,
#         },
#         "models": {},
#         "identity": {},
#         "determinism": {},
#         "stress": {},
#         "performance": {},
#     }

#     rss0 = peak_rss_mib()

#     for model in cf.CFGS:
#         inp, result, cap, wall = run(model)

#         rec = reconcile(
#             model,
#             inp,
#             result,
#             cap,
#         )

#         rec["wall_s"] = round(
#             wall,
#             3,
#         )

#         rec["stages_s"] = {k: round(v, 3) for k, v in cap.t.items()}

#         rec["stages_s"]["scenario_and_overhead"] = round(
#             wall - sum(cap.t.values()),
#             3,
#         )

#         report["models"][model] = rec

#     report["identity"] = identity_mutations()

#     report["determinism"] = determinism_and_batching()

#     report["stress"] = stress_execution()

#     report["performance"] = {
#         "peak_rss_mib_before": round(
#             rss0,
#             1,
#         ),
#         "peak_rss_mib_after": round(
#             peak_rss_mib(),
#             1,
#         ),
#         "threads": int(
#             os.environ.get(
#                 "NUMBA_NUM_THREADS",
#                 cores,
#             )
#         ),
#         "cores": cores,
#     }

#     ok_models = all(m["ok"] for m in report["models"].values())

#     ok_identity = all(
#         v.startswith("refused")
#         or v
#         in (
#             "identity_changed",
#             "same_identity_as_expected",
#         )
#         for v in report["identity"].values()
#     )

#     ok_det = all(report["determinism"].values())

#     ok_stress = all(rec["ok"] for rec in report["stress"].values())

#     report["verdict"] = {
#         "functional_correctness": (
#             "CERTIFIED"
#             if (ok_models and ok_identity and ok_det and ok_stress)
#             else "NOT CERTIFIED"
#         ),
#         "production_scale_execution": (
#             "CERTIFIED" if (ok_models and ok_stress) else "NOT CERTIFIED"
#         ),
#         "multi_core_determinism": ("NOT TESTED" if cores < 2 else "see release gate"),
#         "production_data_readiness": ("NOT TESTED"),
#     }

#     output = json.dumps(
#         report,
#         indent=2,
#         sort_keys=True,
#     )

#     print(output)

#     # Always emit the machine-readable release certificate.
#     json_path = pathlib.Path(args.json_out).resolve()

#     json_path.write_text(
#         output,
#         encoding="utf-8",
#     )

#     print(f"\n  JSON certificate written to {json_path}")

#     return 0 if (ok_models and ok_identity and ok_det and ok_stress) else 1


# if __name__ == "__main__":
#     raise SystemExit(main())


# scripts/release/spot/certify_spot_production_scale.py
"""
SPOT production-scale certification gate.

WINDOWS REFACTOR:
    This release certification script was explicitly refactored to run on
    native Windows Python.

    The Unix-only `resource` module is imported lazily and is only used on
    Unix-like systems. On Windows, peak process resident memory is measured
    through the native Win32 `GetProcessMemoryInfo` API using
    `PeakWorkingSetSize`.

    The script also bootstraps the RiskQuant repository root when executed
    directly so that the canonical certification fixture under
    `tests.solutions.risk_run._certfix` is importable regardless of the
    caller's current working directory.

Executes the ADMITTED SPOT production path at the release dimensions through
the real entry point (`run_scenario_var_es`) on the canonical-ingest
certification world (`tests/solutions/risk_run/_certfix.py`:
raw frame -> IngestBatch -> run-owned compile), and reconciles it against an
INDEPENDENT NumPy oracle that calls no production scenario generator,
valuation, Taylor, PnlGrid, D081 or VaR/ES code.

Levels reported:
    - functional correctness
    - production-scale execution
    - multi-core determinism
    - production-data readiness

The canonical release dimensions remain controlled by the certification
fixture. Large-S stress cases override only the per-invocation scenario count
and batch size.

The machine-readable certificate is written automatically to:

    scripts/release/spot/spot_production_scale_certifcate.json

The output path can be overridden with:

    --json-out <path>

Usage:
    python scripts/release/spot/certify_spot_production_scale.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pathlib
import sys
import time
from ctypes import wintypes

import numpy as np

# ---------------------------------------------------------------------------
# Repository-root bootstrap
# ---------------------------------------------------------------------------
# This release script imports the canonical certification fixture from
# `tests.solutions.risk_run._certfix`. When the script is executed directly,
# Python places only this script's directory on sys.path. Explicitly add the
# repository root so the certification world is importable regardless of the
# caller's current working directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        f"Could not locate RiskQuant repository root from {__file__!s}: "
        f"expected {_REPO_ROOT / 'pyproject.toml'}"
    )

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import riskquant.solutions.risk_run.api as api  # noqa: E402
import tests.solutions.risk_run._certfix as cf  # noqa: E402
from riskquant import constants  # noqa: E402
from riskquant.solutions.risk_run.api import run_scenario_var_es  # noqa: E402
from riskquant.solutions.risk_run.schemas import (  # noqa: E402
    MethodologyDeclaration,
    ValuationContext,
)
from riskquant.solutions.sensitivity_pnl.api import (  # noqa: E402
    SensitivityPnlCalculator,
)

FX_ID = cf.K_TOTAL  # USDEUR node id (EUR per USD)
ROW = {int(f): i for i, f in enumerate(cf.FIDS.tolist())}

ATOL_PNL = 1e-3  # float32 PnlGrid
ATOL_V0 = 1e-8
ATOL_LEVEL = 1e-4  # float32 shocks on levels of order 100

# ---------------------------------------------------------------------------
# Large-S production-path stress cases
# ---------------------------------------------------------------------------
# These are deliberately scenario-heavy, with batch size materially smaller
# than S so the production batching path is exercised across many batches.
#
# Canonical release world:
#     S = 2,000
#     batch = 256
#
# Large-S stress worlds:
#     S = 20,000    batch = 2,048   -> 10x scenario count
#     S = 100,000   batch = 4,096   -> 50x scenario count
STRESS_CASES = (
    (20_000, 2_048),
    (100_000, 4_096),
)

# Independent oracle chunk size. This bounds temporary oracle working memory
# independently of the total scenario population.
ORACLE_CHUNK_SIZE = 4_096


# ---------------------------------------------------------------------------
# Windows peak-working-set support
# ---------------------------------------------------------------------------
# Explicit Windows refactor:
#   Windows -> Win32 GetProcessMemoryInfo().PeakWorkingSetSize
#   Unix    -> resource.getrusage().ru_maxrss
#
# This preserves the intent of the original metric: peak process resident
# memory rather than current RSS at a single point in time.
if sys.platform == "win32":

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        """
        Win32 PROCESS_MEMORY_COUNTERS structure.

        `PeakWorkingSetSize` is the Windows measurement used for this
        certification script's peak resident-memory metric.
        """

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

    _KERNEL32 = ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    )

    _PSAPI = ctypes.WinDLL(
        "psapi",
        use_last_error=True,
    )

    _GET_CURRENT_PROCESS = _KERNEL32.GetCurrentProcess
    _GET_CURRENT_PROCESS.argtypes = []
    _GET_CURRENT_PROCESS.restype = wintypes.HANDLE

    _GET_PROCESS_MEMORY_INFO = _PSAPI.GetProcessMemoryInfo
    _GET_PROCESS_MEMORY_INFO.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    _GET_PROCESS_MEMORY_INFO.restype = wintypes.BOOL


def _peak_rss_mib_windows() -> float:
    """Return peak process working-set size in MiB on native Windows."""
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)

    process = _GET_CURRENT_PROCESS()

    ok = _GET_PROCESS_MEMORY_INFO(
        process,
        ctypes.byref(counters),
        counters.cb,
    )

    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

    return counters.PeakWorkingSetSize / (1024.0 * 1024.0)


def _peak_rss_mib_unix() -> float:
    """
    Return peak resident set size in MiB on Unix-like systems.

    Linux reports `ru_maxrss` in KiB.
    macOS reports `ru_maxrss` in bytes.
    """
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)

    return float(rss) / 1024.0


def peak_rss_mib() -> float:
    """
    Return process-lifetime peak resident memory in MiB.

    The implementation is deliberately platform-specific because the Python
    standard-library `resource` module is unavailable on native Windows.
    """
    if sys.platform == "win32":
        return _peak_rss_mib_windows()

    return _peak_rss_mib_unix()


# ---------------------------------------------------------------------------
# Independent oracle (plain NumPy; raw sources only)
# ---------------------------------------------------------------------------
def book():
    """Per-position economics read from the RAW positions frame, ordered by U."""
    f = cf.POSITIONS

    by_uid = {
        int(u): (float(q), str(n), str(c))
        for u, q, n, c in zip(
            f["uid"],
            f["quantity"],
            f["underlying"],
            f["settlement_ccy"],
            strict=True,
        )
    }

    uids = cf.UIDS.tolist()

    qty = np.array([by_uid[u][0] for u in uids])

    u_row = np.array([ROW[cf.NAMES.index(by_uid[u][1]) + 1] for u in uids])

    eur = np.array([by_uid[u][2] == "EUR" for u in uids])

    return uids, qty, u_row, eur


def oracle_rows(x0, xs, derived_mask):
    """
    V_0, dV (FULL exact / DERIVED SECOND_ORDER_FULL Taylor) and dW per UID.
    """
    _, qty, u_row, eur = book()

    fx = ROW[FX_ID]

    p0 = x0[u_row]
    ps = xs[u_row]

    x0fx = float(x0[fx])
    xsfx = xs[fx]

    fx0 = np.where(
        eur,
        x0fx,
        1.0,
    )

    fxs = np.where(
        eur[:, None],
        xsfx[None, :],
        1.0,
    )

    v0 = qty * p0 * fx0

    dv_full = qty[:, None] * ps * fxs - v0[:, None]

    dp = ps - p0[:, None]

    dx = (xsfx - x0fx)[None, :] * eur[:, None]

    dv_taylor = qty[:, None] * (fx0[:, None] * dp + p0[:, None] * dx + dp * dx)

    dv = np.where(
        derived_mask[:, None],
        dv_taylor,
        dv_full,
    )

    q0 = np.where(
        eur,
        1.0 / x0fx,
        1.0,
    )

    qs = np.where(
        eur[:, None],
        1.0 / xsfx[None, :],
        1.0,
    )

    dw = dv * qs + v0[:, None] * (qs - q0[:, None])

    check = (v0[:, None] + dv) * qs - v0[:, None] * q0[:, None]

    assert np.abs(dw - check).max() < 1e-7, "D081 forms disagree in the oracle"

    return v0, dv, dw


def oracle_historical_levels(x0):
    """
    Historical, horizon 1, LOG_RETURN, no proxy:

        x_s = x_0 exp(ln(H_t / H_t-1)).
    """
    vals = np.asarray(
        cf.VALS,
        dtype=np.float64,
    )[:, cf.FIDS - 1]

    r = np.log(vals[1:] / vals[:-1]).T

    return x0[:, None] * np.exp(r)


def oracle_var_es(p, c):
    """
    Re-implement the production convention:

        alpha = 1 - c
        m = alpha * S
        m is snapped to the nearest integer within 16 eps
        VaR = ceil(m)-th lower order statistic
        ES = (
            sum of the floor(m) worst observations
            + frac * the next observation
        ) / m
    """
    s = np.sort(
        np.asarray(
            p,
            dtype=np.float64,
        )
    )

    m = (1.0 - c) * s.shape[0]

    near = np.rint(m)

    if abs(m - near) <= (16.0 * np.finfo(np.float64).eps * max(1.0, abs(m))):
        m = float(near)

    f = int(math.floor(m))
    w = m - f

    var = s[max(0, f - 1)] if w == 0.0 else s[f]

    es = (s[:f].sum() + (w * s[f] if w > 0.0 else 0.0)) / m

    return float(var), float(es)


def oracle_portfolio_pnl(
    x0,
    xs,
    derived_mask,
    *,
    chunk_size=ORACLE_CHUNK_SIZE,
):
    """
    Independent large-S oracle.

    Computes the portfolio reported P&L directly from raw economics and
    shocked levels, in bounded scenario chunks. Unlike `oracle_rows`, this
    deliberately does not materialize full U x S intermediate matrices.

    Returns:
        v0: per-UID base values
        portfolio_pnl: independent portfolio P&L by scenario
    """
    _, qty, u_row, eur = book()

    fx = ROW[FX_ID]

    p0 = x0[u_row]

    x0fx = float(x0[fx])

    fx0 = np.where(
        eur,
        x0fx,
        1.0,
    )

    v0 = qty * p0 * fx0

    q0 = np.where(
        eur,
        1.0 / x0fx,
        1.0,
    )

    scenarios = int(xs.shape[1])

    portfolio = np.empty(
        scenarios,
        dtype=np.float64,
    )

    for start in range(
        0,
        scenarios,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            scenarios,
        )

        ps = xs[
            u_row,
            start:stop,
        ]

        xsfx = xs[
            fx,
            start:stop,
        ]

        fxs = np.where(
            eur[:, None],
            xsfx[None, :],
            1.0,
        )

        dv_full = qty[:, None] * ps * fxs - v0[:, None]

        dp = ps - p0[:, None]

        dx = (xsfx - x0fx)[None, :] * eur[:, None]

        dv_taylor = qty[:, None] * (fx0[:, None] * dp + p0[:, None] * dx + dp * dx)

        dv = np.where(
            derived_mask[:, None],
            dv_taylor,
            dv_full,
        )

        qs = np.where(
            eur[:, None],
            1.0 / xsfx[None, :],
            1.0,
        )

        dw = dv * qs + v0[:, None] * (qs - q0[:, None])

        portfolio[start:stop] = dw.sum(axis=0)

    return v0, portfolio


# ---------------------------------------------------------------------------
# Production execution with capture and coarse stage timing
# ---------------------------------------------------------------------------
class Capture:
    def __init__(
        self,
        *,
        capture_detail=True,
        capture_shocked=True,
    ):
        self.capture_detail = bool(capture_detail)
        self.capture_shocked = bool(capture_shocked)

        self.t = {
            "valuation": 0.0,
            "taylor": 0.0,
            "d081": 0.0,
            "var_es": 0.0,
        }

        self.seen = {}
        self._real = {}

    def __enter__(self):
        self._real = {
            "value": api._value_positions,
            "convert": api.convert_scenario_pnl_to_reporting_currency,
            "var": api.compute_var_es,
            "taylor": SensitivityPnlCalculator.run_into_grid,
            "write": api._write_full_reval_block,
        }

        def timed(key, fn):
            def wrapper(*a, **kw):
                t0 = time.perf_counter()

                out = fn(*a, **kw)

                self.t[key] += time.perf_counter() - t0

                return out

            return wrapper

        api._value_positions = timed(
            "valuation",
            self._real["value"],
        )

        SensitivityPnlCalculator.run_into_grid = timed(
            "taylor",
            self._real["taylor"],
        )

        def convert(**kw):
            self.seen["v0"] = np.array(
                kw["base_settlement_values"],
                copy=True,
            )

            if self.capture_detail:
                self.seen["settlement"] = np.array(
                    kw["settlement_pnl"],
                    copy=True,
                )

            t0 = time.perf_counter()

            out = self._real["convert"](**kw)

            self.t["d081"] += time.perf_counter() - t0

            if self.capture_detail:
                self.seen["reported"] = np.array(
                    out.converted_pnl,
                    dtype=np.float64,
                    copy=True,
                )

            self.seen["portfolio"] = np.array(
                out.portfolio_pnl,
                dtype=np.float64,
                copy=True,
            )

            return out

        def write(**kw):
            if self.capture_shocked:
                self.seen["shocked"] = np.array(
                    kw["shocked_levels"],
                    dtype=np.float64,
                    copy=True,
                )

            return self._real["write"](**kw)

        api.convert_scenario_pnl_to_reporting_currency = convert
        api.compute_var_es = timed(
            "var_es",
            self._real["var"],
        )
        api._write_full_reval_block = write

        return self

    def __exit__(self, *exc):
        api._value_positions = self._real["value"]
        api.convert_scenario_pnl_to_reporting_currency = self._real["convert"]
        api.compute_var_es = self._real["var"]
        SensitivityPnlCalculator.run_into_grid = self._real["taylor"]
        api._write_full_reval_block = self._real["write"]


def run(
    model,
    *,
    batch_size=256,
    seed=7,
    mixed=True,
    total_scenarios=None,
    capture_detail=True,
    capture_shocked=True,
):
    inp = cf.inputs(
        model,
        batch_size=batch_size,
        seed=seed,
        mixed=mixed,
        total_scenarios=total_scenarios,
    )

    with Capture(
        capture_detail=capture_detail,
        capture_shocked=capture_shocked,
    ) as cap:
        t0 = time.perf_counter()

        result = run_scenario_var_es(**inp)

        wall = time.perf_counter() - t0

    return inp, result, cap, wall


def reconcile(model, inp, result, cap):
    """Full per-UID and tail reconciliation against the independent oracle."""
    x0 = np.asarray(
        inp["base_levels"],
        dtype=np.float64,
    )

    xs = np.asarray(
        cap.seen["shocked"],
        dtype=np.float64,
    )

    derived = np.isin(
        cf.UIDS,
        inp["methodology"].taylor_derived_uids,
    )

    v0, dv, dw = oracle_rows(
        x0,
        xs,
        derived,
    )

    out = {
        "scenarios": int(xs.shape[1]),
        "max_abs_v0_error": float(np.abs(cap.seen["v0"] - v0).max()),
        "max_abs_dv_error": float(
            np.abs(np.asarray(cap.seen["settlement"]) - dv).max()
        ),
        "max_abs_dw_error": float(np.abs(cap.seen["reported"] - dw).max()),
        "max_abs_portfolio_error": float(
            np.abs(cap.seen["portfolio"] - dw.sum(axis=0)).max()
        ),
    }

    var, es = oracle_var_es(
        dw.sum(axis=0),
        inp["confidence_level"],
    )

    out["var_production"] = float(result.var[-1])
    out["var_oracle"] = var

    out["es_production"] = float(result.es[-1])
    out["es_oracle"] = es

    out["var_error"] = abs(float(result.var[-1]) - var)

    out["es_error"] = abs(float(result.es[-1]) - es)

    if model == "historical":
        out["max_abs_level_error"] = float(
            np.abs(xs - oracle_historical_levels(x0)).max()
        )

    out["ok"] = (
        out["max_abs_v0_error"] < ATOL_V0
        and out["max_abs_dv_error"] < ATOL_PNL
        and out["max_abs_dw_error"] < ATOL_PNL
        and out["var_error"] < 10 * ATOL_PNL
        and out["es_error"] < 10 * ATOL_PNL
        and out.get(
            "max_abs_level_error",
            0.0,
        )
        < ATOL_LEVEL
    )

    return out


def reconcile_large_s(
    inp,
    result,
    cap,
    *,
    oracle_chunk_size=ORACLE_CHUNK_SIZE,
):
    """
    Memory-bounded large-S reconciliation.

    The canonical certification performs full per-UID reconciliation. Large-S
    stress deliberately validates:
        - producer base values
        - independent portfolio P&L oracle
        - independent VaR / ES oracle
        - scenario count
    while avoiding full U x S oracle materialization.
    """
    x0 = np.asarray(
        inp["base_levels"],
        dtype=np.float64,
    )

    xs = np.asarray(
        cap.seen["shocked"],
        dtype=np.float64,
    )

    if xs.ndim != 2:
        raise AssertionError("large-S oracle expected 2-D shocked levels")

    scenario_count = int(xs.shape[1])

    expected_s = inp["scenario_config"].total_scenarios

    if int(expected_s) != scenario_count:
        raise AssertionError(
            "production shocked-level scenario count does not match "
            f"scenario_config: {scenario_count} != {expected_s}"
        )

    derived = np.isin(
        cf.UIDS,
        inp["methodology"].taylor_derived_uids,
    )

    v0, portfolio_oracle = oracle_portfolio_pnl(
        x0,
        xs,
        derived,
        chunk_size=oracle_chunk_size,
    )

    production_v0 = np.asarray(
        cap.seen["v0"],
        dtype=np.float64,
    )

    production_portfolio = np.asarray(
        cap.seen["portfolio"],
        dtype=np.float64,
    )

    if production_portfolio.shape != portfolio_oracle.shape:
        raise AssertionError(
            "production portfolio P&L shape does not match oracle: "
            f"{production_portfolio.shape} != "
            f"{portfolio_oracle.shape}"
        )

    var, es = oracle_var_es(
        portfolio_oracle,
        inp["confidence_level"],
    )

    out = {
        "scenarios": scenario_count,
        "max_abs_v0_error": float(np.abs(production_v0 - v0).max()),
        "max_abs_portfolio_error": float(
            np.abs(production_portfolio - portfolio_oracle).max()
        ),
        "var_production": float(result.var[-1]),
        "var_oracle": var,
        "es_production": float(result.es[-1]),
        "es_oracle": es,
    }

    out["var_error"] = abs(out["var_production"] - out["var_oracle"])

    out["es_error"] = abs(out["es_production"] - out["es_oracle"])

    out["ok"] = (
        out["max_abs_v0_error"] < ATOL_V0
        and out["max_abs_portfolio_error"] < ATOL_PNL
        and out["var_error"] < 10 * ATOL_PNL
        and out["es_error"] < 10 * ATOL_PNL
    )

    return out


# ---------------------------------------------------------------------------
# Adversarial sections
# ---------------------------------------------------------------------------
def identity_mutations():
    base_inp, base_run, _, _ = run("sbb")
    results = {}

    def outcome(name, inp):
        try:
            r = run_scenario_var_es(**inp)
        except Exception as exc:
            results[name] = f"refused:{type(exc).__name__}"
            return

        results[name] = (
            "identity_changed" if r.identity != base_run.identity else "SAME IDENTITY"
        )

    mapper2 = cf.FX.build_uix_mapper(
        [cf._BATCH],
        as_of_ord32=cf.AS_OF + 1,
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    inp["valuation_context"] = ValuationContext(
        static_config=cf.RSC,
        ingest_batch=cf._BATCH,
        uix_mapper=mapper2,
        discounting_mode=cf.NONE,
    )

    outcome(
        "as_of",
        inp,
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    inp["valuation_context"] = ValuationContext(
        static_config=cf._static_config(first_equity_shift="SIMPLE_RETURN"),
        ingest_batch=cf._BATCH,
        uix_mapper=cf.MAPPER,
        discounting_mode=cf.NONE,
    )

    outcome(
        "static_config",
        inp,
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    inp["valuation_context"] = ValuationContext(
        static_config=cf._static_config(),
        ingest_batch=cf._BATCH,
        uix_mapper=cf.MAPPER,
        discounting_mode=cf.NONE,
    )

    results["static_config_same_content_new_object"] = (
        "same_identity_as_expected"
        if run_scenario_var_es(**inp).identity == base_run.identity
        else "IDENTITY DRIFT"
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    inp["valuation_context"] = ValuationContext(
        static_config=cf.RSC,
        ingest_batch=cf._BATCH,
        uix_mapper=cf.MAPPER,
        discounting_mode=cf.FX.DiscountingMode.CURVE,
    )

    outcome(
        "discounting_mode",
        inp,
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    order = np.r_[
        1,
        0,
        np.arange(2, cf.K),
    ]

    for key in (
        "required_factor_ids",
        "required_shift_type_id",
        "source_factor_ids",
        "shock_scale",
        "base_levels",
    ):
        inp[key] = np.ascontiguousarray(np.asarray(inp[key])[order])

    outcome(
        "factor_axis_order",
        inp,
    )

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    keep = cf.UIDS[:-1]

    inp["methodology"] = MethodologyDeclaration(
        production_position_uids=keep,
        full_reval_uids=np.intersect1d(
            keep,
            cf.partition(True)[0],
        ),
        taylor_supplied_uids=[],
        taylor_derived_uids=np.intersect1d(
            keep,
            cf.partition(True)[1],
        ),
    )

    outcome(
        "economic_universe",
        inp,
    )

    u_full, u_derived = cf.partition(True)

    inp = cf.inputs(
        "sbb",
        mixed=True,
    )

    moved_f = np.sort(
        np.append(
            u_full,
            u_derived[0],
        )
    )

    inp["methodology"] = MethodologyDeclaration(
        production_position_uids=cf.UIDS,
        full_reval_uids=moved_f,
        taylor_supplied_uids=[],
        taylor_derived_uids=u_derived[1:],
    )

    outcome(
        "methodology_partition",
        inp,
    )

    outcome(
        "scenario_seed",
        cf.inputs(
            "sbb",
            seed=11,
            mixed=True,
        ),
    )

    fp = constants.TTE_FLOOR

    try:
        constants.TTE_FLOOR = fp * 2.0

        outcome(
            "valuation_contract",
            cf.inputs(
                "sbb",
                mixed=True,
            ),
        )
    finally:
        constants.TTE_FLOOR = fp

    return results


def determinism_and_batching():
    out = {}

    runs = {
        batch_size: run(
            "sbb",
            batch_size=batch_size,
        )
        for batch_size in (
            64,
            256,
            2000,
        )
    }

    ref = runs[256][1]

    out["batch_identity_equal"] = all(
        r[1].identity == ref.identity for r in runs.values()
    )

    out["batch_var_bitwise_equal"] = all(
        [v.hex() for v in r[1].var.tolist()] == [v.hex() for v in ref.var.tolist()]
        for r in runs.values()
    )

    out["batch_es_bitwise_equal"] = all(
        [v.hex() for v in r[1].es.tolist()] == [v.hex() for v in ref.es.tolist()]
        for r in runs.values()
    )

    again = run(
        "sbb",
        batch_size=256,
    )[1]

    out["repeat_bitwise_equal"] = [v.hex() for v in again.var.tolist()] == [
        v.hex() for v in ref.var.tolist()
    ] and again.identity == ref.identity

    other = run(
        "filtered_bootstrap",
        batch_size=256,
    )[1]

    third = run(
        "sbb",
        batch_size=256,
    )[1]

    out["no_cross_run_leakage"] = [v.hex() for v in third.var.tolist()] == [
        v.hex() for v in ref.var.tolist()
    ] and other.identity != ref.identity

    return out


def stress_execution():
    """
    Exercise the official production path at materially larger S.

    Each case has:
        - a same-S canonical-batch reference run
        - a large-S, larger-batch production run
        - an independent, chunked NumPy portfolio/tail oracle

    The large-S oracle deliberately avoids materializing full U x S
    intermediate matrices so the stress measurement remains primarily a
    production-path stress rather than an oracle-memory stress.
    """
    out = {}

    for total_scenarios, batch_size in STRESS_CASES:
        if batch_size >= total_scenarios:
            raise ValueError(
                "stress batch_size must be smaller than total_scenarios: "
                f"S={total_scenarios}, batch={batch_size}"
            )

        # Same stochastic world, canonical batch size. This establishes the
        # batch-invariance reference for this exact large-S population.
        _, baseline, _, _ = run(
            "sbb",
            batch_size=256,
            seed=7,
            mixed=True,
            total_scenarios=total_scenarios,
            capture_detail=False,
            capture_shocked=False,
        )

        rss_before = peak_rss_mib()

        inp, result, cap, wall = run(
            "sbb",
            batch_size=batch_size,
            seed=7,
            mixed=True,
            total_scenarios=total_scenarios,
            capture_detail=False,
            capture_shocked=True,
        )

        rss_after = peak_rss_mib()

        rec = reconcile_large_s(
            inp,
            result,
            cap,
            oracle_chunk_size=ORACLE_CHUNK_SIZE,
        )

        scenario_batches = math.ceil(total_scenarios / batch_size)

        rec["S_target"] = int(total_scenarios)
        rec["batch_size"] = int(batch_size)
        rec["scenario_batches"] = int(scenario_batches)
        rec["wall_s"] = round(
            wall,
            3,
        )

        rec["scenarios_per_second"] = round(
            total_scenarios / wall,
            1,
        )

        rec["position_scenario_evaluations"] = int(cf.M * total_scenarios)

        rec["position_scenario_evaluations_per_second"] = round(
            (cf.M * total_scenarios) / wall,
            1,
        )

        rec["stages_s"] = {key: round(value, 3) for key, value in cap.t.items()}

        rec["stages_s"]["scenario_and_overhead"] = round(
            wall - sum(cap.t.values()),
            3,
        )

        # `peak_rss_mib` is process-lifetime high-water memory, not a
        # per-run current-RSS measurement. The labels deliberately say HWM.
        rec["process_peak_rss_hwm_before_mib"] = round(
            rss_before,
            1,
        )

        rec["process_peak_rss_hwm_after_mib"] = round(
            rss_after,
            1,
        )

        rec["identity_equal_same_S"] = result.identity == baseline.identity

        rec["var_bitwise_equal_same_S"] = [v.hex() for v in result.var.tolist()] == [
            v.hex() for v in baseline.var.tolist()
        ]

        rec["es_bitwise_equal_same_S"] = [v.hex() for v in result.es.tolist()] == [
            v.hex() for v in baseline.es.tolist()
        ]

        rec["ok"] = (
            rec["ok"]
            and rec["identity_equal_same_S"]
            and rec["var_bitwise_equal_same_S"]
            and rec["es_bitwise_equal_same_S"]
        )

        out[f"S{total_scenarios}_B{batch_size}"] = rec

    return out


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--json-out",
        default=str(
            pathlib
            .Path(__file__)
            .resolve()
            .with_name("spot_production_scale_certifcate.json")
        ),
        help=(
            "Path for the machine-readable certification JSON. "
            "Defaults to spot_production_scale_certifcate.json "
            "beside this script."
        ),
    )

    args = parser.parse_args()

    cores = os.cpu_count() or 1

    report = {
        "dimensions": {
            "M": int(cf.M),
            "K_axis": int(cf.K),
            "K_nodes": int(cf.K_TOTAL),
            "T": int(cf.T),
            "S_target": int(cf.S),
            "batch_size": 256,
            "confidence_level": 0.99,
            "discounting": "NONE",
            "cores": cores,
        },
        "models": {},
        "identity": {},
        "determinism": {},
        "stress": {},
        "performance": {},
    }

    rss0 = peak_rss_mib()

    for model in cf.CFGS:
        inp, result, cap, wall = run(model)

        rec = reconcile(
            model,
            inp,
            result,
            cap,
        )

        rec["wall_s"] = round(
            wall,
            3,
        )

        rec["stages_s"] = {key: round(value, 3) for key, value in cap.t.items()}

        rec["stages_s"]["scenario_and_overhead"] = round(
            wall - sum(cap.t.values()),
            3,
        )

        report["models"][model] = rec

    report["identity"] = identity_mutations()

    report["determinism"] = determinism_and_batching()

    report["stress"] = stress_execution()

    report["performance"] = {
        "peak_rss_mib_before": round(
            rss0,
            1,
        ),
        "peak_rss_mib_after": round(
            peak_rss_mib(),
            1,
        ),
        "threads": int(
            os.environ.get(
                "NUMBA_NUM_THREADS",
                cores,
            )
        ),
        "cores": cores,
    }

    ok_models = all(model["ok"] for model in report["models"].values())

    ok_identity = all(
        value.startswith("refused")
        or value
        in (
            "identity_changed",
            "same_identity_as_expected",
        )
        for value in report["identity"].values()
    )

    ok_det = all(report["determinism"].values())

    ok_stress = bool(report["stress"]) and all(
        rec["ok"] for rec in report["stress"].values()
    )

    report["verdict"] = {
        "functional_correctness": (
            "CERTIFIED"
            if (ok_models and ok_identity and ok_det and ok_stress)
            else "NOT CERTIFIED"
        ),
        "production_scale_execution": (
            "CERTIFIED" if (ok_models and ok_stress) else "NOT CERTIFIED"
        ),
        "multi_core_determinism": ("NOT TESTED" if cores < 2 else "see release gate"),
        "production_data_readiness": "NOT TESTED",
    }

    output = json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )

    print(output)

    # Always emit the machine-readable release certificate.
    json_path = pathlib.Path(args.json_out).resolve()

    json_path.write_text(
        output,
        encoding="utf-8",
    )

    print(f"\n  JSON certificate written to {json_path}")

    return 0 if (ok_models and ok_identity and ok_det and ok_stress) else 1


if __name__ == "__main__":
    raise SystemExit(main())
