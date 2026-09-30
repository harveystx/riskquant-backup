# scripts/release/simlab/certify_simlab_multicore.py
"""
SimLab multi-core release certification gate.

Run on a genuinely multi-core machine:

    python scripts/release/simlab/certify_simlab_multicore.py

It exercises the OFFICIAL production path only —
`riskquant.solutions.risk_run.api.run_scenario_var_es` — and emits a
human-readable certificate plus a machine-readable JSON one.

Three claims are kept strictly separate, because collapsing them is exactly how a
single-CPU host can produce a false multi-core PASS:

    A. NUMBA MULTI-THREAD EXECUTION VERIFIED
       >= 2 distinct `numba.get_thread_id()` values execute inside a real
       `prange`. True even when those workers time-slice one CPU.

    B. MULTI-CORE HARDWARE AVAILABLE
       The PROCESS has >= 2 CPUs, measured by `len(os.sched_getaffinity(0))`
       where available (affinity, not `os.cpu_count()`: a cgroup-limited or
       taskset-pinned process may see many CPUs and be allowed one).

    C. MULTI-CORE DETERMINISM CERTIFIED
       A and B both hold AND every certification check passes. Only C is the
       release gate.

Exit codes: 0 = PASS, 1 = FAIL, 2 = NOT EXECUTED (hardware prerequisite unmet).
NOT EXECUTED is never reported as PASS.

Comparisons are exact: sha256 over `dtype|shape|bytes` for arrays and
`float.hex()` for scalars. No tolerance is used anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"

#: Objects compared, in the order a divergence must be reported (the earliest
#: differing object is the diagnosis, not the final number).
DIVERGENCE_ORDER = [
    "scenario_stream",
    "shocked_levels",
    "trade_pnl",
    "portfolio_pnl",
    "var",
    "es",
    "manifest_identity",
    "riskrun_identity",
]

MODELS = [
    "historical",
    "filtered_historical",
    "sbb",
    "filtered_bootstrap",
    "parametric",
]
THREAD_MATRIX = [1, 2, 4, 8]
BATCH_MATRIX = [1, 3, 7, 64, "S", "S+1"]
BLAS_ENVS = [
    {"OMP_NUM_THREADS": "1"},
    {"OMP_NUM_THREADS": "8"},
    {"OPENBLAS_NUM_THREADS": "1"},
    {"OPENBLAS_NUM_THREADS": "8"},
    {"MKL_NUM_THREADS": "1"},
]


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------


def effective_cpus() -> tuple[int, str]:
    """Effective CPUs for THIS process. Affinity wins over `cpu_count`."""
    # Dynamically fetch the function, returning None if it doesn't exist
    affinity_func = getattr(os, "sched_getaffinity", None)

    if affinity_func:
        try:
            return len(affinity_func(0)), "sched_getaffinity"
        except OSError:
            pass

    return (os.cpu_count() or 1), "cpu_count"


# --------------------------------------------------------------------------
# child-process plumbing (NUMBA_NUM_THREADS fixes the ceiling at import time,
# so every thread configuration needs its own interpreter)
# --------------------------------------------------------------------------


def _child(script: str, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(env_extra or {})
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        return {"__error__": proc.stderr[-1500:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"__error__": f"unparsable child output: {proc.stdout[-500:]}"}


_WORKER_PROBE = textwrap.dedent(
    """
    import json
    import numpy as np, numba
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _ids(out):
        for i in prange(out.shape[0]):
            out[i] = np.int64(numba.get_thread_id())

    o = np.zeros(16384, dtype=np.int64)
    _ids(o)
    print(json.dumps({
        "configured": int(numba.get_num_threads()),
        "observed_workers": int(np.unique(o).size),
        "worker_ids": [int(x) for x in np.unique(o)][:16],
        "layer": str(numba.threading_layer()),
        "numba": numba.__version__,
    }))
    """
)

#: The fixture and the official call, executed in the child. Intermediates are
#: INSPECTED with the same production functions; the numerical path is not
#: recreated. Since PR-04 (D085) the run derives every settlement currency and
#: reporting FX route itself and binds scenario data to ONE canonical run axis
#: (valuation prefix, then any reporting-only FX suffix). The inspection reads
#: the axis from `derive_run_factor_axis`, prices Taylor on the valuation
#: prefix, and rebuilds the reporting plan from the TRADE RECORD (raw positions
#: frame) and the public FX authority, never from run internals.
_FIXTURE = textwrap.dedent(
    """
    import hashlib, json
    import numpy as np, polars as pl, numba

    from riskquant.constants import NO_FX_RF_REQUIRED
    from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
    from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
    from riskquant.reporting.fx_conversion import (
        build_fx_conversion_plan, convert_scenario_pnl_to_reporting_currency)
    from riskquant.scenario_metrics.api.var_es import compute_var_es
    from riskquant.sensitivity_pnl.conventions import DisplacementConvention
    from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
    from riskquant.config.proxy_model.compile import compile_proxy_model
    from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
    from riskquant.market_data.risk_factors.shift_types import ShiftType
    from riskquant.simlab.shocks.axis_operator import lower_axis_operator
    from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
    from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world
    from riskquant.utils.business_days import ord_to_date
    from riskquant.solutions.risk_run.api import (
        derive_run_factor_axis, run_scenario_var_es)
    from riskquant.solutions.risk_run.schemas import (
        MethodologyDeclaration, ReportingSpec, TaylorBlock)
    from riskquant.solutions.run_identity.hashing import (
        hash_parametric_risk_data, hash_risk_factor_matrix)
    from riskquant.solutions.scenario_models.api import (
        build_scenario_generator, parse_scenario_model_config)
    from riskquant.solutions.sensitivity_pnl.api import (
        FactorAxis, SensitivityBookSpec, SensitivityPnlCalculator)

    # P1-B: the gate certifies the SAME real valuation world as the release tests
    # (tests/solutions/risk_run/_certfix.py), resized to this gate's dimensions.
    import os, pathlib, sys
    import riskquant
    os.environ.update(RQ_CERT_FACTORS="{K}", RQ_CERT_OBSERVATIONS="{T}",
                      RQ_CERT_SCENARIOS="{S}", RQ_CERT_POSITIONS="{M}")
    sys.path.insert(0, str(pathlib.Path(riskquant.__file__).resolve().parents[2]
                           / "tests" / "solutions" / "risk_run"))
    from _certfix import (CFGS, FIDS, K, K_TOTAL, M, POSITIONS, RSC, S, T, UIDS, VALS,
                          calculator, derived, matrix, parametric_data, v0)
    from _certfix import inputs as _certfix_inputs

    def inputs(model, batch_size, seed=7):
        return _certfix_inputs(model=model, batch_size=batch_size, seed=seed)

    def h(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(
            f"{{a.dtype.str}}|{{a.shape}}|".encode() + a.tobytes()).hexdigest()

    def fixture_digest(data):
        # Content digest of the construction data this child actually priced.
        if isinstance(data, ParametricRiskData):
            return hash_parametric_risk_data(data)
        return hash_risk_factor_matrix(data)

    def blas_threads():
        # Evidence only (never a pass criterion): the BLAS pools as loaded.
        try:
            import threadpoolctl
        except ImportError:
            return "threadpoolctl unavailable"
        return [f"{{p['internal_api']}}:{{p['num_threads']}}"
                for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"]

    def reporting_route(inp):
        # D085: settlement currencies from the RAW positions frame (U order);
        # FX routes by name from every configured FX pair on the run axis.
        settle = dict(zip(POSITIONS["uid"].to_list(), POSITIONS["settlement_ccy"].to_list()))
        pairs = np.asarray(RSC.fx_spot_matrix)
        fx_ids = np.unique(pairs[pairs != int(NO_FX_RF_REQUIRED)])
        axis_ids = np.asarray(inp["required_factor_ids"])
        rows = np.flatnonzero(np.isin(axis_ids, fx_ids))
        names = {{int(v): k for k, v in RSC.rel_name_to_rf_id.items()}}
        plan = build_fx_conversion_plan(
            settlement_ccys=[str(settle[int(u)]) for u in UIDS.tolist()],
            reporting_ccy=inp["reporting"].reporting_ccy,
            fx_factor_names=[names[int(axis_ids[r])] for r in rows])
        return plan, rows

    def capture(model, batch_size):
        inp = inputs(model, batch_size)
        rr = run_scenario_var_es(**inp)                      # OFFICIAL API
        run_axis = derive_run_factor_axis(
            valuation_context=inp["valuation_context"],
            reporting_ccy=inp["reporting"].reporting_ccy)
        k_run, k_val = int(run_axis.factor_ids.shape[0]), int(run_axis.num_valuation_factors)
        # Reference rebuild through the D088 blocks the run composes: the
        # authored terms compiled over the full axis, lowered, bound to the
        # world's (d, tau, h) columns, streamed, then X = sum beta Y, T^-1_kappa.
        ctx = inp["valuation_context"]
        operator = lower_axis_operator(
            model=compile_proxy_model(
                authored_terms=inp["proxy_terms"],
                static_config=ctx.static_config,
                as_of_date=ord_to_date(int(ctx.uix_mapper.as_of_ord32)),
                run_axis_rf_ids=run_axis.factor_ids),
            axis_rf_ids=run_axis.factor_ids,
            axis_shift_type_ids=run_axis.shift_type_ids)
        binding = bind_operator_to_world(operator, inp["scenario_config"], inp["data"])
        gen = build_scenario_generator(inp["scenario_config"], data=inp["data"])
        shocks, chunks = [], []
        for b in gen.iter_batches(driver_rf_ids=binding.request_driver_rf_ids,
                chart_ids=binding.request_chart_ids, batch_size=batch_size):
            out = np.empty((k_run, b.shocks.shape[0]), dtype=np.float64, order="C")
            apply_linear_terms_to_base_levels(
                base_levels=inp["base_levels"], row_term_offsets=operator.row_term_offsets,
                term_column_uix=binding.term_column_uix, term_loadings=operator.term_loadings,
                target_shift_types=operator.target_shift_type_ids, shocks=b.shocks, out=out)
            shocks.append(np.array(b.shocks, copy=True))
            chunks.append(out)
        stream = np.vstack(shocks)
        levels = np.ascontiguousarray(np.hstack(chunks))
        # Taylor lives on the valuation prefix of the run axis (D082 as amended).
        axis = FactorAxis.from_keys(
            pl.Series("factor_id", run_axis.factor_ids[:k_val].tolist(), dtype=pl.Int32))
        settlement = inp["taylor_derived"].calculator.run(
            axis.levels(base_levels=inp["base_levels"][:k_val], shocked_levels=levels[:k_val]))
        plan, fx_rows = reporting_route(inp)
        conv = convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=np.ascontiguousarray(settlement.pnl, dtype=np.float64),
            fx_levels=np.ascontiguousarray(levels[fx_rows], dtype=np.float64),
            plan=plan,
            base_settlement_values=v0(),   # P1-B: the real base values of U
            base_fx_levels=np.ascontiguousarray(
                inp["base_levels"][fx_rows], dtype=np.float64))
        # The inspection is independent of the run's derived route; it must
        # reproduce the RiskRun bitwise, or one of the two routes is wrong.
        metrics = compute_var_es(np.ascontiguousarray(
            np.vstack([conv.converted_pnl, conv.portfolio_pnl[None, :]])),
            inp["confidence_level"])
        return {{
            "requested_workers": int(numba.config.NUMBA_NUM_THREADS),
            "actual_workers": int(numba.get_num_threads()),
            "layer": str(numba.threading_layer()),
            "fixture_digest": fixture_digest(inp["data"]),
            "blas_threads": blas_threads(),
            "scenario_stream": h(stream),
            "run_axis": {{"K_val": k_val, "K_rep": k_run - k_val, "K_run": k_run,
                         "bound": bool(np.array_equal(run_axis.factor_ids,
                                                      inp["required_factor_ids"]))}},
            "shocked_levels": h(levels),
            "trade_pnl": h(np.asarray(settlement.pnl)),
            "portfolio_pnl": h(np.asarray(conv.portfolio_pnl)),
            "inspection_matches_run": (
                [float(v).hex() for v in metrics.var] == [float(v).hex() for v in rr.var]
                and [float(v).hex() for v in metrics.es] == [float(v).hex() for v in rr.es]),
            "var": [float(v).hex() for v in rr.var],
            "es": [float(v).hex() for v in rr.es],
            "manifest_identity": rr.manifest.identity(),
            "riskrun_identity": rr.identity,
            "reproducible": bool(rr.reproducible),
        }}
    """
)


def _capture_script(model: str, batch_size: int, dims: dict[str, int]) -> str:
    return _FIXTURE.format(**dims) + textwrap.dedent(
        f"""
        print(json.dumps(capture({model!r}, {batch_size})))
        """
    )


def _resolve_batch(b: Any, total_scenarios: int) -> int:
    if b == "S":
        return total_scenarios
    if b == "S+1":
        return total_scenarios + 1
    return int(b)


def _first_divergence(a: dict, b: dict) -> str | None:
    for key in DIVERGENCE_ORDER:
        if a.get(key) != b.get(key):
            return key
    return None


# --------------------------------------------------------------------------
# certification checks
# --------------------------------------------------------------------------


class Certificate:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}

    def record(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        display_detail: str | None = None,
    ) -> bool:
        """Record full evidence while allowing concise console formatting."""
        self.checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

        mark = "PASS" if passed else "FAIL"

        if display_detail is None:
            print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        else:
            print(f"  [{mark}] {name}")
            for raw_line in display_detail.splitlines():
                wrapped = textwrap.wrap(
                    raw_line,
                    width=72,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if not wrapped:
                    print("      ")
                else:
                    for line in wrapped:
                        print(f"      {line}")

        return passed

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


def certify_threads(cert: Certificate, threads: list[int]) -> dict[int, dict]:
    print("\n[1] Numba worker execution proof (distinct ids inside a real prange)")
    observed: dict[int, dict] = {}
    for t in threads:
        probe = _child(_WORKER_PROBE, {"NUMBA_NUM_THREADS": str(t)})
        if "__error__" in probe:
            cert.record(f"worker probe threads={t}", False, probe["__error__"][:120])
            continue
        observed[t] = probe
        ok = probe["configured"] == t and probe["observed_workers"] == t
        cert.record(
            f"threads={t} genuinely executed",
            ok,
            f"configured={probe['configured']} observed_workers="
            f"{probe['observed_workers']} layer={probe['layer']}",
        )
    cert.evidence["worker_probe"] = observed
    return observed


def certify_matrix(
    cert: Certificate, threads: list[int], dims: dict[str, int]
) -> dict[str, Any]:
    print("\n[2] Thread matrix — official API, all models, divergence order")
    ref_by_model: dict[str, dict] = {}
    detail: dict[str, Any] = {}
    for model in MODELS:
        base = _child(
            _capture_script(model, 64, dims), {"NUMBA_NUM_THREADS": str(threads[0])}
        )
        if "__error__" in base:
            cert.record(f"{model} baseline", False, base["__error__"][:140])
            continue
        ref_by_model[model] = base
        axis = base["run_axis"]
        cert.record(
            f"{model}: inspection reproduces the RiskRun on the canonical run axis",
            bool(base["inspection_matches_run"]) and bool(axis["bound"]),
            f"K_val={axis['K_val']} K_rep={axis['K_rep']} K_run={axis['K_run']}; "
            "reporting route rebuilt from the trade record (D085)",
        )
        cert.evidence.setdefault("run_axis", axis)
        for t in threads[1:]:
            got = _child(
                _capture_script(model, 64, dims), {"NUMBA_NUM_THREADS": str(t)}
            )
            if "__error__" in got:
                cert.record(f"{model} threads={t}", False, got["__error__"][:140])
                continue
            if got["actual_workers"] != t:
                cert.record(f"{model} threads={t}", False, "worker count clamped")
                continue
            div = _first_divergence(base, got)
            cert.record(
                f"{model}: threads {threads[0]} vs {t} identical",
                div is None,
                "" if div is None else f"FIRST DIVERGENCE at {div}",
            )
            detail[f"{model}:threads{t}"] = div or "identical"

    print("\n[3] Batch matrix at the highest genuinely executing thread count")
    top = threads[-1]
    for model in MODELS:
        base = _child(_capture_script(model, 64, dims), {"NUMBA_NUM_THREADS": str(top)})
        if "__error__" in base:
            continue
        for b in BATCH_MATRIX:
            bs = _resolve_batch(b, dims["S"])
            got = _child(
                _capture_script(model, bs, dims), {"NUMBA_NUM_THREADS": str(top)}
            )
            if "__error__" in got:
                cert.record(f"{model} bs={b}", False, got["__error__"][:140])
                continue
            div = _first_divergence(base, got)
            cert.record(
                f"{model}: batch_size {b} identical @ {top} threads",
                div is None,
                "" if div is None else f"FIRST DIVERGENCE at {div}",
            )
            detail[f"{model}:bs{b}"] = div or "identical"
    cert.evidence["matrix"] = detail
    return detail


#: Must equal `_certfix.PARAMETRIC_DATA_PATH_ENV`.
_PARAMETRIC_DATA_PATH_ENV = "RQ_CERT_PARAMETRIC_DATA"


def _freeze_parametric_fixture(dims: dict[str, int], path: str, top: int) -> dict:
    """Build the parametric construction data ONCE, OUTSIDE every attacked
    configuration, and persist its exact bytes at `path`."""
    script = _FIXTURE.format(**dims) + textwrap.dedent(
        f"""
        from _certfix import freeze_parametric_data
        print(json.dumps({{"digest": freeze_parametric_data({path!r})}}))
        """
    )
    return _child(script, {"NUMBA_NUM_THREADS": str(top)})


def _blas_record(r: dict) -> dict[str, Any]:
    """Per-configuration evidence: every certified object, identities included."""
    return {
        **{k: r[k] for k in ("fixture_digest", *DIVERGENCE_ORDER[:4])},
        "var_digest": hashlib.sha256(json.dumps(r["var"]).encode()).hexdigest(),
        "es_digest": hashlib.sha256(json.dumps(r["es"]).encode()).hexdigest(),
        "portfolio_var": r["var"][-1],
        "portfolio_es": r["es"][-1],
        "manifest_identity": r["manifest_identity"],
        "riskrun_identity": r["riskrun_identity"],
        "blas_threads": r["blas_threads"],
    }


def certify_parametric_blas(
    cert: Certificate, threads: list[int], dims: dict[str, int]
) -> None:
    """
    Same economic input, different BLAS execution configuration.

    The parametric construction data is itself BLAS-sensitive to build, so it is
    frozen once before the attack and every child (baseline included) LOADS
    those exact bytes. A child whose data digest differs from the frozen one is
    a certification construction failure, reported before any result
    comparison; only then is the unchanged divergence order applied.
    """
    print("\n[4] Parametric BLAS attack (np.matmul in simlab/scenarios/parametric.py)")
    top = threads[-1]
    with tempfile.TemporaryDirectory(prefix="rq_blas_fixture_") as tmp:
        path = str(Path(tmp) / "parametric_data.npz")
        frozen = _freeze_parametric_fixture(dims, path, top)
        if "__error__" in frozen:
            cert.record(
                "parametric BLAS fixture freeze", False, frozen["__error__"][:140]
            )
            return
        pinned = {"NUMBA_NUM_THREADS": str(top), _PARAMETRIC_DATA_PATH_ENV: path}
        cert.evidence["parametric_blas_fixture"] = {
            "digest": frozen["digest"],
            "frozen_under": {"NUMBA_NUM_THREADS": str(top)},
        }
        base = _child(_capture_script("parametric", 64, dims), pinned)
        if "__error__" in base:
            cert.record("parametric BLAS baseline", False, base["__error__"][:140])
            return
        if base["fixture_digest"] != frozen["digest"]:
            cert.record(
                "parametric BLAS baseline fixture isolation",
                False,
                f"CERTIFICATION CONSTRUCTION FAILURE: baseline priced data "
                f"{base['fixture_digest'][:16]}..., frozen {frozen['digest'][:16]}...",
            )
            return
        results: dict[str, str] = {}
        records: dict[str, Any] = {"baseline": _blas_record(base)}
        for env in BLAS_ENVS:
            got = _child(_capture_script("parametric", 64, dims), {**pinned, **env})
            label = ",".join(f"{k}={v}" for k, v in env.items())
            if "__error__" in got:
                cert.record(f"parametric under {label}", False, got["__error__"][:140])
                continue
            records[label] = _blas_record(got)
            if got["fixture_digest"] != frozen["digest"]:
                cert.record(
                    f"parametric identical under {label}",
                    False,
                    f"CERTIFICATION CONSTRUCTION FAILURE: child priced data "
                    f"{got['fixture_digest'][:16]}..., "
                    f"frozen {frozen['digest'][:16]}...",
                )
                results[label] = "fixture_digest"
                continue
            div = _first_divergence(base, got)
            cert.record(
                f"parametric identical under {label}",
                div is None,
                "" if div is None else f"FIRST DIVERGENCE at {div}",
            )
            results[label] = div or "identical"
    cert.evidence["parametric_blas"] = results
    cert.evidence["parametric_blas_records"] = records


def certify_fresh_process(
    cert: Certificate, threads: list[int], dims: dict[str, int]
) -> None:
    print("\n[5] Fresh-process reproduction")
    top = threads[-1]
    a = _child(_capture_script("sbb", 64, dims), {"NUMBA_NUM_THREADS": str(top)})
    b = _child(_capture_script("sbb", 64, dims), {"NUMBA_NUM_THREADS": str(top)})
    if "__error__" in a or "__error__" in b:
        cert.record("fresh-process reproduction", False, "child failed")
        return
    div = _first_divergence(a, b)
    cert.record(
        f"two fresh processes identical @ {top} threads",
        div is None,
        "" if div is None else f"FIRST DIVERGENCE at {div}",
    )
    cert.evidence["fresh_process"] = {"divergence": div or "none", "threads": top}


_CONTRACTS = textwrap.dedent(
    """
    import json
    import numpy as np

    results = {}

    # ---- bypass audit -----------------------------------------------------
    import pathlib, re
    root = pathlib.Path(__SRC__) / "riskquant"
    producers = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"compute_var_es\\(", text) and "def compute_var_es" not in text:
            producers.append(str(p.relative_to(root)))
    results["var_es_producers"] = sorted(producers)

    # ---- environment gate / seed / stale / tamper -------------------------
    from riskquant.solutions.run_identity.api import (
        EnvironmentMismatchError, InputMismatchError)
    from riskquant.solutions.run_identity.schemas import RunManifest
    from riskquant.solutions.risk_run.api import run_scenario_var_es
    from riskquant.solutions.risk_run.schemas import (
        CompositionContractError, NonReproducibleRunError, RiskRun)
    from riskquant.solutions.scenario_models.api import parse_scenario_model_config
    from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix

    base = inputs("sbb", 64)
    ref = run_scenario_var_es(**base)
    _NAMES = dict((int(v), k) for k, v in RSC.rel_name_to_rf_id.items())

    # environment gate: one changed field must stop the run pre-numerics
    env_results = {}
    for field in ("riskquant_version", "python_version", "numpy_version", "numba_version"):
        payload = json.loads(ref.manifest.to_json())
        payload["engine"][field] = "0.0.0-not-this"
        foreign = RunManifest.model_validate(payload)
        try:
            run_scenario_var_es(**inputs("sbb", 64), reproduce=foreign)
            env_results[field] = "ACCEPTED (defect)"
        except EnvironmentMismatchError:
            env_results[field] = "rejected"
        except Exception as exc:
            env_results[field] = f"other:{type(exc).__name__}"
    results["environment_gate"] = env_results

    # seed policy
    seed_results = {}
    for label, value in [("0", 0), ("1", 1), ("2**70", 2 ** 70), ("2**130", 2 ** 130),
                         ("-1", -1), ("True", True), ("1.0", 1.0), ('"1"', "1")]:
        try:
            parse_scenario_model_config({"model": "sbb", "horizon": 1,
                                          "total_scenarios": 16, "seed": value})
            seed_results[label] = "accepted"
        except Exception as exc:
            seed_results[label] = f"rejected:{type(exc).__name__}"
    try:
        run_scenario_var_es(**inputs("sbb", 64, seed=None))
        seed_results["None+reproducible"] = "ACCEPTED (defect)"
    except NonReproducibleRunError:
        seed_results["None+reproducible"] = "NonReproducibleRunError"
    expl = run_scenario_var_es(**inputs("sbb", 64, seed=None), mode="exploratory")
    seed_results["None+exploratory"] = f"reproducible={expl.reproducible}"
    results["seed_policy"] = seed_results

    # stale-input substitution
    stale = {}
    def _remat(m, **over):
        parts = {"dates": m.dates, "factor_ids": m.factor_ids, "values": m.values}
        parts.update(over)
        return RiskFactorMatrix(**parts)
    revised = np.array(base["data"].values, copy=True)
    revised[30, 1] *= 1.01
    subs = {
        "historical_value": {"data": _remat(base["data"],
                                values=np.asfortranarray(revised))},
        "factor_order": {"data": _remat(base["data"], values=np.asfortranarray(
            np.array(base["data"].values)[:, list(range(1, K_TOTAL)) + [0]]))},
        # A different Proxy V1 operator: the first axis factor driven by its
        # own column with loading 0.9 (D088; the model is result-defining).
        "proxy_model": {"proxy_terms": (NodeProxyTerm(
            target_name=_NAMES[int(FIDS[0])], driver_name=_NAMES[int(FIDS[0])],
            driver_chart=ShiftType(int(RSC.rf_id_to_shift_type_id[int(FIDS[0])])),
            loading=0.9),)},
        "shift_types": {"required_shift_type_id":
            np.full(K, 2, dtype=np.int8)},
        "base_levels": {"base_levels": base["base_levels"] * 1.001},
        "confidence": {"confidence_level": 0.95},
        "compiled_book": {"taylor_derived": derived(calculator(variant=1))},
    }
    for name, over in subs.items():
        inp = dict(inputs("sbb", 64)); inp.update(over)
        try:
            run_scenario_var_es(**inp, reproduce=ref.manifest)
            stale[name] = "ACCEPTED (defect)"
        except InputMismatchError:
            stale[name] = "InputMismatchError"
        except CompositionContractError:
            stale[name] = "CompositionContractError"   # P1-B: refused by the run context
        except Exception as exc:
            stale[name] = f"other:{type(exc).__name__}"
    results["stale_input"] = stale

    # identity mutation: result AND identity must move
    mut = {}
    for name, over in subs.items():
        if name == "compiled_book":
            continue
        inp = dict(inputs("sbb", 64)); inp.update(over)
        if name == "base_levels":
            # P1-A: a new base is a new Taylor representation; re-certify it there
            inp["taylor_derived"] = derived(calculator(base=inp["base_levels"]))
        try:
            got = run_scenario_var_es(**inp)
        except CompositionContractError:
            # P1-B: shift types are owned by the static configuration; the
            # substitution is refused outright, which is stronger than a move.
            mut[name] = {"refused": "CompositionContractError"}
            continue
        changed_result = [float(v).hex() for v in got.var] != [float(v).hex() for v in ref.var]
        mut[name] = {"result_changed": changed_result,
                      "identity_changed": got.identity != ref.identity}
    results["identity_mutation"] = mut

    # operational neutrality: batch size must not move identity
    alt = run_scenario_var_es(**inputs("sbb", 7))
    results["identity_operational_neutral"] = (alt.identity == ref.identity)

    # tamper
    tamper = {}
    try:
        RiskRun(position_uids=ref.position_uids, var=ref.var, es=ref.es,
                reporting_ccy=ref.reporting_ccy, manifest=ref.manifest,
                identity="0" * 64, reproducible=True,
                pnl_provenance=ref.pnl_provenance)
        tamper["forged_identity_ctor"] = "ACCEPTED (defect)"
    except Exception as exc:
        tamper["forged_identity_ctor"] = f"rejected:{type(exc).__name__}"
    for field, value in [("identity", "0" * 64), ("var_hex", ["0x0p+0"]),
                         ("reporting_ccy", "ZZZ")]:
        payload = json.loads(ref.to_json()); payload[field] = value
        try:
            RiskRun.from_json(json.dumps(payload))
            tamper[f"forged_{field}"] = "ACCEPTED (defect)"
        except Exception as exc:
            tamper[f"forged_{field}"] = f"rejected:{type(exc).__name__}"
    results["tamper"] = tamper

    print(json.dumps(results))
    """
)


def certify_contracts(cert: Certificate, dims: dict[str, int]) -> None:
    print("\n[6] Contract gates")

    script = _FIXTURE.format(**dims) + _CONTRACTS.replace("__SRC__", repr(str(_SRC)))
    res = _child(script, {"NUMBA_NUM_THREADS": "2"})

    if "__error__" in res:
        cert.record(
            "contract gates",
            False,
            res["__error__"][:400],
        )
        return

    # ------------------------------------------------------------------
    # environment gate
    # ------------------------------------------------------------------
    env = res["environment_gate"]
    env_ok = all(v == "rejected" for v in env.values())

    env_rejected = [k for k, v in env.items() if v == "rejected"]
    env_unexpected = {k: v for k, v in env.items() if v != "rejected"}

    env_display = [
        f"{len(env_rejected)}/{len(env)} changed fields rejected",
    ]
    if env_rejected:
        env_display.append("fields: " + ", ".join(env_rejected))
    if env_unexpected:
        env_display.append("unexpected: " + json.dumps(env_unexpected, sort_keys=True))

    cert.record(
        "Environment gate",
        env_ok,
        json.dumps(env),
        "\n".join(env_display),
    )

    # ------------------------------------------------------------------
    # seed policy
    # ------------------------------------------------------------------
    seed = res["seed_policy"]

    seed_ok = (
        seed["0"] == "accepted"
        and seed["1"] == "accepted"
        and seed["2**70"] == "accepted"
        and seed["2**130"] == "accepted"
        and seed["-1"].startswith("rejected")
        and seed["True"].startswith("rejected")
        and seed["1.0"].startswith("rejected")
        and seed['"1"'].startswith("rejected")
        and seed["None+reproducible"] == "NonReproducibleRunError"
        and seed["None+exploratory"] == "reproducible=False"
    )

    accepted = [k for k, v in seed.items() if v == "accepted"]
    rejected = [k for k, v in seed.items() if v.startswith("rejected")]

    seed_display = [
        "accepted: " + ", ".join(accepted),
        "rejected: " + ", ".join(rejected),
        (f"None + reproducible -> {seed['None+reproducible']}"),
        (f"None + exploratory -> {seed['None+exploratory']}"),
    ]

    cert.record(
        "Seed policy",
        seed_ok,
        json.dumps(seed),
        "\n".join(seed_display),
    )

    # ------------------------------------------------------------------
    # stale input
    # ------------------------------------------------------------------
    stale = res["stale_input"]

    stale_ok = all(
        v in ("InputMismatchError", "CompositionContractError") for v in stale.values()
    )

    by_error: dict[str, list[str]] = {}
    for name, outcome in stale.items():
        by_error.setdefault(outcome, []).append(name)

    stale_display = [
        f"{len(stale)}/{len(stale)} substitutions rejected before numerics"
    ]

    for error_type, names in by_error.items():
        stale_display.append(f"{error_type}: {', '.join(names)}")

    cert.record(
        "Stale input gate",
        stale_ok,
        json.dumps(stale),
        "\n".join(stale_display),
    )

    # ------------------------------------------------------------------
    # identity mutation
    # ------------------------------------------------------------------
    mut = res["identity_mutation"]

    mut_ok = all(v.get("refused") or v["identity_changed"] for v in mut.values())

    checked = [(name, value) for name, value in mut.items() if not value.get("refused")]
    identity_moved = [name for name, value in checked if value["identity_changed"]]
    refused = [
        (name, value["refused"]) for name, value in mut.items() if value.get("refused")
    ]
    weak = [name for name, value in checked if not value["result_changed"]]

    identity_display = [
        f"{len(identity_moved)}/{len(checked)} non-refused mutations changed identity"
    ]

    if refused:
        for name, reason in refused:
            identity_display.append(f"{name}: refused by {reason}")

    if weak:
        identity_display.append(
            "conservative over-separation: "
            + ", ".join(f"{name} (identity moved; result unchanged)" for name in weak)
        )

    cert.record(
        "Identity mutation",
        mut_ok,
        (
            "conservative over-separation (identity moved, result did not): "
            + ",".join(weak)
        )
        if weak
        else "",
        "\n".join(identity_display),
    )

    # ------------------------------------------------------------------
    # operational neutrality
    # ------------------------------------------------------------------
    batch_identity_ok = bool(res["identity_operational_neutral"])

    cert.record(
        "Batch-size identity",
        batch_identity_ok,
        "",
        "identity invariant to batch size",
    )

    # ------------------------------------------------------------------
    # tamper protection
    # ------------------------------------------------------------------
    tamper = res["tamper"]
    tamper_ok = all("rejected" in v for v in tamper.values())

    rejected_tamper = [
        (name, outcome) for name, outcome in tamper.items() if "rejected" in outcome
    ]
    unexpected_tamper = {
        name: outcome for name, outcome in tamper.items() if "rejected" not in outcome
    }

    tamper_display = [f"{len(rejected_tamper)}/{len(tamper)} forged objects rejected"]

    if unexpected_tamper:
        tamper_display.append(
            "unexpected: "
            + json.dumps(
                unexpected_tamper,
                sort_keys=True,
            )
        )

    cert.record(
        "RiskRun tamper protection",
        tamper_ok,
        json.dumps(tamper),
        "\n".join(tamper_display),
    )

    # ------------------------------------------------------------------
    # bypass audit
    # ------------------------------------------------------------------
    producers = res["var_es_producers"]
    expected = {
        "solutions/risk_run/api.py",
        "credit/aggregate/loss_metrics.py",
    }
    found = {p.replace(os.sep, "/") for p in producers}
    bypass_ok = found == expected

    if bypass_ok:
        bypass_display = (
            "public market entry point: solutions/risk_run/api.py\n"
            "separate credit consumer: credit/aggregate/loss_metrics.py"
        )
    else:
        bypass_display = "unexpected producers: " + ", ".join(sorted(found))

    cert.record(
        "Bypass audit",
        bypass_ok,
        f"producers={sorted(found)}",
        bypass_display,
    )

    cert.evidence["contracts"] = res


def measure_performance(threads: list[int], dims: dict[str, int]) -> list[dict]:
    print("\n[7] Performance (informational)")
    rows = []
    for t in threads:
        script = _FIXTURE.format(**dims) + textwrap.dedent(
            """
            import time, tracemalloc
            from riskquant.solutions.run_identity.api import verify_run_inputs
            inp = inputs("sbb", 256)
            rr = run_scenario_var_es(**inp)
            t0 = time.perf_counter(); rr = run_scenario_var_es(**inp)
            dt = time.perf_counter() - t0
            t0 = time.perf_counter(); run_scenario_var_es(**inp, reproduce=rr.manifest)
            dtv = time.perf_counter() - t0
            t0 = time.perf_counter(); js = rr.to_json(); dts = time.perf_counter() - t0
            tracemalloc.start(); run_scenario_var_es(**inp)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            print(json.dumps({"threads": int(numba.get_num_threads()),
                              "runtime_s": dt, "verify_s": dtv, "serialize_s": dts,
                              "peak_mb": peak / 1e6, "artifact_kb": len(js) / 1000,
                              "scen_per_s": S / dt}))
            """
        )
        row = _child(script, {"NUMBA_NUM_THREADS": str(t)})
        if "__error__" not in row:
            rows.append(row)
            print(
                f"  threads={row['threads']}: {row['runtime_s'] * 1000:7.1f} ms | "
                f"{row['scen_per_s'] / 1000:6.1f} kscen/s | verify "
                f"{row['verify_s'] * 1000:6.1f} ms | peak {row['peak_mb']:6.1f} MB"
            )
    return rows


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default="simlab_multicore_certificate.json")
    parser.add_argument("--factors", type=int, default=250)
    parser.add_argument("--observations", type=int, default=600)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument(
        "--allow-oversubscription",
        action="store_true",
        help="Record worker counts above the effective CPU count. They prove "
        "multi-THREAD execution, never multi-CORE hardware.",
    )
    args = parser.parse_args()

    dims = {
        "K": args.factors,
        "T": args.observations,
        "S": args.scenarios,
        "M": args.positions,
    }

    cpus, source = effective_cpus()
    print("=" * 78)
    print("SimLab multi-core release certification")
    print("=" * 78)
    print(f"  effective CPUs      : {cpus}  (via {source})")
    print(f"  os.cpu_count()      : {os.cpu_count()}")
    print(f"  platform            : {platform.platform()}")
    print(f"  python              : {platform.python_version()}")
    print(
        f"  fixture             : K={dims['K']} T={dims['T']} "
        f"S={dims['S']} M={dims['M']}"
    )

    multi_core_available = cpus >= 2
    print(f"\n  MULTI-CORE HARDWARE AVAILABLE : {multi_core_available}")

    certificate: dict[str, Any] = {
        "classification": None,
        "multi_core_hardware_available": multi_core_available,
        "numba_multi_thread_execution_verified": False,
        "multi_core_determinism_certified": False,
        "hardware": {
            "effective_cpus": cpus,
            "cpu_source": source,
            "os_cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "fixture": dims,
        "checks": [],
        "evidence": {},
        "performance": [],
    }

    if not multi_core_available:
        # A single-CPU host can still spawn workers; record that fact WITHOUT
        # letting it masquerade as hardware certification.
        cert = Certificate()
        probe = _child(_WORKER_PROBE, {"NUMBA_NUM_THREADS": "2"})
        if "__error__" not in probe and probe.get("observed_workers", 0) >= 2:
            certificate["numba_multi_thread_execution_verified"] = True
            certificate["evidence"]["worker_probe_single_cpu"] = probe
            print(
                f"  NUMBA MULTI-THREAD EXECUTION VERIFIED : True "
                f"({probe['observed_workers']} workers, layer={probe['layer']})"
            )
            print("    -> workers time-slice ONE cpu; this is NOT multi-core evidence")
        certificate["classification"] = "NOT EXECUTED"
        certificate["reason"] = (
            f"effective CPU availability is {cpus} (need >= 2). Numba worker "
            "threads can still be created on one CPU, which proves multi-THREAD "
            "execution only. The multi-core determinism gate remains OPEN."
        )
        print("\n" + "=" * 78)
        print("RESULT: NOT EXECUTED — multi-core determinism gate remains OPEN")
        print("=" * 78)
        print(f"  {certificate['reason']}")
        Path(args.json_out).write_text(json.dumps(certificate, indent=2))
        print(f"\n  JSON certificate written to {args.json_out}")
        return 2

    threads = [t for t in THREAD_MATRIX if t <= cpus or args.allow_oversubscription]
    if len(threads) < 2:
        threads = [1, 2]
    print(f"  thread matrix       : {threads}")

    cert = Certificate()
    observed = certify_threads(cert, threads)
    genuinely_multi = any(v.get("observed_workers", 0) >= 2 for v in observed.values())
    certificate["numba_multi_thread_execution_verified"] = genuinely_multi
    cert.record(
        "NUMBA MULTI-THREAD EXECUTION VERIFIED",
        genuinely_multi,
        ">= 2 distinct worker ids inside a real prange",
    )

    executing = [t for t in threads if observed.get(t, {}).get("observed_workers") == t]
    if len(executing) < 2:
        certificate["classification"] = "NOT EXECUTED"
        certificate["reason"] = (
            "fewer than two thread configurations genuinely executed; the "
            "requested worker counts were clamped."
        )
        certificate["checks"] = cert.checks
        Path(args.json_out).write_text(json.dumps(certificate, indent=2))
        print("\nRESULT: NOT EXECUTED — worker counts clamped")
        return 2

    certify_matrix(cert, executing, dims)
    certify_parametric_blas(cert, executing, dims)
    certify_fresh_process(cert, executing, dims)
    certify_contracts(cert, dims)
    certificate["performance"] = measure_performance(executing, dims)

    certificate["checks"] = cert.checks
    certificate["evidence"].update(cert.evidence)
    certificate["threads_executed"] = executing

    passed = not cert.failures and genuinely_multi and multi_core_available
    certificate["multi_core_determinism_certified"] = passed
    certificate["classification"] = "PASS" if passed else "FAIL"

    print("\n" + "=" * 78)
    if passed:
        print("RESULT: PASS — MULTI-CORE DETERMINISM CERTIFIED")
        print("=" * 78)
        print(f"  effective CPUs={cpus}; thread configurations executed: {executing}")
        print("  All official-path objects bitwise identical across threads, batch")
        print("  sizes, BLAS settings and processes; contract gates all passed.")
    else:
        print("RESULT: FAIL — first failing checks below")
        print("=" * 78)
        for failure in cert.failures[:10]:
            print(f"  - {failure['check']}: {failure['detail']}")
    Path(args.json_out).write_text(json.dumps(certificate, indent=2))
    print(f"\n  JSON certificate written to {args.json_out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
