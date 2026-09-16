# scripts/release/certify_simlab_multicore.py
"""
SimLab multi-core release certification gate.

Run on a genuinely multi-core machine:

    python scripts/release/certify_simlab_multicore.py

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
import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    try:
        return len(os.sched_getaffinity(0)), "sched_getaffinity"
    except (AttributeError, OSError):
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
#: recreated.
_FIXTURE = textwrap.dedent(
    """
    import hashlib, json
    import numpy as np, polars as pl, numba

    from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
    from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
    from riskquant.reporting.fx_conversion import (
        build_fx_conversion_plan, convert_scenario_pnl_to_reporting_currency)
    from riskquant.sensitivity_pnl.conventions import DisplacementConvention
    from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
    from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
    from riskquant.solutions.risk_run.api import run_scenario_var_es
    from riskquant.solutions.risk_run.schemas import ReportingSpec
    from riskquant.solutions.scenario_models.api import (
        build_scenario_generator, parse_scenario_model_config)
    from riskquant.solutions.sensitivity_pnl.api import (
        FactorAxis, SensitivityBookSpec, SensitivityPnlCalculator)

    K, T, S, M = {K}, {T}, {S}, {M}
    FIDS = np.arange(1, K + 1, dtype=np.int32)

    def _vals():
        r = np.random.default_rng(0)
        v = 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, (T, K)), axis=0))
        v[:, K - 1] = 0.9 * np.exp(np.cumsum(r.normal(0, 0.005, T)))
        return np.asfortranarray(v)

    VALS = _vals()

    def matrix():
        return RiskFactorMatrix(dates=np.arange(T, dtype=np.int32),
                                factor_ids=FIDS, values=VALS)

    def parametric_data():
        r = np.random.default_rng(1)
        a = r.normal(0, 1, (K, K))
        return ParametricRiskData(
            factor_ids=FIDS,
            cholesky_matrix=np.ascontiguousarray(
                np.linalg.cholesky(a @ a.T / K + np.eye(K) * 0.05)),
            volatilities=np.ascontiguousarray(np.full(K, 0.02)))

    def calculator(variant=0):
        r = np.random.default_rng(2 + variant)
        pos = np.repeat(np.arange(1, M + 1), 6)
        fac = r.choice(FIDS, size=M * 6)
        df = pl.DataFrame({{
            "position_id": pos.tolist(), "factor_id": fac.tolist(),
            "first_order": r.normal(0, 100, M * 6).tolist(),
            "second_order": r.normal(0, 5, M * 6).tolist(),
        }}).unique(subset=["position_id", "factor_id"]).sort(["position_id", "factor_id"])
        meta = pl.DataFrame({{
            "factor_id": FIDS.tolist(),
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * K,
            "quote_base": [1.0] * K}})
        return SensitivityPnlCalculator.compile(SensitivityBookSpec(
            sensitivities=df, factor_meta=meta,
            factor_axis=FactorAxis.from_keys(FIDS.tolist()),
            position_uids=list(range(1, M + 1))))

    CFGS = {{
        "historical": {{"model": "historical", "horizon": 1}},
        "filtered_historical": {{"model": "filtered_historical", "horizon": 1}},
        "sbb": {{"model": "sbb", "horizon": 1, "total_scenarios": S,
                 "expected_block_size": 20.0, "seed": 7}},
        "filtered_bootstrap": {{"model": "filtered_bootstrap", "horizon": 1,
                                "total_scenarios": S, "seed": 7}},
        "parametric": {{"model": "parametric", "total_scenarios": S,
                        "distribution": {{"kind": "student_t", "df": 5.0}}, "seed": 7}},
    }}

    def inputs(model, batch_size, seed=7):
        cfg = dict(CFGS[model])
        if "seed" in cfg:
            cfg["seed"] = seed
        settle = tuple(("EUR" if i % 2 else "USD") for i in range(M))
        return dict(
            scenario_config=parse_scenario_model_config(cfg),
            data=parametric_data() if model == "parametric" else matrix(),
            required_factor_ids=FIDS,
            required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * K, np.int8),
            source_factor_ids=FIDS.copy(),
            shock_scale=np.ones(K, np.float32),
            base_levels=np.ascontiguousarray(VALS[-1].astype(np.float64)),
            calculator=calculator(),
            reporting=ReportingSpec(
                reporting_ccy="USD", settlement_ccys=settle,
                fx_factor_names=("USDEUR",),
                fx_factor_rows=np.array([K - 1], np.int32)),
            confidence_level=0.99, batch_size=batch_size)

    def h(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(
            f"{{a.dtype.str}}|{{a.shape}}|".encode() + a.tobytes()).hexdigest()

    def capture(model, batch_size):
        inp = inputs(model, batch_size)
        rr = run_scenario_var_es(**inp)                      # OFFICIAL API
        gen = build_scenario_generator(inp["scenario_config"], data=inp["data"])
        shocks, chunks = [], []
        for b in gen.iter_batches(
                required_shift_type_id=inp["required_shift_type_id"],
                source_factor_ids=inp["source_factor_ids"],
                shock_scale=inp["shock_scale"], batch_size=batch_size):
            out = np.empty((K, b.shocks.shape[0]), dtype=np.float64, order="C")
            apply_shocks_to_base_levels(base_levels=inp["base_levels"], batch=b, out=out)
            shocks.append(np.array(b.shocks, copy=True))
            chunks.append(out)
        stream = np.vstack(shocks)
        levels = np.ascontiguousarray(np.hstack(chunks))
        axis = FactorAxis.from_keys(pl.Series("factor_id", FIDS.tolist(), dtype=pl.Int32))
        settlement = inp["calculator"].run(
            axis.levels(base_levels=inp["base_levels"], shocked_levels=levels))
        plan = build_fx_conversion_plan(
            settlement_ccys=list(inp["reporting"].settlement_ccys),
            reporting_ccy=inp["reporting"].reporting_ccy,
            fx_factor_names=list(inp["reporting"].fx_factor_names))
        conv = convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=np.ascontiguousarray(settlement.pnl, dtype=np.float64),
            fx_levels=np.ascontiguousarray(
                levels[inp["reporting"].fx_factor_rows], dtype=np.float64),
            plan=plan)
        return {{
            "requested_workers": int(numba.config.NUMBA_NUM_THREADS),
            "actual_workers": int(numba.get_num_threads()),
            "layer": str(numba.threading_layer()),
            "scenario_stream": h(stream),
            "shocked_levels": h(levels),
            "trade_pnl": h(np.asarray(settlement.pnl)),
            "portfolio_pnl": h(np.asarray(conv.portfolio_pnl)),
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

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
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


def certify_parametric_blas(
    cert: Certificate, threads: list[int], dims: dict[str, int]
) -> None:
    print("\n[4] Parametric BLAS attack (np.matmul in simlab/scenarios/parametric.py)")
    top = threads[-1]
    base = _child(
        _capture_script("parametric", 64, dims), {"NUMBA_NUM_THREADS": str(top)}
    )
    if "__error__" in base:
        cert.record("parametric BLAS baseline", False, base["__error__"][:140])
        return
    results = {}
    for env in BLAS_ENVS:
        got = _child(
            _capture_script("parametric", 64, dims),
            {"NUMBA_NUM_THREADS": str(top), **env},
        )
        label = ",".join(f"{k}={v}" for k, v in env.items())
        if "__error__" in got:
            cert.record(f"parametric under {label}", False, got["__error__"][:140])
            continue
        div = _first_divergence(base, got)
        cert.record(
            f"parametric identical under {label}",
            div is None,
            "" if div is None else f"FIRST DIVERGENCE at {div}",
        )
        results[label] = div or "identical"
    cert.evidence["parametric_blas"] = results


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
    from riskquant.solutions.risk_run.schemas import NonReproducibleRunError, RiskRun
    from riskquant.solutions.scenario_models.api import parse_scenario_model_config
    from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix

    base = inputs("sbb", 64)
    ref = run_scenario_var_es(**base)

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
            np.array(base["data"].values)[:, list(range(1, K)) + [0]]))},
        "source_factor_ids": {"source_factor_ids": base["source_factor_ids"][::-1].copy()},
        "shift_types": {"required_shift_type_id":
            np.full(K, 2, dtype=np.int8)},
        "shock_scale": {"shock_scale": (base["shock_scale"] * 0.9).astype(np.float32)},
        "base_levels": {"base_levels": base["base_levels"] * 1.001},
        "confidence": {"confidence_level": 0.95},
        "compiled_book": {"calculator": calculator(variant=1)},
    }
    for name, over in subs.items():
        inp = dict(inputs("sbb", 64)); inp.update(over)
        try:
            run_scenario_var_es(**inp, reproduce=ref.manifest)
            stale[name] = "ACCEPTED (defect)"
        except InputMismatchError:
            stale[name] = "InputMismatchError"
        except Exception as exc:
            stale[name] = f"other:{type(exc).__name__}"
    results["stale_input"] = stale

    # identity mutation: result AND identity must move
    mut = {}
    for name, over in subs.items():
        if name == "compiled_book":
            continue
        inp = dict(inputs("sbb", 64)); inp.update(over)
        got = run_scenario_var_es(**inp)
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
    print(
        "\n[6] Contract gates (environment, seed, stale input, identity, tamper, bypass)"
    )
    script = _FIXTURE.format(**dims) + _CONTRACTS.replace("__SRC__", repr(str(_SRC)))
    res = _child(script, {"NUMBA_NUM_THREADS": "2"})
    if "__error__" in res:
        cert.record("contract gates", False, res["__error__"][:400])
        return

    env_ok = all(v == "rejected" for v in res["environment_gate"].values())
    cert.record(
        "environment gate rejects every changed field",
        env_ok,
        json.dumps(res["environment_gate"]),
    )

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
    cert.record("seed policy", seed_ok, json.dumps(seed))

    stale_ok = all(v == "InputMismatchError" for v in res["stale_input"].values())
    cert.record(
        "stale input refused before numerics", stale_ok, json.dumps(res["stale_input"])
    )

    mut = res["identity_mutation"]
    mut_ok = all(v["identity_changed"] for v in mut.values())
    weak = [k for k, v in mut.items() if not v["result_changed"]]
    cert.record(
        "every result-defining mutation changes identity",
        mut_ok,
        (
            "conservative over-separation (identity moved, result did not): "
            + ",".join(weak)
        )
        if weak
        else "",
    )

    cert.record(
        "identity invariant to batch size", bool(res["identity_operational_neutral"])
    )

    tamper_ok = all("rejected" in v for v in res["tamper"].values())
    cert.record("RiskRun tamper protection", tamper_ok, json.dumps(res["tamper"]))

    producers = res["var_es_producers"]
    expected = {"solutions/risk_run/api.py", "credit/aggregate/loss_metrics.py"}
    found = {p.replace(os.sep, "/") for p in producers}
    cert.record(
        "bypass audit: one public scenario-VaR entry point",
        found == expected,
        f"producers={sorted(found)}",
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
