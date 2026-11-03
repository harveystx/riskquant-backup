# src/riskquant/solutions/credit_var/run.py
"""
`run_credit_var` — the credit-VaR solution entry point.

JSON/Pydantic in, JSON/Pydantic out; the core engine in between. This is the
notebook-facing convenience today and Emissary's integration target tomorrow:
the same call serves both because the contract is a validated payload, not a set
of engine objects. The function depends only on the frozen core API
(the solution recipe + the engine), never the other way round.

Notebook ergonomics: pass a `CreditVarRequest`, a plain ``dict`` (it is validated
into one), or keyword the request fields via `CreditVarRequest(...)`. The result
is a `CreditVarResult` whose `.model_dump()` / `.model_dump_json()` is wire-ready.

Result assembly (result schema version 1.0)
-------------------------------------------
* Each requested alpha becomes one `AlphaRisk` record that carries the request
  float bit for bit; the records are sorted ascending. The engine does not
  deduplicate alphas (O-1).
* Obligor contributions and stress-test obligors are sorted ascending by
  obligor_uid; values are computed in loss-matrix row order and only the
  records are reordered, so no number changes.
* The forced-default overlay reports under the JtdModel name "FORCED_DEFAULT",
  the key the engine itself uses for the forced_default model.
* The build identity and the input hash are computed BEFORE the simulation: an
  engine that cannot state its build fails before it spends the run time.

Request admission and errors (WS_CREDIT_VAR_API S2, C.15, C.20)
---------------------------------------------------------------
* Each entry point validates the request state at call time, also for a
  `CreditVarRequest` instance: ``frozen=True`` blocks assignment but not the
  in-place mutation of a dict field such as ``params`` (F-3), so the instance
  is dumped and validated again. The run uses the validated copy, never the
  caller's object.
* A request fault raises ``pydantic.ValidationError`` (R), and only from
  admission, before the first engine call. A bad ``max_samples`` argument of
  `simulate_portfolio_losses` is a caller fault too: it raises R at loc
  ``("max_samples",)`` in the same admission step. After admission, every
  ``Exception`` becomes `CreditVarEngineError` (E) with ``__cause__`` set
  (`errors.engine_fault`), including a ``ValidationError`` from the
  construction of a result model.

Concurrent calls (WS_CREDIT_VAR_API S3, C.23 to C.25, O-10)
-----------------------------------------------------------
* The engine path enters 11 parallel Numba kernels: the 9 JTD kernels (through
  `run_jtd_for_obligor_ead`) and the 2 tail kernels of ``scenario_metrics``
  (through `summarize_jtd_loss`, `model_tail_var_es` and
  `obligor_tail_attribution`). Each of these calls runs while the thread holds
  `_PARALLEL_KERNEL_LOCK`, one process-wide lock of this module.
* Why: concurrent entry into parallel regions from several Python threads is
  safe only under some Numba threading layers, and Numba selects the layer at
  run time per host (F-18). The lock makes the run independent of that choice.
* The lock covers the parallel-kernel calls only. Request admission, input
  assembly, factor generation and result assembly stay concurrent.
* The lock serializes the parallel kernels of credit_var only. It does not
  serialize parallel kernels that other riskquant modules enter.
* Calls on separate threads share no mutable state of this module, so each
  gives the result of a serial call.
"""

from __future__ import annotations

__all__ = [
    "run_credit_var",
]

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from riskquant.credit.aggregate.summary import (
    model_tail_var_es,
    summarize_jtd_loss,
)
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.credit.jtd.engine import run_jtd_for_obligor_ead
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.jtd.models import JtdModel
from riskquant.solutions.credit_var._attribution import (
    ObligorAttribution,
    loss_histogram,
    obligor_tail_attribution,
)
from riskquant.solutions.credit_var.errors import CreditVarEngineError, engine_fault
from riskquant.solutions.credit_var.provenance import (
    NUMBA_VERSION,
    NUMPY_VERSION,
    SCENARIO_CONSTRUCTION_ID,
    SCIPY_VERSION,
    engine_version,
    request_input_hash,
)
from riskquant.solutions.credit_var.recipes import assemble_inputs
from riskquant.solutions.credit_var.schemas import (
    RESULT_SCHEMA_VERSION,
    AlphaRisk,
    CreditVarRequest,
    CreditVarResult,
    LossHistogram,
    ModelTail,
    ObligorContribution,
    ObligorRow,
    PercentileLoss,
    RiskSummary,
    RunProvenance,
    StressTestObligor,
    StressTestResult,
)

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

# NEW (16 points: body fill 10.0 + tail density 99.5/99.95/99.99)
_PERCENTILE_GRID: tuple[float, ...] = (
    00.10,
    01.00,
    02.50,
    05.00,
    10.00,
    25.00,
    50.00,
    75.00,
    90.00,
    95.00,
    97.50,
    99.00,
    99.50,
    99.90,
    99.95,
    99.99,
)


_FACTOR_BATCH = 25_000

# Key of the forced-default overlay in `per_model_tail` and
# `expected_loss_by_model`: the JtdModel member name, the same key the engine
# emits for the forced_default model (C.7). One concept, one key.
_FORCED_DEFAULT_KEY: Final[str] = JtdModel.FORCED_DEFAULT.name

# Bounded offender preview in engine-fault messages.
_PREVIEW: Final[int] = 10

# One process-wide lock for the entry into the parallel Numba kernels of
# credit_var (O-10, C.25). A plain Lock and not an RLock: no locked section
# calls another, so a re-entry is a defect and deadlocks loudly instead of
# nesting silently. Hold it only around the calls that enter a parallel kernel.
_PARALLEL_KERNEL_LOCK: Final[threading.Lock] = threading.Lock()


def _obligor_frame(obligors: Sequence[ObligorRow]) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": [o.obligor_uid for o in obligors],
        "ead": [o.ead for o in obligors],
        "pd": [o.pd for o in obligors],
        "lgd": [o.lgd for o in obligors],
        "group": [o.group for o in obligors],
        "rho": [o.rho for o in obligors],
    })


def _group_params(
    request: CreditVarRequest,
) -> dict[str, dict[str, float]] | None:
    if not request.groups:
        return None
    return {g.name: dict(g.params) for g in request.groups}


def _distribution_metrics(losses: _F64, expected_loss: float) -> dict[str, float]:
    """Shape metrics of the portfolio-loss distribution (cheap, from the S-vector).

    A forced-default overlay only shifts losses by a constant, so dispersion/shape
    here are unaffected; the constant lands in expected_loss/VaR/ES only.

    Zero-variance rule (owner decision ZV): when ``loss_min == loss_max`` every
    scenario has the same loss, so the standard deviation, the skewness and the
    excess kurtosis are exactly 0.0. The floating-point mean of S equal values
    can differ from that value by 1 ulp (S = 2000 copies of 0.1 give a mean
    1.4e-17 away), which gives a nonzero std and a skewness near -1 from
    rounding noise only. The exact test on the extremes removes that noise; the
    coefficient of variation then follows from the zero std.
    """
    mean = float(losses.mean())
    loss_min = float(losses.min())
    loss_max = float(losses.max())
    if loss_min == loss_max:
        std = 0.0
        skewness = 0.0
        kurtosis = 0.0
    else:
        # loss_min < loss_max implies n >= 2, so ddof=1 is defined.
        std = float(losses.std(ddof=1))
        if std > 0.0:
            z = (losses - mean) / std
            skewness = float((z**3).mean())
            kurtosis = float((z**4).mean() - 3.0)  # excess kurtosis (Gaussian = 0)
        else:
            skewness = 0.0
            kurtosis = 0.0
    return {
        "loss_mean": mean,
        "loss_std_dev": std,
        "coefficient_of_variation": (std / expected_loss) if expected_loss > 0 else 0.0,
        "loss_skewness": skewness,
        "loss_excess_kurtosis": kurtosis,
        "loss_probability": float((losses > 0.0).mean()),
        "loss_min": loss_min,
        "loss_max": loss_max,
    }


def _loadings_frame(
    request: CreditVarRequest, keep_uids: set[int]
) -> pd.DataFrame | None:
    # Only loadings for modeled obligors; forced-default names bypass the model.
    if not request.loadings:
        return None
    rows = [lo for lo in request.loadings if lo.obligor_uid in keep_uids]
    if not rows:
        return None
    return pd.DataFrame({
        "uid": [lo.obligor_uid for lo in rows],
        "factor": [lo.factor for lo in rows],
        "loading": [lo.loading for lo in rows],
    })


def _stress_test_result(request: CreditVarRequest) -> StressTestResult | None:
    """The standalone hard-default stress over the `in_stress_test` names.

    `total_loss` sums in request order, as before the record sort, so the total
    stays bitwise equal; only the `obligors` records are sorted by uid.
    """
    flagged = [o for o in request.obligors if o.in_stress_test]
    if not flagged:
        return None
    items = [
        StressTestObligor(
            obligor_uid=int(o.obligor_uid),
            ead=float(o.ead),
            lgd=float(o.lgd),
            loss=float(o.ead) * float(o.lgd),
        )
        for o in flagged
    ]
    total_loss = float(sum(i.loss for i in items))
    return StressTestResult(
        n_obligors=len(items),
        total_loss=total_loss,
        obligors=tuple(sorted(items, key=lambda i: i.obligor_uid)),
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _Simulated:
    loss: _F64  # (m, S)
    uids: _I64  # (m,)
    expected_loss: float
    var: dict[float, float]  # alpha -> VaR
    es: dict[float, float]  # alpha -> ES
    expected_loss_by_model: dict[str, float]
    per_model_tail: dict[str, tuple[float, float]]  # name -> (var, es)


def _simulate(request: CreditVarRequest, obligors: list[ObligorRow]) -> _Simulated:
    """Run the selected model over `obligors` (the unflagged subset)."""
    keep = {o.obligor_uid for o in obligors}
    inputs = assemble_inputs(
        model_key=request.model,
        obligor_table=_obligor_frame(obligors),
        params=request.params,
        loadings_table=_loadings_frame(request, keep),
        group_params=_group_params(request),
    )
    portfolio = compile_jtd_portfolio(bundle=inputs.bundle, masters=inputs.masters)

    universe = inputs.masters.factor_universe
    generator = MixedCreditFactorGenerator.from_universe(
        universe,
        total_scenarios=request.n_scenarios,
        seed=request.seed,
        stress_p_by_uix=inputs.stress_p_by_uix,
    )
    credit_factors = build_credit_factor_matrix(
        generator.iter_batches(batch_size=min(request.n_scenarios, _FACTOR_BATCH)),
        num_factors=universe.n_factors,
        num_scenarios=request.n_scenarios,
    )
    alphas = tuple(request.alphas)
    # C.25: the JTD kernels and the tail kernels run under the process lock.
    # Factor generation above stays outside it; it enters no parallel kernel.
    with _PARALLEL_KERNEL_LOCK:
        result = run_jtd_for_obligor_ead(
            jtd_portfolio=portfolio,
            obligor_ead=inputs.obligor_ead,
            credit_factors=credit_factors,
        )
        summary = summarize_jtd_loss(
            result=result, jtd_portfolio=portfolio, alphas=alphas
        )
        per_model = model_tail_var_es(
            result=result,
            jtd_portfolio=portfolio,
            alpha=max(alphas),
        )
    return _Simulated(
        loss=result.loss,
        uids=result.uid,
        expected_loss=float(summary.expected_loss),
        var={a: float(summary.var[a]) for a in alphas},
        es={a: float(summary.es[a]) for a in alphas},
        expected_loss_by_model={
            k: float(v) for k, v in summary.expected_loss_by_model.items()
        },
        per_model_tail={
            name: (float(v), float(e)) for name, (v, e) in per_model.items()
        },
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _LossMatrix:
    """The combined obligor loss matrix (modelled rows + forced-default rows) plus
    the model-level summary scaffolding the overlay produces. Single source of the
    simulation product, shared by run_credit_var and simulate_portfolio_losses."""

    loss: _F64  # (N, S)
    uids: _I64  # (N,)
    base_el: float
    base_var: dict[float, float]
    base_es: dict[float, float]
    el_by_model: dict[str, float]
    per_model: dict[str, tuple[float, float]]
    forced_total: float


def _build_loss_matrix(
    request: CreditVarRequest, alphas: tuple[float, ...]
) -> _LossMatrix:
    """Run the selected model on the unflagged names and stack the forced-default
    constant rows (a forced default is a constant added to every scenario)."""
    forced = (
        []
        if request.model == "forced_default"
        else [o for o in request.obligors if o.force_default]
    )
    forced_uids = {o.obligor_uid for o in forced}
    modeled = [o for o in request.obligors if o.obligor_uid not in forced_uids]

    if modeled:
        sim = _simulate(request, modeled)
        modeled_loss, modeled_uids = sim.loss, sim.uids
        scenarios = int(modeled_loss.shape[1])
        base_el = sim.expected_loss
        base_var = dict(sim.var)
        base_es = dict(sim.es)
        el_by_model = dict(sim.expected_loss_by_model)
        per_model = dict(sim.per_model_tail)
    else:  # everything forced (and not the forced_default model) -> deterministic
        scenarios = request.n_scenarios
        modeled_loss = np.empty((0, scenarios), dtype=np.float64)
        modeled_uids = np.empty(0, dtype=np.int64)
        base_el = 0.0
        base_var = {a: 0.0 for a in alphas}
        base_es = {a: 0.0 for a in alphas}
        el_by_model = {}
        per_model = {}

    forced_total = float(sum(float(o.ead) * float(o.lgd) for o in forced))
    if forced:
        # The overlay only runs when the model is NOT forced_default, so the
        # engine output holds no FORCED_DEFAULT band. A key already present is
        # an engine invariant failure: merging would silently drop one value.
        if _FORCED_DEFAULT_KEY in per_model or _FORCED_DEFAULT_KEY in el_by_model:
            raise CreditVarEngineError(
                f"forced-default overlay: the engine output already holds key "
                f"'{_FORCED_DEFAULT_KEY}' for model '{request.model}'."
            )
        forced_rows = np.empty((len(forced), scenarios), dtype=np.float64)
        for i, o in enumerate(forced):
            forced_rows[i, :] = float(o.ead) * float(o.lgd)
        forced_uid_arr = np.array([int(o.obligor_uid) for o in forced], dtype=np.int64)
        if modeled_uids.size:
            loss = np.vstack([modeled_loss, forced_rows])
            uids = np.concatenate([modeled_uids.astype(np.int64), forced_uid_arr])
        else:
            loss = forced_rows
            uids = forced_uid_arr
        per_model = {**per_model, _FORCED_DEFAULT_KEY: (forced_total, forced_total)}
        el_by_model = {**el_by_model, _FORCED_DEFAULT_KEY: forced_total}
    else:
        loss = modeled_loss
        uids = modeled_uids

    return _LossMatrix(
        loss=loss,
        uids=uids,
        base_el=base_el,
        base_var=base_var,
        base_es=base_es,
        el_by_model=el_by_model,
        per_model=per_model,
        forced_total=forced_total,
    )


def _obligor_contributions(
    request: CreditVarRequest,
    uids: _I64,
    attribution: ObligorAttribution,
) -> tuple[ObligorContribution, ...]:
    """One record per request obligor, ascending by obligor_uid (C.6).

    `attribution` arrays are aligned to the loss-matrix rows (`uids`). The
    values are taken row by row; only the record order changes. Each record
    carries the request ead of its obligor. A row uid that is not a request
    obligor, a request obligor without a row, or a duplicate row uid is an
    engine fault: the old NaN-ead default hid the first case.
    """
    ead_by_uid = {int(o.obligor_uid): float(o.ead) for o in request.obligors}
    row_uids = [int(u) for u in uids]
    unknown = sorted(set(row_uids) - ead_by_uid.keys())
    absent = sorted(ead_by_uid.keys() - set(row_uids))
    if unknown or absent or len(set(row_uids)) != len(row_uids):
        raise CreditVarEngineError(
            f"obligor attribution: loss-matrix rows do not match the request "
            f"obligors one to one. Row uids without a request ead: "
            f"{unknown[:_PREVIEW]}; request uids without a row: "
            f"{absent[:_PREVIEW]}; rows: {len(row_uids)}, distinct row uids: "
            f"{len(set(row_uids))}."
        )
    es_total = attribution.es_total
    ul_total = attribution.ul_total
    order = sorted(range(len(row_uids)), key=row_uids.__getitem__)
    return tuple(
        ObligorContribution(
            obligor_uid=row_uids[i],
            ead=ead_by_uid[row_uids[i]],
            expected_loss=float(attribution.expected_loss[i]),
            es_contribution=float(attribution.es_contribution[i]),
            es_pct=(
                float(attribution.es_contribution[i] / es_total) if es_total else 0.0
            ),
            incremental_var=float(attribution.incremental_var[i]),
            incremental_es=float(attribution.incremental_es[i]),
            ul_contribution=float(attribution.ul_contribution[i]),
            ul_pct=(
                float(attribution.ul_contribution[i] / ul_total) if ul_total else 0.0
            ),
        )
        for i in order
    )


def _admit_request(
    request: CreditVarRequest | Mapping[str, Any], *, entry_point: str
) -> CreditVarRequest:
    """Validate the request state at call time (C.15); raise R or E only.

    A `CreditVarRequest` instance is dumped and validated again: a dict field
    of a frozen model can change in place after construction (F-3), and the
    instance state at call time is what the run would use. ``warnings=False``
    keeps the dump silent for a value of the wrong type; validation then
    reports it as R. Any other input goes to ``model_validate`` as given, so a
    non-mapping raises R as well.

    Raises
    ------
    pydantic.ValidationError
        The request content breaks a rule of A_CONTRACT section 2.
    CreditVarEngineError
        Admission itself failed with another exception type. A rule must
        report a fault as R; any other exception is a defect of the solution.
    """
    try:
        if isinstance(request, CreditVarRequest):
            return CreditVarRequest.model_validate(
                request.model_dump(mode="python", warnings=False)
            )
        return CreditVarRequest.model_validate(request)
    except ValidationError:
        raise
    except Exception as exc:
        raise engine_fault(entry_point, exc) from exc


class _SampleLimit(BaseModel):
    """Admission rule of the `max_samples` argument of `simulate_portfolio_losses`.

    ``None`` (no subsampling) or an int >= 0, strict as the request fields
    (O-8): a bool, a float or a str raises. 0 stays valid: it gives an empty
    sample, as before this rule. A model and not a bare check, so that the
    fault is R with the loc ``("max_samples",)`` and one entry (owner decision
    on S2, O-9 (a)).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_samples: Annotated[int, Field(ge=0)] | None


def _admit_max_samples(max_samples: object, *, entry_point: str) -> int | None:
    """Validate `max_samples` before the engine runs; raise R or E only.

    Raises
    ------
    pydantic.ValidationError
        `max_samples` is not ``None`` and not an int >= 0.
    CreditVarEngineError
        Admission itself failed with another exception type.
    """
    try:
        return _SampleLimit.model_validate({"max_samples": max_samples}).max_samples
    except ValidationError:
        raise
    except Exception as exc:
        raise engine_fault(entry_point, exc) from exc


def simulate_portfolio_losses(
    request: CreditVarRequest | dict[str, Any],
    *,
    max_samples: int | None = 100_000,
) -> _F64:
    """Simulated portfolio loss per scenario, a Python-only member (O-6).

    Returns the column sums of the run's obligor loss matrix, float64 and
    C-contiguous. Above `max_samples` scenarios it returns `max_samples`
    losses drawn without replacement by ``default_rng(seed).choice``.

    Raises
    ------
    pydantic.ValidationError
        The request content is invalid, or `max_samples` is not ``None`` and
        not an int >= 0 (R, loc ``("max_samples",)``). Raised before the first
        engine call; the request is checked first.
    CreditVarEngineError
        Any fault after admission (E), the original exception as
        ``__cause__``.
    """
    entry_point = "simulate_portfolio_losses"
    req = _admit_request(request, entry_point=entry_point)
    limit = _admit_max_samples(max_samples, entry_point=entry_point)
    try:
        return _portfolio_losses(req, max_samples=limit)
    except CreditVarEngineError:
        raise
    except Exception as exc:
        raise engine_fault(entry_point, exc) from exc


def _portfolio_losses(req: CreditVarRequest, *, max_samples: int | None) -> _F64:
    portfolio_losses = _build_loss_matrix(req, tuple(req.alphas)).loss.sum(axis=0)
    portfolio_losses = np.ascontiguousarray(portfolio_losses, dtype=np.float64)

    n = portfolio_losses.shape[0]
    if max_samples is None or n <= max_samples:
        return portfolio_losses

    rng = np.random.default_rng(req.seed)

    # Uniform random sampling without replacement
    idx = rng.choice(n, size=max_samples, replace=False)
    return portfolio_losses[idx]


def run_credit_var(request: CreditVarRequest | dict[str, Any]) -> CreditVarResult:
    """Run one credit-VaR request end to end.

    Two orthogonal obligor flags drive what happens:

    * ``in_stress_test`` -> a standalone deterministic hard-default stress
      (Sum EAD*LGD over the flagged names), reported on `result.stress_test`,
      independent of the model and the simulation.
    * ``force_default`` -> those names default with certainty INSIDE the
      simulation while the selected model runs on the rest. A forced default is a
      constant addition to every scenario's portfolio loss, so the engine runs on
      the unflagged subset and the constant is stacked on. (No-op when the model is
      already ``forced_default``.)

    Raises
    ------
    pydantic.ValidationError
        The request content is invalid (R), before the first engine call.
    CreditVarEngineError
        Any fault after admission (E): for example the build identity cannot be
        established (C.9), the loss-matrix rows do not match the request
        obligors one to one (C.6), or a result model rejects an engine value.
        The original exception, where one exists, is ``__cause__``.
    """
    entry_point = "run_credit_var"
    req = _admit_request(request, entry_point=entry_point)
    try:
        return _run(req)
    except CreditVarEngineError:
        raise
    except Exception as exc:
        raise engine_fault(entry_point, exc) from exc


def _run(req: CreditVarRequest) -> CreditVarResult:
    """The engine path of `run_credit_var` for an admitted request."""
    build = engine_version()
    input_hash = request_input_hash(req)

    alphas = tuple(req.alphas)
    headline_alpha = max(alphas)

    stress_test = _stress_test_result(req)

    lm = _build_loss_matrix(req, alphas)
    loss, uids = lm.loss, lm.uids
    base_el, base_var, base_es = lm.base_el, lm.base_var, lm.base_es
    el_by_model, per_model, forced_total = lm.el_by_model, lm.per_model, lm.forced_total

    # Sum across obligors to get total portfolio loss per scenario
    portfolio_losses = loss.sum(axis=0)
    pct = np.percentile(portfolio_losses, _PERCENTILE_GRID)

    # Forced default shifts every scenario by a constant, so the modeled VaR/ES
    # shift by that constant (order statistics translate).
    expected_loss = base_el + forced_total
    var = {a: base_var[a] + forced_total for a in alphas}
    es = {a: base_es[a] + forced_total for a in alphas}

    # C.25: the attribution enters both tail kernels (VaR/ES and contribution).
    with _PARALLEL_KERNEL_LOCK:
        attribution = obligor_tail_attribution(
            obligor_loss=loss,
            portfolio_losses=portfolio_losses,
            confidence_level=headline_alpha,
        )
    contributions = _obligor_contributions(req, uids, attribution)

    hist_edges, hist_counts = loss_histogram(portfolio_losses)

    return CreditVarResult(
        result_schema_version=RESULT_SCHEMA_VERSION,
        model=req.model,
        headline_alpha=headline_alpha,
        summary=RiskSummary(
            by_alpha=tuple(
                AlphaRisk(
                    alpha=a,
                    var=float(var[a]),
                    es=float(es[a]),
                    economic_capital_var=float(var[a] - expected_loss),
                    economic_capital_es=float(es[a] - expected_loss),
                )
                for a in sorted(alphas)
            ),
            expected_loss_by_model={k: float(v) for k, v in el_by_model.items()},
            **_distribution_metrics(portfolio_losses, float(expected_loss)),
        ),
        per_model_tail={
            name: ModelTail(var=float(v), es=float(e))
            for name, (v, e) in per_model.items()
        },
        loss_percentiles=tuple(
            PercentileLoss(percentile=p, loss=float(v))
            for p, v in zip(_PERCENTILE_GRID, pct, strict=True)
        ),
        obligor_contributions=contributions,
        loss_histogram=LossHistogram(
            bin_edges=tuple(hist_edges), counts=tuple(hist_counts)
        ),
        stress_test=stress_test,
        provenance=RunProvenance(
            engine_version=build,
            scenario_construction_id=SCENARIO_CONSTRUCTION_ID,
            numpy_version=NUMPY_VERSION,
            scipy_version=SCIPY_VERSION,
            numba_version=NUMBA_VERSION,
            model=req.model,
            params=dict(req.params),
            n_obligors=len(req.obligors),
            n_scenarios=req.n_scenarios,
            seed=req.seed,
            alphas=tuple(req.alphas),
            input_hash=input_hash,
            generated_at=datetime.now(UTC).isoformat(),
        ),
    )
