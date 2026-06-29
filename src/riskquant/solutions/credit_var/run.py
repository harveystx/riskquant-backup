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
"""

from __future__ import annotations

__all__ = [
    "run_credit_var",
]

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from riskquant.credit.aggregate.summary import (
    model_tail_var_es,
    summarize_jtd_loss,
)
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.credit.jtd.engine import run_jtd_for_obligor_ead
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.solutions.credit_var._attribution import (
    loss_histogram,
    obligor_tail_attribution,
)
from riskquant.solutions.credit_var.recipes import assemble_inputs
from riskquant.solutions.credit_var.schemas import (
    CreditVarRequest,
    CreditVarResult,
    LossHistogram,
    ModelTail,
    ObligorContribution,
    RiskSummary,
    RunProvenance,
    StressTestObligor,
    StressTestResult,
)

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


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("riskquant")
    except Exception:
        return "0+unknown"


def _obligor_frame(obligors: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [o.uid for o in obligors],
            "ead": [o.ead for o in obligors],
            "pd": [o.pd for o in obligors],
            "lgd": [o.lgd for o in obligors],
            "group": [o.group for o in obligors],
        }
    )


def _group_params(
    request: CreditVarRequest,
) -> dict[str, dict[str, float]] | None:
    if not request.groups:
        return None
    return {g.name: dict(g.params) for g in request.groups}


def _loadings_frame(
    request: CreditVarRequest, keep_uids: set[int]
) -> pd.DataFrame | None:
    # Only loadings for modeled obligors; forced-default names bypass the model.
    if not request.loadings:
        return None
    rows = [lo for lo in request.loadings if lo.uid in keep_uids]
    if not rows:
        return None
    return pd.DataFrame(
        {
            "uid": [lo.uid for lo in rows],
            "factor": [lo.factor for lo in rows],
            "loading": [lo.loading for lo in rows],
        }
    )


def _input_hash(request: CreditVarRequest) -> str:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _alpha_key(alpha: float) -> str:
    """S-tier with integer arithmetic - 10x faster than Decimal."""
    if not math.isfinite(alpha) or alpha <= 0:
        return "0.00%"
    if alpha >= 1:
        return "100.00%"

    # Scale to basis points (1/10000)
    cents = int(round(alpha * 10000))

    # Clamp
    if cents < 0:
        cents = 0
    elif cents > 10000:
        cents = 10000

    # Format: 99.99%
    return f"{cents // 100:02d}.{cents % 100:02d}%"


def _percentile_key(p: float) -> str:
    """S-tier with integer arithmetic."""
    if not math.isfinite(p) or p <= 0:
        return "0.00%"
    if p >= 100:
        return "100.00%"

    # Scale to basis points (direct since already percentage)
    cents = int(round(p * 100))

    if cents < 0:
        cents = 0
    elif cents > 10000:
        cents = 10000

    return f"{cents // 100:02d}.{cents % 100:02d}%"


def _stress_test_result(request: CreditVarRequest) -> StressTestResult | None:
    """The standalone hard-default stress over the `in_stress_test` names."""
    flagged = [o for o in request.obligors if o.in_stress_test]
    if not flagged:
        return None
    items = [
        StressTestObligor(
            uid=int(o.uid),
            ead=float(o.ead),
            lgd=float(o.lgd),
            loss=float(o.ead) * float(o.lgd),
        )
        for o in flagged
    ]
    return StressTestResult(
        n_obligors=len(items),
        total_loss=float(sum(i.loss for i in items)),
        obligors=items,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Simulated:
    loss: np.ndarray  # (m, S)
    uids: np.ndarray  # (m,)
    expected_loss: float
    var: dict  # alpha -> VaR
    es: dict  # alpha -> ES
    expected_loss_by_model: dict
    per_model_tail: dict  # name -> (var, es)


def _simulate(request: CreditVarRequest, obligors: list) -> _Simulated:
    """Run the selected model over `obligors` (the unflagged subset)."""
    keep = {o.uid for o in obligors}
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
    result = run_jtd_for_obligor_ead(
        jtd_portfolio=portfolio,
        obligor_ead=inputs.obligor_ead,
        credit_factors=credit_factors,
    )
    alphas = tuple(request.alphas)
    summary = summarize_jtd_loss(result=result, jtd_portfolio=portfolio, alphas=alphas)
    per_model = model_tail_var_es(
        result=result, jtd_portfolio=portfolio, alpha=max(alphas)
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


def run_credit_var(request: CreditVarRequest | dict) -> CreditVarResult:
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
    """
    req = (
        request
        if isinstance(request, CreditVarRequest)
        else CreditVarRequest.model_validate(request)
    )

    alphas = tuple(req.alphas)
    headline_alpha = max(alphas)

    stress_test = _stress_test_result(req)

    forced = (
        []
        if req.model == "forced_default"
        else [o for o in req.obligors if o.force_default]
    )
    forced_uids = {o.uid for o in forced}
    modeled = [o for o in req.obligors if o.uid not in forced_uids]

    if modeled:
        sim = _simulate(req, modeled)
        modeled_loss, modeled_uids = sim.loss, sim.uids
        scenarios = int(modeled_loss.shape[1])
        base_el = sim.expected_loss
        base_var = dict(sim.var)
        base_es = dict(sim.es)
        el_by_model = dict(sim.expected_loss_by_model)
        per_model = dict(sim.per_model_tail)
    else:  # everything forced (and not the forced_default model) -> deterministic
        scenarios = req.n_scenarios
        modeled_loss = np.empty((0, scenarios), dtype=np.float64)
        modeled_uids = np.empty(0, dtype=np.int64)
        base_el = 0.0
        base_var = {a: 0.0 for a in alphas}
        base_es = {a: 0.0 for a in alphas}
        el_by_model = {}
        per_model = {}

    forced_total = float(sum(float(o.ead) * float(o.lgd) for o in forced))
    if forced:
        forced_rows = np.empty((len(forced), scenarios), dtype=np.float64)
        for i, o in enumerate(forced):
            forced_rows[i, :] = float(o.ead) * float(o.lgd)
        forced_uid_arr = np.array([int(o.uid) for o in forced], dtype=np.int64)
        if modeled_uids.size:
            loss = np.vstack([modeled_loss, forced_rows])
            uids = np.concatenate([modeled_uids.astype(np.int64), forced_uid_arr])
        else:
            loss = forced_rows
            uids = forced_uid_arr
        per_model = {**per_model, "forced_default": (forced_total, forced_total)}
        el_by_model = {**el_by_model, "forced_default": forced_total}
    else:
        loss = modeled_loss
        uids = modeled_uids

    portfolio_losses = loss.sum(axis=0)
    pct = np.percentile(portfolio_losses, _PERCENTILE_GRID)

    # Forced default shifts every scenario by a constant, so the modeled VaR/ES
    # shift by that constant (order statistics translate).
    expected_loss = base_el + forced_total
    var = {a: base_var[a] + forced_total for a in alphas}
    es = {a: base_es[a] + forced_total for a in alphas}

    attribution = obligor_tail_attribution(
        obligor_loss=loss,
        portfolio_losses=portfolio_losses,
        confidence_level=headline_alpha,
    )
    ead_by_uid = {int(o.uid): float(o.ead) for o in req.obligors}
    es_total = attribution.es_total
    contributions = [
        ObligorContribution(
            uid=int(uid),
            ead=ead_by_uid.get(int(uid), float("nan")),
            expected_loss=float(attribution.expected_loss[i]),
            es_contribution=float(attribution.es_contribution[i]),
            es_pct=(
                float(attribution.es_contribution[i] / es_total) if es_total else 0.0
            ),
            incremental_var=float(attribution.incremental_var[i]),
            incremental_es=float(attribution.incremental_es[i]),
        )
        for i, uid in enumerate(uids)
    ]
    hist_edges, hist_counts = loss_histogram(portfolio_losses)

    return CreditVarResult(
        model=req.model,
        headline_alpha=headline_alpha,
        summary=RiskSummary(
            expected_loss=float(expected_loss),
            var={_alpha_key(a): float(var[a]) for a in alphas},
            es={_alpha_key(a): float(es[a]) for a in alphas},
            expected_loss_by_model={k: float(v) for k, v in el_by_model.items()},
        ),
        per_model_tail={
            name: ModelTail(var=float(v), es=float(e))
            for name, (v, e) in per_model.items()
        },
        loss_percentiles={
            _percentile_key(p): float(v)
            for p, v in zip(_PERCENTILE_GRID, pct, strict=True)
        },
        obligor_contributions=contributions,
        loss_histogram=LossHistogram(bin_edges=hist_edges, counts=hist_counts),
        stress_test=stress_test,
        provenance=RunProvenance(
            engine_version=_engine_version(),
            model=req.model,
            params=dict(req.params),
            n_obligors=len(req.obligors),
            n_scenarios=req.n_scenarios,
            seed=req.seed,
            alphas=list(req.alphas),
            input_hash=_input_hash(req),
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
