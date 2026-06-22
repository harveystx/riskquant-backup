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
from riskquant.solutions.credit_var.recipes import assemble_inputs
from riskquant.solutions.credit_var.schemas import (
    CreditVarRequest,
    CreditVarResult,
    ModelTail,
    RiskSummary,
    RunProvenance,
)

_PERCENTILE_GRID: tuple[float, ...] = (
    0.1,
    1.0,
    2.5,
    5.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    97.5,
    99.0,
    99.9,
)
_FACTOR_BATCH = 25_000


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("riskquant")
    except Exception:
        return "0+unknown"


def _obligor_frame(request: CreditVarRequest) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [o.uid for o in request.obligors],
            "ead": [o.ead for o in request.obligors],
            "pd": [o.pd for o in request.obligors],
            "lgd": [o.lgd for o in request.obligors],
        }
    )


def _loadings_frame(request: CreditVarRequest) -> pd.DataFrame | None:
    if not request.loadings:
        return None
    return pd.DataFrame(
        {
            "uid": [lo.uid for lo in request.loadings],
            "factor": [lo.factor for lo in request.loadings],
            "loading": [lo.loading for lo in request.loadings],
        }
    )


def _input_hash(request: CreditVarRequest) -> str:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _alpha_key(alpha: float) -> str:
    # Stable, JSON-friendly key (no trailing-zero noise): 0.99 -> "0.99".
    return repr(float(alpha))


def run_credit_var(request: CreditVarRequest | dict) -> CreditVarResult:
    """Run one credit-VaR request end to end.

    Parameters
    ----------
    request
        A `CreditVarRequest`, or a dict validated into one (Pydantic raises
        `ValidationError` with cell-addressable `loc` on bad input).
    """
    req = (
        request
        if isinstance(request, CreditVarRequest)
        else CreditVarRequest.model_validate(request)
    )

    obligor_df = _obligor_frame(req)
    loadings_df = _loadings_frame(req)
    inputs = assemble_inputs(
        model_key=req.model,
        obligor_table=obligor_df,
        params=req.params,
        loadings_table=loadings_df,
    )
    portfolio = compile_jtd_portfolio(bundle=inputs.bundle, masters=inputs.masters)

    universe = inputs.masters.factor_universe
    generator = MixedCreditFactorGenerator.from_universe(
        universe,
        total_scenarios=req.n_scenarios,
        seed=req.seed,
        stress_p_by_uix=inputs.stress_p_by_uix,
    )
    credit_factors = build_credit_factor_matrix(
        generator.iter_batches(batch_size=min(req.n_scenarios, _FACTOR_BATCH)),
        num_factors=universe.n_factors,
        num_scenarios=req.n_scenarios,
    )

    result = run_jtd_for_obligor_ead(
        jtd_portfolio=portfolio,
        obligor_ead=inputs.obligor_ead,
        credit_factors=credit_factors,
    )

    alphas = tuple(req.alphas)
    summary = summarize_jtd_loss(result=result, jtd_portfolio=portfolio, alphas=alphas)
    headline_alpha = max(alphas)
    per_model = model_tail_var_es(
        result=result, jtd_portfolio=portfolio, alpha=headline_alpha
    )

    portfolio_losses = result.loss.sum(axis=0)
    pct = np.percentile(portfolio_losses, _PERCENTILE_GRID)

    return CreditVarResult(
        model=req.model,
        headline_alpha=headline_alpha,
        summary=RiskSummary(
            expected_loss=float(summary.expected_loss),
            var={_alpha_key(a): float(summary.var[a]) for a in alphas},
            es={_alpha_key(a): float(summary.es[a]) for a in alphas},
            expected_loss_by_model={
                k: float(v) for k, v in summary.expected_loss_by_model.items()
            },
        ),
        per_model_tail={
            name: ModelTail(var=float(v), es=float(e))
            for name, (v, e) in per_model.items()
        },
        loss_percentiles={
            repr(float(p)): float(v) for p, v in zip(_PERCENTILE_GRID, pct, strict=True)
        },
        provenance=RunProvenance(
            engine_version=_engine_version(),
            model=req.model,
            params=dict(req.params),
            n_obligors=int(portfolio.total_obligors),
            n_scenarios=req.n_scenarios,
            seed=req.seed,
            alphas=list(req.alphas),
            input_hash=_input_hash(req),
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
