# src/riskquant/solutions/credit_var/api.py
"""
Public API for the credit-VaR solution.

This module is THE entry surface for the solution: every symbol a consumer
(notebook, Emissary, the xlwings tool) is meant to use is re-exported here, and
nothing else in `riskquant.solutions.credit_var.*` should be imported from
outside the package. The other modules (`run`, `schemas`, `catalog`, `recipes`,
`errors`, `provenance`) are supportive internals. They may change shape freely
as long as this surface holds.

Why a module and not the package `__init__`: in `src/riskquant`, every
`__init__.py` is empty, without exception. A solution that wants a single import
line gets it from a named `api` module instead, so the "one stable surface" need
is met without weakening the empty-init rule.

Surface (WS_CREDIT_VAR_API C.28, D106 clause 1)
---------------------------------------------------------------
The 22 names of `__all__` are the stable surface. The package version follows
SemVer over these names; before 1.0.0 a minor bump marks a breaking change.
`tests/solutions/credit_var/test_public_surface.py` freezes the name list
(S5.T1) and the JSON schemas of `CreditVarRequest` and `CreditVarResult`
(S5.T2). A failure of either test needs a version decision, not a test edit.

Two exemptions apply:

* `CreditVarRequest.from_dataframe` is outside the stable surface (O-7).
* `simulate_portfolio_losses` is a Python-only member (O-6). Its `np.ndarray`
  return has no JSON form, no result schema version and no golden file.

The request error type is ``pydantic.ValidationError``. It comes from pydantic;
this module does not re-export it. The engine error type is
`CreditVarEngineError` (O-9).

Numeric domain (WS_CREDIT_VAR_API S7, owner decision OD-2 (b))
--------------------------------------------------------------
Request admission checks the rules of the request contract only. It does not
check the float64 and array limits of the engine. An admitted request can
therefore still raise `CreditVarEngineError` (E) in these cases:

* ``n_scenarios`` near or above 2**63: the scenario arrays exceed the maximum
  array size or dimension of NumPy.
* ``seed`` with more than 4300 decimal digits, on the dict path: the int to
  str conversion limit of Python (``sys.get_int_max_str_digits``) stops the
  run.
* ``ead`` near 1e154 and above: the square of a loss overflows float64, so a
  moment or a UL contribution is not finite and a result model rejects it.
* ``(1 - alpha) x n_scenarios < 16 eps`` for an alpha in ``alphas``: the tail
  holds no scenario mass, and the ES contribution is not finite.

Backlog item R-004 covers the first three cases. Backlog item R-005 covers the
last case. The request has no size limit (O-12). A consumer sets its own limit on ``n_obligors x n_scenarios``.
One obligor x scenario cell costs about 32 B of memory (audit class G).

Import as::

    from riskquant.solutions.credit_var.api import run_credit_var, CreditVarRequest
"""

from __future__ import annotations

__all__ = [
    "MODEL_CATALOG",
    "AlphaRisk",
    "CreditVarEngineError",
    "CreditVarRequest",
    "CreditVarResult",
    "GroupRow",
    "LoadingRow",
    "LossHistogram",
    "ModelEntry",
    "ModelTail",
    "ObligorContribution",
    "ObligorRow",
    "ParamSpec",
    "PercentileLoss",
    "RiskSummary",
    "RunProvenance",
    "StressTestObligor",
    "StressTestResult",
    "model_entry",
    "run_credit_var",
    "simulate_portfolio_losses",
    "supported_model_keys",
]

from riskquant.solutions.credit_var.catalog import (
    MODEL_CATALOG,
    ModelEntry,
    ParamSpec,
    model_entry,
    supported_model_keys,
)
from riskquant.solutions.credit_var.errors import CreditVarEngineError
from riskquant.solutions.credit_var.run import (
    run_credit_var,
    simulate_portfolio_losses,
)
from riskquant.solutions.credit_var.schemas import (
    AlphaRisk,
    CreditVarRequest,
    CreditVarResult,
    GroupRow,
    LoadingRow,
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
