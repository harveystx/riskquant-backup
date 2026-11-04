# src/riskquant/solutions/credit_var/errors.py
"""
Public error types of the credit-VaR solution (WS_CREDIT_VAR_API O-9, C.20).

Two error types leave the public entry points `run_credit_var` and
`simulate_portfolio_losses`:

* R = ``pydantic.ValidationError`` for a request fault. The caller can fix it.
  R is raised only by request admission, before the first engine call. Each
  entry carries the loc of A_CONTRACT section 2.5. riskquant does not
  re-export the type; it comes from pydantic.
* E = `CreditVarEngineError` for an engine fault. The caller cannot fix it by
  a change of the request; the engine, its environment or its installation is
  at fault.

Mapping rule
------------
After admission, every ``Exception`` from the engine path becomes E, with the
original exception chained as ``__cause__`` (`engine_fault`). That includes a
``ValueError`` of an engine invariant and a ``ValidationError`` from the
construction of a result model: the request was valid, so a bad result is the
engine's fault, not the caller's. An exception that is already E passes
through unchanged.

`CreditVarEngineError` subclasses `RiskQuantError` and NOT `ValueError`.
``pydantic.ValidationError`` subclasses ``ValueError`` (F-14), so an
``except ValueError`` clause in a consumer catches only request faults and
never hides an engine fault behind a "bad input" message.
"""

from __future__ import annotations

__all__ = [
    "CreditVarEngineError",
    "engine_fault",
]

from riskquant.exceptions import RiskQuantError


class CreditVarEngineError(RiskQuantError):
    """An engine fault in a credit-VaR run.

    Raised when an engine invariant fails after the request passed validation,
    for example a result row without a request obligor, or when the build
    identity of the installed engine cannot be established. The original
    exception, where one exists, is chained as ``__cause__``.
    """


def engine_fault(entry_point: str, cause: Exception) -> CreditVarEngineError:
    """The E for an exception that the engine path of `entry_point` raised.

    The caller raises the return value ``from cause``, so that the chain keeps
    the original exception and its traceback as ``__cause__``.
    """
    return CreditVarEngineError(
        f"{entry_point}: engine fault after request validation: "
        f"{type(cause).__name__}: {cause}"
    )
