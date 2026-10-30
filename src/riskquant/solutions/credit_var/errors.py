# src/riskquant/solutions/credit_var/errors.py
"""
Public error type of the credit-VaR solution for an engine fault.

Two error types leave the public entry points (WS_CREDIT_VAR_API O-9):

* ``pydantic.ValidationError`` for a request fault. The caller can fix it.
* `CreditVarEngineError` for an engine fault. The caller cannot fix it by a
  change of the request; the engine, its environment or its installation is
  at fault.

`CreditVarEngineError` subclasses `RiskQuantError` and NOT `ValueError`.
``pydantic.ValidationError`` subclasses ``ValueError``, so an
``except ValueError`` clause in a consumer catches only request faults and
never hides an engine fault behind a "bad input" message.
"""

from __future__ import annotations

__all__ = [
    "CreditVarEngineError",
]

from riskquant.exceptions import RiskQuantError


class CreditVarEngineError(RiskQuantError):
    """An engine fault in a credit-VaR run.

    Raised when an engine invariant fails after the request passed validation,
    for example a result row without a request obligor, or when the build
    identity of the installed engine cannot be established. The original
    exception, where one exists, is chained as ``__cause__``.
    """
