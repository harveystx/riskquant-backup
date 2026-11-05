# src/riskquant/solutions/credit_exposure/handoff.py
"""
Credit-exposure -> credit-VaR handoff.

The exposure result's per-obligor EAD is exactly the ``ead`` the credit-VaR layer
consumes. ``attach_exposure_ead`` joins it onto a credit obligor table by ``uid``,
turning the two solution packages into one line::

    from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var

    obligors = attach_exposure_ead(credit_obligors, exposure_result)  # fills `ead`
    run_credit_var(CreditVarRequest.from_dataframe(obligors, model="sf_gaussian"))
"""

from __future__ import annotations

__all__ = [
    "attach_exposure_ead",
]

import polars as pl

from riskquant.solutions.credit_exposure.schemas import CreditExposureResult


def attach_exposure_ead(
    obligor_table: pl.DataFrame,
    result: CreditExposureResult,
    *,
    uid_col: str = "obligor_uid",
) -> pl.DataFrame:
    """Return a new ``obligor_table`` with an ``ead`` Float64 column filled from the run.

    Strict: every obligor in the table must have an EAD in ``result`` (a missing
    one is a portfolio/exposure mismatch, not a zero to invent). Extra obligors in
    the result are ignored. An existing ``ead`` column is replaced in place; the
    caller's frame is not changed (polars frames are immutable, D095 form).

    Raises
    ------
    TypeError
        ``obligor_table`` is not a ``pl.DataFrame``.
    ValueError
        ``uid_col`` is missing, holds a null, or names an obligor without an
        EAD in ``result``.
    """
    if not isinstance(obligor_table, pl.DataFrame):
        kind = f"{type(obligor_table).__module__}.{type(obligor_table).__qualname__}"
        raise TypeError(
            "attach_exposure_ead: 'obligor_table' must be a pl.DataFrame "
            f"(polars.DataFrame); got {kind}."
        )
    if uid_col not in obligor_table.columns:
        raise ValueError(f"attach_exposure_ead: obligor_table missing '{uid_col}'.")
    uid_series = obligor_table.get_column(uid_col)
    if uid_series.null_count():
        raise ValueError(f"attach_exposure_ead: null in '{uid_col}'.")
    ead_by_obligor_uid = result.ead_by_obligor_uid()
    obligor_uids = [int(u) for u in uid_series.to_list()]
    obligor_uids_missing_eads: list[int] = sorted({
        u for u in obligor_uids if u not in ead_by_obligor_uid
    })
    if obligor_uids_missing_eads:
        raise ValueError(
            f"attach_exposure_ead: no exposure EAD for obligor uid(s): {obligor_uids_missing_eads}."
        )
    return obligor_table.with_columns(
        pl.Series(
            "ead", [ead_by_obligor_uid[u] for u in obligor_uids], dtype=pl.Float64
        )
    )
