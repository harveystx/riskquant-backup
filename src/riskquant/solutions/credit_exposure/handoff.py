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

import pandas as pd

from riskquant.solutions.credit_exposure.schemas import CreditExposureResult


def attach_exposure_ead(
    obligor_table: pd.DataFrame,
    result: CreditExposureResult,
    *,
    uid_col: str = "uid",
) -> pd.DataFrame:
    """Return ``obligor_table`` with an ``ead`` column filled from the exposure run.

    Strict: every obligor in the table must have an EAD in ``result`` (a missing
    one is a portfolio/exposure mismatch, not a zero to invent). Extra obligors in
    the result are ignored.
    """
    if uid_col not in obligor_table.columns:
        raise ValueError(f"attach_exposure_ead: obligor_table missing '{uid_col}'.")
    ead_by_uid = result.ead_by_uid()
    obligor_uids = [int(u) for u in obligor_table[uid_col]]
    obligor_uids_missing_eads: list[int] = sorted(
        {u for u in obligor_uids if u not in ead_by_uid}
    )
    if obligor_uids_missing_eads:
        raise ValueError(
            f"attach_exposure_ead: no exposure EAD for obligor uid(s): {obligor_uids_missing_eads}."
        )
    out = obligor_table.copy()
    out["ead"] = [ead_by_uid[u] for u in obligor_uids]
    return out
