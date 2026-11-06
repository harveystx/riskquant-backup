# src/riskquant/credit/jtd/ingest/loading_sum.py
"""
Per-obligor loading sum of squares for multi-factor admission (WS_CREDIT_POLARS C.5).

Two callers decide sum(loading^2) <= 1 + 1e-9 per obligor:

* `riskquant.credit.jtd.ingest.validate._semantic_mf_gaussian`, the engine
  admission check of `mf_gaussian` and `mf_students_t`;
* `riskquant.solutions.credit_var.schemas.CreditVarRequest._check_loading_sum_sq`,
  request rule 14.

Near the bound, two summations of the same squares can round to adjacent floats
on the two sides of 1 + 1e-9. Only one arithmetic gives one decision, so both
callers call `loading_sum_sq` and hold no other sum of loading squares. The
bound comparison stays in the callers.

Arithmetic
----------
Per obligor, over the rows in input order, from ``total = compensation = 0.0``::

    y = x * x - compensation
    t = total + y
    compensation = (t - total) - y
    total = t

then ``compensation = 0.0`` when the compensation is not finite. This is the
float64 group sum of pandas 3.0.x (``pandas._libs.groupby.group_sum``), which
the engine check used up to riskquant 0.2.0 (F-3). The reset changes a total
only after a square or a running total overflows to infinity: it keeps such a
total at +inf, where the unreset state gives NaN, and NaN passes a ``>``
bound test. pandas 2.2.3 resets on NaN only; the two differ only when a
compensation is +-inf.

``x * x`` equals the float64 ``x ** 2`` of the former engine expression bit for
bit: both are one correctly rounded IEEE 754 product.

Not used here, by design: ``np.add.reduceat`` (sequential, F-7), ``math.fsum``
(exact) and the Polars group sum (F-4) give other bits near the bound. The
builder sum that gives ``idio_weight`` stays ``np.add.reduceat`` (F-7); it is
not an admission decision and is outside this module.

Cost: one pass, O(n) time, O(g) state for g obligors. The loop runs on Python
floats, which are IEEE 754 binary64, so the bits equal those of a float64
array loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["LoadingSumSq", "loading_sum_sq"]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class LoadingSumSq:
    """Output of `loading_sum_sq`.

    Attributes
    ----------
    running : NDArray[np.float64], shape (n,)
        For each input row, the total of that row's obligor after the row.
        Rule 14 uses it for the row loc.
    uids : NDArray[np.int64], shape (g,)
        The distinct obligor uids in first-seen order.
    totals : NDArray[np.float64], shape (g,)
        The final total of each uid of `uids`, in the same order.
    """

    running: NDArray[np.float64]
    uids: NDArray[np.int64]
    totals: NDArray[np.float64]


def loading_sum_sq(
    *, obligor_uid: NDArray[np.int64], loading: NDArray[np.float64]
) -> LoadingSumSq:
    """Kahan sum of ``loading ** 2`` per obligor, in row order (C.5).

    Parameters
    ----------
    obligor_uid : NDArray[np.int64], shape (n,)
        The obligor of each row. Rows of one obligor need not be adjacent.
    loading : NDArray[np.float64], shape (n,)
        The loading of each row. Every value must be finite.

    Returns
    -------
    LoadingSumSq
        Running totals per row, uids in first-seen order and their final
        totals. Each array is C-contiguous and writeable.

    Raises
    ------
    TypeError
        An input is not a numpy array of the stated dtype. The function does
        not cast: a cast can wrap an integer or round a float silently.
    ValueError
        An input is not one-dimensional, the two lengths differ, or a loading
        is NaN or infinite. Both callers admit finite loadings only; a
        non-finite loading here is a caller defect.
    """
    _require_vector(obligor_uid, np.int64, name="obligor_uid")
    _require_vector(loading, np.float64, name="loading")
    if obligor_uid.shape != loading.shape:
        raise ValueError(
            f"loading_sum_sq: obligor_uid has shape {obligor_uid.shape} and "
            f"loading has shape {loading.shape}; one value per row is required."
        )
    with np.errstate(all="ignore"):
        finite = np.isfinite(loading)
    if not bool(finite.all()):
        row = int(np.flatnonzero(~finite)[0])
        raise ValueError(
            f"loading_sum_sq: loading at row {row} is {float(loading[row])!r}; "
            "every loading must be finite."
        )

    slot_of: dict[int, int] = {}
    totals: list[float] = []
    compensations: list[float] = []
    running: list[float] = []
    for uid, x in zip(obligor_uid.tolist(), loading.tolist(), strict=True):
        slot = slot_of.get(uid)
        if slot is None:
            slot = slot_of[uid] = len(totals)
            totals.append(0.0)
            compensations.append(0.0)
        total = totals[slot]
        y = x * x - compensations[slot]
        t = total + y
        compensation = (t - total) - y
        compensations[slot] = compensation if math.isfinite(compensation) else 0.0
        totals[slot] = t
        running.append(t)

    return LoadingSumSq(
        running=np.array(running, dtype=np.float64),
        uids=np.array(list(slot_of), dtype=np.int64),
        totals=np.array(totals, dtype=np.float64),
    )


def _require_vector(arr: object, dtype: type[np.generic], *, name: str) -> None:
    if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype):
        got = arr.dtype if isinstance(arr, np.ndarray) else type(arr).__name__
        raise TypeError(
            f"loading_sum_sq: {name} must be a numpy array of dtype "
            f"{np.dtype(dtype)}, got {got}."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"loading_sum_sq: {name} must be one-dimensional, got shape {arr.shape}."
        )
