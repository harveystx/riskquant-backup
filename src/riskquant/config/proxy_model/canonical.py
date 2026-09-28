# src/riskquant/config/proxy_model/canonical.py
"""
Deterministic, bit-exact canonicalisation of resolved Proxy V1 terms (D088 §6).

Input is already resolved: every term names a row (position in the ordered
V1 run-axis subset), a driver rf_id, a driver chart id and an authored binary64
loading. Names, containers, the as-of and the static configuration do not
reach this module.

Algorithm (D088 §6, steps 2-8; step 1 "resolve" is the caller's job):

  2. drop authored loadings that compare equal to 0.0 (this drops -0.0 too);
  3. group terms by (row, driver_rf_id, chart_id);
  4. order the loadings of a group by their raw binary64 bit pattern, read as
     an unsigned 64-bit integer (MSB-first bit-string order);
  5. sum each group left to right in binary64;
  6. drop a merged loading only if it compares equal to 0.0;
  7. report rows left without terms (the caller rejects them);
  8. emit terms sorted by (row, driver_rf_id, chart_id).

No tolerance, no rounding, no compensated or pairwise summation: the canonical
loading is exactly the binary64 value the sequence above produces. Hence
``0.1 + 0.2`` canonicalises to 0x1.3333333333334p-2, never to authored
``0.3`` (0x1.3333333333333p-2), and ``0.1 + 0.2 + (-0.3)`` survives as
2**-54. Canonicalisation is deterministic over the resolved term multiset; it is
not symbolic algebraic normalisation.

The group sum is an explicit left-to-right loop on purpose: NumPy reductions
may use pairwise summation, which would silently change the summation order.
This runs once per run at compile time over authored terms, never per scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["CanonicalTerms", "canonicalise_resolved_terms"]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CanonicalTerms:
    """
    Canonical terms in CSR form plus the rows that ended up empty.

    ``empty_rows`` lists row positions whose authored terms all vanished (zero
    authored loadings or exact cancellation). The caller decides the error text;
    a canonical model never contains an empty row.
    """

    row_term_offsets: npt.NDArray[np.int32]
    term_driver_rf_ids: npt.NDArray[np.int32]
    term_chart_ids: npt.NDArray[np.int8]
    term_loadings: npt.NDArray[np.float64]
    empty_rows: npt.NDArray[np.int64]
    non_finite_rows: npt.NDArray[np.int64]


def canonicalise_resolved_terms(
    *,
    num_rows: int,
    term_rows: npt.NDArray[np.int64],
    term_driver_rf_ids: npt.NDArray[np.int32],
    term_chart_ids: npt.NDArray[np.int8],
    term_loadings: npt.NDArray[np.float64],
) -> CanonicalTerms:
    """
    Canonicalise resolved terms into CSR form (D088 §6 steps 2-8).

    Parameters
    ----------
    num_rows
        Number of rows (V1 run-axis nodes). Rows are identified by position.
    term_rows
        Row position of each resolved term, in ``[0, num_rows)``.
    term_driver_rf_ids, term_chart_ids, term_loadings
        Resolved term content. Loadings must already be finite binary64.

    Returns
    -------
    CanonicalTerms
        Canonical CSR arrays. ``empty_rows`` are rows with no surviving term;
        ``non_finite_rows`` are rows whose merged loading overflowed to a
        non-finite value (finite authored values can sum to +/-inf).
    """
    n = term_rows.shape[0]
    if not (
        term_driver_rf_ids.shape[0] == n
        and term_chart_ids.shape[0] == n
        and term_loadings.shape[0] == n
    ):
        raise ValueError("resolved term arrays must share one length.")
    if n and (int(term_rows.min()) < 0 or int(term_rows.max()) >= num_rows):
        raise ValueError(f"term_rows must lie in [0, {num_rows}).")

    # Step 2: authored exact zeros (+0.0 and -0.0 compare equal to 0.0).
    keep = term_loadings != 0.0
    rows = np.ascontiguousarray(term_rows[keep], dtype=np.int64)
    drivers = np.ascontiguousarray(term_driver_rf_ids[keep], dtype=np.int32)
    charts = np.ascontiguousarray(term_chart_ids[keep], dtype=np.int8)
    loadings = np.ascontiguousarray(term_loadings[keep], dtype=np.float64)

    # Steps 3-4: group by (row, driver, chart); inside a group, order by the
    # raw binary64 bit pattern as an unsigned integer. np.lexsort keys run from
    # least to most significant, so the bit pattern is listed first.
    bits = loadings.view(np.uint64)
    order = np.lexsort((bits, charts, drivers, rows))

    out_rows: list[int] = []
    out_drivers: list[int] = []
    out_charts: list[int] = []
    out_loadings: list[float] = []
    non_finite: set[int] = set()

    m = order.shape[0]
    i = 0
    while i < m:
        k = int(order[i])
        row, driver, chart = int(rows[k]), int(drivers[k]), int(charts[k])
        # Step 5: left-to-right binary64 accumulation in bit-pattern order.
        acc = float(loadings[k])
        j = i + 1
        while j < m:
            kj = int(order[j])
            if (
                int(rows[kj]) != row
                or int(drivers[kj]) != driver
                or int(charts[kj]) != chart
            ):
                break
            acc = acc + float(loadings[kj])
            j += 1
        i = j
        if not math.isfinite(acc):
            non_finite.add(row)
            continue
        # Step 6: exact cancellation only.
        if acc == 0.0:
            continue
        out_rows.append(row)
        out_drivers.append(driver)
        out_charts.append(chart)
        out_loadings.append(acc)

    # Step 8: lexsort already produced (row, driver, chart) order; the CSR
    # offsets follow from per-row counts.
    counts = np.bincount(np.asarray(out_rows, dtype=np.int64), minlength=num_rows)
    offsets = np.zeros(num_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("canonical term count exceeds the int32 offset range.")

    return CanonicalTerms(
        row_term_offsets=offsets.astype(np.int32),
        term_driver_rf_ids=np.asarray(out_drivers, dtype=np.int32),
        term_chart_ids=np.asarray(out_charts, dtype=np.int8),
        term_loadings=np.asarray(out_loadings, dtype=np.float64),
        empty_rows=np.flatnonzero(counts == 0).astype(np.int64),
        non_finite_rows=np.asarray(sorted(non_finite), dtype=np.int64),
    )
