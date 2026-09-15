# src/riskquant/simlab/builder/shift_domain.py
"""
Shift-specific level-domain validation.

The change-side kernels compute a per-shift-type return from a historical level
series. Each shift type has a different mathematical domain; violating it would
otherwise produce a silent NaN/Inf shock (a P0 failure class). This edge check
runs once per `iter_batches` over the source columns the compute plan actually
reads, and fails fast with the offending factor id, shift type, and rule.

Domains (per the RiskQuant shift taxonomy):
  NO_SHIFT              : no constraint (the column is never read for returns)
  LOG_RETURN           : finite and strictly positive levels
  SIMPLE_RETURN        : finite and non-zero levels (denominator v_{t-1})
  DIFFERENCE           : finite levels
  ABSOLUTE_BASE_RETURN : finite and non-zero levels (denominator |v_{t-1}|)

This is NOT a global positive-levels requirement: DIFFERENCE and
ABSOLUTE_BASE_RETURN admit negative levels; only LOG_RETURN requires positivity.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
    SHIFT_TYPE_NAME_TO_ID_REGISTRY,
)

_ID_TO_NAME: dict[int, str] = {
    int(v): k for k, v in SHIFT_TYPE_NAME_TO_ID_REGISTRY.items()
}


def validate_shift_domain(
    *,
    values: npt.NDArray[np.float64],
    cols_sorted: npt.NDArray[np.int32],
    shift_type_ids: npt.NDArray[np.int8],
    factor_ids: npt.NDArray[np.int32],
) -> None:
    """
    Validate that every source column satisfies its shift type's level domain.

    Parameters
    ----------
    values
        `(T, K_total)` historical level matrix.
    cols_sorted
        `(K_req,)` column indices into `values`, one per unique shock.
    shift_type_ids
        `(K_req,)` shift-type id aligned with `cols_sorted`.
    factor_ids
        `(K_total,)` factor ids aligned with the columns of `values` (for
        diagnostics only).

    Raises
    ------
    ValueError
        On any non-finite level, a non-positive level under LOG_RETURN, or a
        zero level under SIMPLE_RETURN / ABSOLUTE_BASE_RETURN.
    """
    for i in range(int(cols_sorted.shape[0])):
        st = int(shift_type_ids[i])
        if st == int(SHIFT_NO_SHIFT_ID):
            continue

        col = int(cols_sorted[i])
        series = values[:, col]
        fid = int(factor_ids[col])
        name = _ID_TO_NAME.get(st, str(st))

        if not np.isfinite(series).all():
            raise ValueError(
                f"factor_id={fid} (shift_type={name}) has non-finite levels; "
                "every shift type requires finite level observations."
            )

        if st == int(SHIFT_LOG_RETURN_ID) and not bool((series > 0.0).all()):
            raise ValueError(
                f"factor_id={fid} shift_type=LOG_RETURN requires strictly "
                "positive levels (log domain); found a level <= 0."
            )
        if st in (
            int(SHIFT_SIMPLE_RETURN_ID),
            int(SHIFT_ABSOLUTE_BASE_RETURN_ID),
        ) and bool((series == 0.0).any()):
            raise ValueError(
                f"factor_id={fid} shift_type={name} requires non-zero levels "
                "(division by v_(t-1)); found a level == 0."
            )
        # DIFFERENCE: finiteness only, already checked above.
