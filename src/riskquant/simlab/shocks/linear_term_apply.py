# src/riskquant/simlab/shocks/linear_term_apply.py
"""
Edge wrapper for the linear-term application kernel (Proxy V1 Slice 2).

`apply_linear_terms_to_base_levels` validates every numerical precondition of
D088 §13.5 before the kernel runs, then calls
`linear_term_kernel._kernel_apply_linear_terms`:

    X[n, b]   = sum_{t = off[n]}^{off[n+1] - 1} loadings[t] * float64(shocks[b, cols[t]])
    out[n, b] = T^-1_{kappa[n]}(base[n], X[n, b])

Inputs are primitive arrays only. Row order, term order and column indices
are taken as given: the canonical term order comes from the compiled proxy
model (D088 §6) and the column indices from world-column binding
(`riskquant.simlab.shocks.column_binding`). Nothing here resolves names,
charts, horizons, availability or sources, and nothing is repaired: every
violation raises. No input is copied or mutated; the kernel receives
zero-copy read-only views and writes only into the caller-owned ``out``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.shocks.linear_term_kernel import _kernel_apply_linear_terms

__all__ = ["APPLICABLE_TARGET_SHIFT_TYPE_IDS", "apply_linear_terms_to_base_levels"]

#: Target shift-type ids the numerical seam reconstructs: the three V1
#: charts (D088 §1) plus the two native ids (D025; D088 §13.4).
APPLICABLE_TARGET_SHIFT_TYPE_IDS: frozenset[int] = frozenset(
    int(x)
    for x in (
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
        SHIFT_ABSOLUTE_BASE_RETURN_ID,
        SHIFT_NO_SHIFT_ID,
    )
)


def apply_linear_terms_to_base_levels(
    *,
    base_levels: npt.NDArray[np.float64],
    row_term_offsets: npt.NDArray[np.int32],
    term_column_uix: npt.NDArray[np.int32],
    term_loadings: npt.NDArray[np.float64],
    target_shift_types: npt.NDArray[np.int8],
    shocks: npt.NDArray[np.float32],
    out: npt.NDArray[np.float64],
) -> None:
    """
    Aggregate row terms in shock space, then reconstruct target levels.

    Parameters
    ----------
    base_levels
        ``(N,)`` float64, C-contiguous. Base level of each row.
    row_term_offsets
        ``(N + 1,)`` int32, C-contiguous. Row ``n`` owns terms
        ``[off[n], off[n + 1])``; ``off[0] == 0``, non-decreasing,
        ``off[N] == T``.
    term_column_uix
        ``(T,)`` int32, C-contiguous. Shock column of each term, in
        ``[0, K)``.
    term_loadings
        ``(T,)`` float64, C-contiguous, finite. Loading of each term.
    target_shift_types
        ``(N,)`` int8, C-contiguous. Reconstruction id of each row: LOG,
        SIMPLE, DIFFERENCE, ABSOLUTE_BASE or NO_SHIFT. A row has zero terms
        if and only if its id is NO_SHIFT.
    shocks
        ``(B, K)`` float32, F-contiguous. ``K`` may be 0 when every row is
        NO_SHIFT.
    out
        ``(N, B)`` float64, C-contiguous, writable, not overlapping any
        input. Overwritten with the reconstructed levels.

    Raises
    ------
    TypeError
        An argument is not a NumPy array.
    ValueError
        Any dtype, rank, shape, layout, offset, index, loading, shift-type,
        term-count or aliasing violation.

    Notes
    -----
    Term sums are float64 in ascending term order (D088 §13.5). Results are
    bitwise reproducible across batch partitions and thread counts within one
    execution environment; they are not claimed bitwise equal across
    platforms (D088 §13.6).
    """
    _require(base_levels, "base_levels", np.float64, 1)
    _require(row_term_offsets, "row_term_offsets", np.int32, 1)
    _require(term_column_uix, "term_column_uix", np.int32, 1)
    _require(term_loadings, "term_loadings", np.float64, 1)
    _require(target_shift_types, "target_shift_types", np.int8, 1)
    _require(shocks, "shocks", np.float32, 2)
    _require(out, "out", np.float64, 2)

    for name, arr in (
        ("base_levels", base_levels),
        ("row_term_offsets", row_term_offsets),
        ("term_column_uix", term_column_uix),
        ("term_loadings", term_loadings),
        ("target_shift_types", target_shift_types),
    ):
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous.")
    if not shocks.flags.f_contiguous:
        raise ValueError(
            "shocks must be F-contiguous (B, K): the kernel reads each column "
            "down the scenario axis. No copy is made here."
        )
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous (N, B).")
    if not out.flags.writeable:
        raise ValueError("out must be writable.")

    n_rows = base_levels.shape[0]
    n_terms = term_column_uix.shape[0]
    n_scn, n_cols = shocks.shape

    if row_term_offsets.shape[0] != n_rows + 1:
        raise ValueError(
            f"row_term_offsets must have length N + 1 = {n_rows + 1}, "
            f"got {row_term_offsets.shape[0]}."
        )
    if target_shift_types.shape[0] != n_rows:
        raise ValueError(
            f"target_shift_types must have length N = {n_rows}, "
            f"got {target_shift_types.shape[0]}."
        )
    if term_loadings.shape[0] != n_terms:
        raise ValueError(
            f"term_loadings (length {term_loadings.shape[0]}) and "
            f"term_column_uix (length {n_terms}) must have one length."
        )
    if out.shape != (n_rows, n_scn):
        raise ValueError(f"out shape {out.shape} must be (N, B) = ({n_rows}, {n_scn}).")

    if int(row_term_offsets[0]) != 0:
        raise ValueError(
            f"row_term_offsets[0] must be 0, got {int(row_term_offsets[0])}."
        )
    if int(row_term_offsets[-1]) != n_terms:
        raise ValueError(
            f"row_term_offsets[-1] must equal the term count {n_terms}, "
            f"got {int(row_term_offsets[-1])}."
        )
    counts = np.diff(row_term_offsets)
    if n_rows and bool((counts < 0).any()):
        raise ValueError("row_term_offsets must be non-decreasing.")

    if n_terms:
        cmin, cmax = int(term_column_uix.min()), int(term_column_uix.max())
        if cmin < 0 or cmax >= n_cols:
            raise ValueError(
                f"term_column_uix must lie in [0, K) = [0, {n_cols}); "
                f"got range [{cmin}, {cmax}]."
            )
        with np.errstate(all="ignore"):
            finite = bool(np.isfinite(term_loadings).all())
        if not finite:
            raise ValueError("term_loadings must be finite (D088 §2).")

    if n_rows:
        allowed = np.array(sorted(APPLICABLE_TARGET_SHIFT_TYPE_IDS), dtype=np.int8)
        bad = ~np.isin(target_shift_types, allowed)
        if bool(bad.any()):
            raise ValueError(
                "target_shift_types contains ids outside "
                f"{sorted(APPLICABLE_TARGET_SHIFT_TYPE_IDS)}: "
                f"{np.unique(target_shift_types[bad]).tolist()}."
            )
        no_shift = target_shift_types == SHIFT_NO_SHIFT_ID
        empty = counts == 0
        if bool((empty & ~no_shift).any()):
            rows = np.flatnonzero(empty & ~no_shift)[:10].tolist()
            raise ValueError(
                f"rows {rows} have zero terms but are not NO_SHIFT; a zero-term "
                "row is admissible only for NO_SHIFT (D088 §13.5)."
            )
        if bool((no_shift & ~empty).any()):
            rows = np.flatnonzero(no_shift & ~empty)[:10].tolist()
            raise ValueError(
                f"NO_SHIFT rows {rows} carry terms; a NO_SHIFT row is a zero-term "
                "row and requires no shock column (D088 §13.4)."
            )

    inputs: tuple[tuple[str, npt.NDArray[Any]], ...] = (
        ("base_levels", base_levels),
        ("row_term_offsets", row_term_offsets),
        ("term_column_uix", term_column_uix),
        ("term_loadings", term_loadings),
        ("target_shift_types", target_shift_types),
        ("shocks", shocks),
    )
    for name, arr in inputs:
        if np.may_share_memory(out, arr):
            raise ValueError(f"out must not overlap {name}.")

    _kernel_apply_linear_terms(
        _read_only(base_levels),
        _read_only(row_term_offsets),
        _read_only(term_column_uix),
        _read_only(term_loadings),
        _read_only(target_shift_types),
        _read_only(shocks),
        out,
    )


def _require(arr: object, name: str, dtype: type[np.generic], ndim: int) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}.")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}.")


def _read_only(arr: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Zero-copy read-only view: one kernel signature for frozen and writable inputs."""
    view = arr.view()
    view.setflags(write=False)
    return view
