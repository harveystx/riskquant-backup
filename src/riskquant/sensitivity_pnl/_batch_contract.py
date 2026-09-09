# src/riskquant/sensitivity_pnl/_batch_contract.py
"""
Shared caller-buffer validation for streaming sensitivity engines.

`SensitivityPnlEngine` (PnL) and `TaylorValidityEngine` (linear/quadratic split
diagnostics) both stream batches of caller-supplied levels or displacement
through the SAME compiled book, against the SAME kernels' bounds assumptions —
`kernel_build_displacement`, `kernel_gather_displacement`, and any kernel that
consumes the `(Ka, B)` displacement scratch they produce. Both therefore need
the IDENTICAL bounds gate: dtype, contiguity, shape, batch/scenario-offset
arithmetic, and — critically — the LOG-convention non-positive-level guard
that stops a silent NaN from reaching an output grid.

Extracted here, once, rather than written twice, so the two engines cannot
silently diverge on this logic as either one changes in the future. The golden
master's `test_both_engines_reject_the_same_malformed_batch_the_same_way` pins
that guarantee directly rather than trusting the extraction.
"""

from __future__ import annotations

__all__ = [
    "check_2d_batch_source",
    "derive_min_source_rows",
    "derive_positive_level_rows",
    "require_positive_levels_for_log_factors",
    "validate_base_levels_for_ratio_factors",
]

from typing import Any

import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import (
    POSITIVE_LEVEL_DISPLACEMENT_IDS,
    RATIO_DISPLACEMENT_IDS,
)


def derive_min_source_rows(book: CompiledSensitivityBook) -> int:
    """
    Highest levels/displacement row any kernel will dereference for `book`.

    Derived from the active-factor axis, NOT from `nnz`: a cross-only book has
    `nnz == 0` and a fully populated active-factor axis.
    """
    return book.max_level_uix + 1


def derive_positive_level_rows(
    book: CompiledSensitivityBook,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Levels rows (and their `factor_id`s) whose displacement is undefined at a
    non-positive value — i.e. the book's LOG-convention factors.

    Resolved once per engine construction, so a book with no LOG factors — the
    common case — pays a single `.size` test per batch and nothing else.
    """
    positive_mask = np.isin(
        book.displacement_convention_ids,
        np.asarray(POSITIVE_LEVEL_DISPLACEMENT_IDS, dtype=np.int8),
    )
    return (
        np.ascontiguousarray(book.active_level_uix[positive_mask]),
        np.ascontiguousarray(book.active_factor_ids[positive_mask]),
    )


def validate_base_levels_for_ratio_factors(
    *,
    book: CompiledSensitivityBook,
    base_levels: npt.NDArray[np.float64],
    min_source_rows: int,
) -> None:
    """
    `base_levels` shape/dtype/contiguity, plus positivity for RELATIVE/LOG
    factors. RELATIVE and LOG displacements divide by the base level;
    `error_model="numpy"` emits inf rather than raising, so this gate is the
    only thing standing between a zero base level and a silent inf in the
    output.
    """
    if base_levels.dtype != np.float64 or base_levels.ndim != 1:
        raise ValueError("base_levels must be 1-D float64")
    if not base_levels.flags.c_contiguous:
        raise ValueError("base_levels must be C-contiguous")
    if base_levels.shape[0] < min_source_rows:
        raise ValueError(
            f"base_levels has {base_levels.shape[0]} rows; the book "
            f"dereferences row {min_source_rows - 1}"
        )

    ratio_mask = np.isin(
        book.displacement_convention_ids,
        np.asarray(RATIO_DISPLACEMENT_IDS, dtype=np.int8),
    )
    if not ratio_mask.any():
        return
    bases = base_levels[book.active_level_uix[ratio_mask]]
    valid = np.isfinite(bases) & (bases > 0.0)
    if not bool(valid.all()):
        bad = book.active_factor_ids[ratio_mask][~valid]
        raise ValueError(
            "RELATIVE / LOG displacement requires a finite, strictly "
            f"positive base level; violated by factor_id(s) {bad[:10].tolist()}"
        )


def require_positive_levels_for_log_factors(
    *,
    positive_level_uix: npt.NDArray[np.int32],
    positive_level_factor_ids: npt.NDArray[np.int32],
    shocked_levels: npt.NDArray[np.float64],
    batch_len: int,
    levels_offset: int,
) -> None:
    """
    Reject a batch that would make `ln(L / L0)` undefined.

    The base-level guard above is not sufficient: it fixes the denominator,
    but the NUMERATOR varies per scenario, and a level that starts positive
    can be shocked through zero — an absolutely-shifted rate, a spread, a
    basis. `ln` of that is NaN or -inf, `error_model="numpy"` emits it
    silently, and the value then survives every downstream contract: a PnL
    container checks dtype and contiguity, not finiteness; summing a portfolio
    row propagates it to the whole scenario; and `np.sort` places NaN LAST, so
    a lower-tail VaR excludes the very scenario that produced it. A
    catastrophic loss becomes an invisible one. Measured on a 10-scenario
    book: a poisoned -100 catastrophe moved 99% VaR from -100 to -10.

    Cost is one contiguous scan per LOG factor row per batch — nothing when
    the book has none, and O(n_log * B) against a PnL kernel's O(nnz * B) when
    it does. Row views are taken one at a time rather than by fancy indexing,
    which would allocate an `(n_log, B)` copy per batch.
    """
    rows = positive_level_uix
    if rows.size == 0:
        return
    stop = levels_offset + batch_len
    worst = np.inf
    offender = -1
    for index in range(rows.shape[0]):
        row_min = float(shocked_levels[rows[index], levels_offset:stop].min())
        if row_min < worst:
            worst = row_min
            offender = index
    if worst > 0.0:
        return
    raise ValueError(
        "LOG displacement requires a strictly positive shocked level; "
        f"factor_id {int(positive_level_factor_ids[offender])} reaches "
        f"{worst!r} in this batch. ln of a non-positive level is NaN or -inf, "
        "which would reach the output grid silently and be sorted OUT of the "
        "lower tail. Route the factor to an ABSOLUTE convention, or repair "
        "the scenario upstream."
    )


def check_2d_batch_source(
    *,
    source: npt.NDArray[np.floating[Any]],
    dtype: type[np.floating[Any]],
    name: str,
    min_source_rows: int,
    batch_len: int,
    batch_size: int,
    source_offset: int,
    scenario_offset: int,
    out_num_scenarios: int,
) -> None:
    """
    The complete bounds gate for a caller-supplied 2-D batch source. Everything
    a kernel indexes into a caller array is bounded here; there are no checks
    downstream, and the kernels run under `boundscheck=False`.
    """
    if source.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if source.dtype != dtype:
        raise ValueError(f"{name} must be {np.dtype(dtype).name}")
    if not source.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if source.shape[0] < min_source_rows:
        raise ValueError(
            f"{name} has {source.shape[0]} factor rows; the book dereferences "
            f"row {min_source_rows - 1}. Pass the matrix over the SAME "
            "factor axis the book was compiled against."
        )
    if not 0 < batch_len <= batch_size:
        raise ValueError(f"batch_len must lie in (0, {batch_size}]")
    if source_offset < 0 or source_offset + batch_len > source.shape[1]:
        raise ValueError(f"batch overruns the {name} scenario axis")
    if scenario_offset < 0 or scenario_offset + batch_len > out_num_scenarios:
        raise ValueError("batch overruns the out scenario axis")
