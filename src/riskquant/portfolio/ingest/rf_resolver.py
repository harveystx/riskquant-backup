# src/riskquant/portfolio/ingest/rf_resolver.py

"""
Vectorized resolution of human-facing risk-factor name strings into the int32
Universe Indices (UIX) consumed by the compiled blocks and JIT kernels.

Sibling of `fx_resolver.py`: that module resolves FX-pair strings; this one
resolves single-name risk factors (underlyings, discount curves, price/vol
factors). Together they are the layer where "strings die early" — every name
column is collapsed to a flat C-contiguous np.int32 array before any data
leaves the ingest edge.

All resolution is performed through pandas `Series.map` (a C-level hash join).
There are no Python row loops, no `.apply`, and no regex matching.

Failure-mode contract
---------------------
Both `resolve_required_ids` and `resolve_optional_ids` HARD-FAIL on
populated-but-unmapped names. The optional path used to swallow these
silently via `.astype(np.float64)[...].astype(np.int32)` (NaN -> platform
MIN_INT32), corrupting the int32 ids array on typos. That silent path is
gone: `_detect_unmapped` now runs before any narrowing for both helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.exceptions import PrettyKeyError

# A canonicalized string column uses the empty string as its missing sentinel
# (canonicalize.py maps NA -> "" on extraction). "" is never a valid RF name.
_BLANK: str = ""


def _detect_unmapped(
    names: npt.NDArray[np.str_],
    mapped: pd.Series,
    blank_mask: npt.NDArray[np.bool_],
    *,
    ctx: str,
) -> None:
    """Raise on names that are populated but absent from the lookup table."""
    unmapped = mapped.isna().to_numpy(copy=False) & ~blank_mask
    if not unmapped.any():
        return

    bad = sorted({str(v) for v in names[unmapped]})[:10]
    raise PrettyKeyError(
        f"{ctx}: {int(unmapped.sum())} name(s) could not be resolved to a "
        f"risk-factor universe index.\nUnresolved names:\n{bad}"
    )


def resolve_required_ids(
    names: npt.NDArray[np.str_],
    mapping: dict[str, int],
    *,
    ctx: str,
) -> npt.NDArray[np.int32]:
    """
    Resolve a name column where every row MUST carry a mappable name.

    Parameters
    ----------
    names
        Flat string array (blanks are treated as errors here).
    mapping
        Name -> universe-index lookup table.
    ctx
        Human-readable column label for diagnostics.
    """
    blank_mask = names == _BLANK
    mapped = pd.Series(names).map(mapping)
    _detect_unmapped(names, mapped, blank_mask, ctx=ctx)

    if blank_mask.any():
        bad = np.flatnonzero(blank_mask)[:10].tolist()
        raise PrettyKeyError(f"{ctx}: required name missing at rows {bad}.")
    # No NaNs survive past the guards above -> direct int32 narrowing is safe.
    arr = np.ascontiguousarray(mapped.to_numpy(dtype=np.int32))
    return cast(npt.NDArray[np.int32], arr)


def resolve_optional_ids(
    names: npt.NDArray[np.str_],
    mapping: Mapping[str, int | np.int32],
    *,
    sentinel: int | np.integer,
    ctx: str,
) -> npt.NDArray[np.int32]:
    """
    Resolve a name column where blank rows are legitimately absent.

    Blank rows collapse to `sentinel`. A populated-but-unmapped name is a
    hard error: it signals a typo or a stale universe rather than absence.

    Parameters
    ----------
    names
        Flat string array; blank rows ("") map to `sentinel`.
    mapping
        Name -> universe-index lookup table.
    sentinel
        Typed sentinel (from riskquant.constants) for "no linkage".
    ctx
        Human-readable column label for diagnostics.

    Notes
    -----
    The detection-before-narrowing ordering matters: `mapped.to_numpy(
    dtype=np.float64)[populated].astype(np.int32)` on a NaN element returns
    `MIN_INT32` rather than raising — that path used to silently corrupt
    optional id columns on typos. Detection now runs unconditionally
    before any narrowing.
    """
    n = names.shape[0]
    blank_mask = names == _BLANK
    mapped = pd.Series(names).map(mapping)

    # Detect populated-but-unmapped BEFORE narrowing. After this gate the
    # only NaN-bearing rows in `mapped` are the blank ones, and those are
    # masked out of the scatter below.
    _detect_unmapped(names, mapped, blank_mask, ctx=ctx)

    # Edge allocation; downstream kernels never touch this array directly.
    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(sentinel))

    populated = ~blank_mask
    if populated.any():
        # Safe int64 narrowing: `_detect_unmapped` guarantees no NaNs in
        # the populated subset. The float64 intermediate is gone — we go
        # straight through int64 to int32 to keep platform behaviour on
        # NaN coercion irrelevant.
        out[populated] = (
            mapped[populated].to_numpy(dtype=np.int64, copy=False).astype(np.int32)
        )

    return out  # np.empty is already C-contiguous
