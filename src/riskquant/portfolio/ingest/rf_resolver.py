# src/riskquant/portfolio/ingest/rf_resolver.py
"""
Vectorized resolution of human-facing risk-factor name strings into the int32
Universe Indices (UIX) consumed by the compiled blocks and JIT kernels.

Sibling of `fx_resolver.py`: that module resolves FX-pair strings; this one
resolves single-name risk factors (underlyings, discount curves, price/vol
factors). Together they are the layer where "strings die early" — every name
column is collapsed to a flat C-contiguous np.int32 array before any data
leaves the ingest edge.

Pandas migration note
---------------------
The prior ``pd.Series(names).map(mapping)`` calls are replaced with numpy
dict lookups using a unique sentinel object. This preserves the failure-mode
contract: populated-but-unmapped names HARD-FAIL before any narrowing.

Failure-mode contract
---------------------
Both `resolve_required_ids` and `resolve_optional_ids` HARD-FAIL on
populated-but-unmapped names. The optional path used to swallow these
silently via NaN -> platform-MIN_INT32 coercion, corrupting the int32 ids
array on typos. That silent path is gone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from riskquant.exceptions import PrettyKeyError

# A canonicalized string column uses the empty string as its missing sentinel
# (canonicalize.py maps NA -> "" on extraction). "" is never a valid RF name.
_BLANK: str = ""

# Unique sentinel object for "name not in mapping". Using object() avoids
# any accidental match against valid integer mapping values.
_MISS = object()


def _detect_unmapped(
    names: npt.NDArray[np.str_],
    mapped_null: npt.NDArray[np.bool_],
    blank_mask: npt.NDArray[np.bool_],
    *,
    ctx: str,
) -> None:
    """Raise on names that are populated but absent from the lookup table."""
    unmapped = mapped_null & ~blank_mask
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

    raw_mapped = [mapping.get(n, _MISS) for n in names]
    mapped_null = np.array([v is _MISS for v in raw_mapped], dtype=bool)

    _detect_unmapped(names, mapped_null, blank_mask, ctx=ctx)

    if blank_mask.any():
        bad = np.flatnonzero(blank_mask)[:10].tolist()
        raise PrettyKeyError(f"{ctx}: required name missing at rows {bad}.")

    # No misses survive past the guards above -> direct int64 → int32 is safe.
    arr = np.ascontiguousarray(
        np.array([v for v in raw_mapped], dtype=np.int64).astype(np.int32)
    )
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
    Detection runs before narrowing. The prior pandas path converted
    NaN to platform-MIN_INT32 silently on typos (``mapped.astype(np.float64)
    [populated].astype(np.int32)`` produced garbage). This path is gone.
    """
    n = names.shape[0]
    blank_mask = names == _BLANK

    raw_mapped = [mapping.get(n, _MISS) for n in names]
    mapped_null = np.array([v is _MISS for v in raw_mapped], dtype=bool)

    # Detect populated-but-unmapped BEFORE narrowing.
    _detect_unmapped(names, mapped_null, blank_mask, ctx=ctx)

    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(sentinel))

    populated = ~blank_mask
    if populated.any():
        pop_idx = np.flatnonzero(populated)
        # All populated entries are valid (guards above): extract directly.
        out[pop_idx] = np.array(
            [raw_mapped[i] for i in pop_idx], dtype=np.int64
        ).astype(np.int32)

    return out  # np.empty is already C-contiguous
