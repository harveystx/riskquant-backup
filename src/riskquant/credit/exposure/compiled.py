# src/riskquant/credit/exposure/compiled.py
"""
Compiled exposure container.

`CompiledExposure` is the EAD-layer analogue of `CompiledPortfolio`: an ordered
set of per-model netting-set views. Each populated view contributes one
contiguous row-band to the eventual (n_netting_sets x n_scenarios) EAD matrix;
`view_offsets[i]` is its first row. `netting_set_uids` aggregates every view's
uids into one flat int64 array in band order — the join key into the JTD layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView

ViewT: TypeAlias = RcNettingSetView  # | SaccrNettingSetView | ...  (future)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledExposure:
    """EAD-ready exposure: an ordered set of per-model netting-set views."""

    # VIEWS: One per exposure model. Each has its own netting set membership, collateral, etc.
    replacement_cost: RcNettingSetView | None = None

    # DERIVED FIELDS: Populated at init by __post_init__. These are the fields the engine dispatches on.
    _views: tuple[ViewT, ...] = field(init=False, repr=False)
    _view_offsets: npt.NDArray[np.int64] = field(init=False, repr=False)
    _total_views: int = field(init=False, repr=False)
    _total_netting_sets: int = field(init=False, repr=False)
    _netting_set_uids: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered: tuple[ViewT | None, ...] = (self.replacement_cost,)
        views: tuple[ViewT, ...] = tuple(v for v in ordered if v is not None)
        total_views = len(views)

        # 1. Calculate the total size needed
        total_ns = sum(v.num_netting_sets for v in views)

        # 2. Allocate uninitialized arrays (faster than zeros)
        view_offsets = np.empty(total_views, dtype=np.int64)
        netting_set_uids = np.empty(total_ns, dtype=np.int64)

        # 3. Populate both arrays in a single, clean pass
        offset = 0
        for i, v in enumerate(views):
            view_offsets[i] = offset
            n = v.num_netting_sets
            netting_set_uids[offset : offset + n] = v.netting_set_uids
            offset += n

        object.__setattr__(self, "_views", views)
        object.__setattr__(self, "_view_offsets", view_offsets)
        object.__setattr__(self, "_total_views", total_views)
        object.__setattr__(self, "_total_netting_sets", total_ns)
        object.__setattr__(self, "_netting_set_uids", netting_set_uids)

    @property
    def views(self) -> tuple[ViewT, ...]:
        return self._views

    @property
    def view_offsets(self) -> npt.NDArray[np.int64]:
        return self._view_offsets

    @property
    def total_views(self) -> int:
        return self._total_views

    @property
    def total_netting_sets(self) -> int:
        return self._total_netting_sets

    @property
    def netting_set_uids(self) -> npt.NDArray[np.int64]:
        return self._netting_set_uids

    def iter_views(self) -> Iterator[ViewT]:
        yield from self._views

    def summary(self) -> str:
        """Return a console-friendly overview of compiled exposure layout."""

        lines: list[str] = []

        width = 90

        lines.append("=" * width)
        lines.append(f"{'COMPILED EXPOSURE SUMMARY':^{width}}")
        lines.append("=" * width)

        lines.append(f"Total Views        : {self._total_views:,}")
        lines.append(f"Total Netting Sets : {self._total_netting_sets:,}")

        lines.append("-" * width)
        lines.append(f"{'View Type':<40} {'Netting Sets':>15} {'Offset':>15}")
        lines.append("-" * width)

        if self._total_views == 0:
            lines.append(f"{'Empty Exposure':^{width}}")
        else:
            for offset, view in zip(
                self._view_offsets,
                self._views,
                strict=True,
            ):
                lines.append(
                    f"{type(view).__name__:<40} "
                    f"{view.num_netting_sets:>15,} "
                    f"{offset:>15,}"
                )

        lines.append("=" * width)

        return "\n".join(lines)

    def print_view_details(self, max_items: int | None = 10) -> None:
        """Cascade into each view and print its raw SoA arrays."""
        print("=" * 90)
        print(f"{f'VIEW DETAILS (MAX {max_items} ITEMS PER VIEW)':^90}")
        print("=" * 90 + "\n")
        for view in self._views:
            title = f"View: {type(view).__name__}"
            print("-" * len(title))
            print(title)
            print("-" * len(title))
            view.print(max_items=max_items)
        print("=" * 90 + "\n")
