# src/riskquant/solutions/sensitivity_pnl/schemas.py
"""
The pinned contracts of this solution.

    CONTRACT A (book)    position sensitivities + factor metadata + the factor
                         axis they resolve against + the position row set the
                         output must have. Compiled once.
    CONTRACT B (market)  a scenario set bound to that same axis. Supplied as
                         often as the caller likes.
    CONTRACT C (output)  a `PnlMatrix` — `(M,)` int64 position UIDs and an
                         `(M, S)` C-contiguous float32 matrix — which is exactly
                         what the full-reval historical-simulation model emits,
                         so downstream aggregation cannot tell the two apart.

A and B were originally fused into `SensitivityPnlRequest`, which is the right
shape for a batch run and the wrong shape for live risk: compiling the book cost
293 ms on a 792k-entry portfolio, and a fused request pays it again on every
tick. They are separable because compilation depends on the factor AXIS, not on
the shocks — so `SensitivityBookSpec` carries A, a `MarketScenarios` carries B,
and `SensitivityPnlRequest` remains their pairing for callers who genuinely have
one of each.

Why the book spec is a dataclass and not a Pydantic model
---------------------------------------------------------
`solutions.credit_var` and `solutions.credit_exposure` model their inputs as
`list[TradeRow]`, which is right for them: a few thousand trades, and a JSON
payload a front-end can post. A sensitivity book is a different animal — a
mid-size book is 10^6 `(position, factor)` entries, and a `list[BaseModel]` of
that length means a million Python objects and a million validation passes to
build arrays the compiler will immediately flatten again. The row-wise idiom is
not a house style to preserve here; it is a cost model that does not apply.

So bulk data crosses this boundary as Polars frames and is validated
column-wise, while the small serialisable artifact — provenance — stays
Pydantic, matching the siblings where the sibling shape makes sense.

Row-set stability
-----------------
`position_uids` declares the rows the output MUST have. It is the difference
between a drop-in replacement and a near-miss: a position whose sensitivities all
net to nothing carries no rows in the sensitivity frame, so a book compiled from
that frame alone would simply not have it, and a downstream join keyed on
`trade_uid` would drop it without comment. Declared positions with no
sensitivities emit exact zeros and hold their row.
"""

from __future__ import annotations

__all__ = [
    "SensitivityBookSpec",
    "SensitivityPnlProvenance",
    "SensitivityPnlRequest",
    "SensitivityPnlResult",
]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from riskquant.pnl.results import PnlMatrix
from riskquant.sensitivity_pnl.compile import DuplicatePolicy
from riskquant.solutions.sensitivity_pnl.market import FactorAxis, MarketScenarios
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorDeclaration


@dataclass(frozen=True, slots=True, kw_only=True)
class SensitivityBookSpec:
    """
    Everything needed to compile a book, and nothing that varies per scenario set.

    Parameters
    ----------
    sensitivities
        `position_id, factor_id, first_order, second_order`. One row per
        `(position, factor)` pair the position is sensitive to; absence means no
        exposure. `second_order` is the pure `d2V/dx2` — zero for a multilinear
        payoff, and zero is not the same as absent.
    factor_meta
        `factor_id, displacement_convention_id, quote_base`. One row per factor
        the book references; rows for factors outside the axis are ignored, so
        the house-wide table can be passed as-is. The convention is the space the
        sensitivity was differentiated in, and getting it wrong is silent, so it
        is required rather than defaulted.
    factor_axis
        The factor keys naming the rows of every scenario matrix this book will
        be priced against. Build it once with `FactorAxis.from_keys`.
    position_uids
        The exact row set of the output matrix, ascending. Positions carrying no
        sensitivities emit zeros rather than vanishing. Omit to take the row set
        from the sensitivity frames, which is convenient and not drop-in safe.
    cross_sensitivities
        `position_id, factor_id_i, factor_id_j, cross_second_order`. The FULL
        mixed partial for each unordered pair, listed once — the kernel does not
        double it, and `i == j` is diagonal curvature that belongs in
        `second_order`.
    duplicate_policy
        `RAISE` treats a repeated key as a fanned-out upstream join. `SUM` is for
        feeds that are leg-level by construction, where coefficients add exactly.
        It is a property of the feed, not a retry after `RAISE` fires.
    """

    sensitivities: pl.DataFrame
    factor_meta: pl.DataFrame
    factor_axis: FactorAxis
    position_uids: Sequence[int] | npt.NDArray[Any] | None = None
    cross_sensitivities: pl.DataFrame | None = None
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.RAISE
    #: P1-A: the Taylor contract this book claims (order, F(p), base binding).
    #: None = UNCERTIFIED: compiles for engine-level use, but the production
    #: entry point refuses it. Present = validated for completeness at compile.
    taylor_declaration: TaylorDeclaration | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SensitivityPnlRequest:
    """
    One book and one scenario set, for callers who have exactly one of each.

    The market supplies the factor axis, so a request never states it twice and
    the two can never disagree. Reach for `SensitivityPnlCalculator` instead when
    the same book is priced against many scenario sets.
    """

    sensitivities: pl.DataFrame
    factor_meta: pl.DataFrame
    market: MarketScenarios
    position_uids: Sequence[int] | npt.NDArray[Any] | None = None
    cross_sensitivities: pl.DataFrame | None = None
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.RAISE

    @property
    def num_scenarios(self) -> int:
        return self.market.num_scenarios

    def book_spec(self) -> SensitivityBookSpec:
        """Project out the scenario-independent half, on the market's own axis."""
        return SensitivityBookSpec(
            sensitivities=self.sensitivities,
            factor_meta=self.factor_meta,
            factor_axis=self.market.axis,
            position_uids=self.position_uids,
            cross_sensitivities=self.cross_sensitivities,
            duplicate_policy=self.duplicate_policy,
        )


class SensitivityPnlProvenance(BaseModel):
    """Audit stamp: enough to reproduce a number, and to explain a slow run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str
    market_contract: str
    """`displacement` (standalone) or `levels` (integrated with full reval)."""

    book_reused: bool
    """False when this run compiled the book, True when it came from a calculator
    that had already compiled it. A ticking run should show True after the first
    tick; if it does not, the book is being recompiled and the cache is not
    being hit."""

    factor_key_encoding: str
    """`direct` when the axis keys are a sorted integer universe handed straight
    to the compiler, `slot` when they were relabelled to row positions here."""

    num_positions: int
    num_scenarios: int
    num_market_factors: int
    num_active_factors: int
    """Factors the book actually touches. The kernel's displacement scratch is
    sized by this, not by the market universe."""

    num_first_order_entries: int
    num_cross_entries: int
    num_positions_without_sensitivities: int
    """Declared positions emitting exact zeros — a hedged book, or a coverage gap."""

    batch_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SensitivityPnlResult:
    """The `(M, S)` PnL block and its audit stamp (Contract C)."""

    pnl_matrix: PnlMatrix
    provenance: SensitivityPnlProvenance

    @property
    def position_uids(self) -> npt.NDArray[np.int64]:
        uids: npt.NDArray[np.int64] = self.pnl_matrix.trade_uids
        return uids

    @property
    def pnl(self) -> npt.NDArray[np.float32]:
        """`(M, S)` float32, C-contiguous. Row order matches `position_uids`."""
        matrix: npt.NDArray[np.float32] = self.pnl_matrix.pnl
        return matrix

    def row_for(self, position_uid: int) -> npt.NDArray[np.float32]:
        """
        The `(S,)` scenario vector for one position — a view, not a copy.

        Deliberately the only row-level accessor. There is no `to_dataframe()`:
        an `(M, S)` block at M=50k, S=10k is 2 GB as float32 and would be 4 GB as
        a float64 frame, so a convenience that quietly doubles peak memory does
        not belong on a result object.
        """
        uids = self.pnl_matrix.trade_uids
        row = int(np.searchsorted(uids, position_uid))
        if row >= uids.shape[0] or int(uids[row]) != position_uid:
            raise KeyError(
                f"position_uid {position_uid} is not a row of this PnL block "
                f"({uids.shape[0]} rows, {uids[0]}..{uids[-1]})"
            )
        view: npt.NDArray[np.float32] = self.pnl_matrix.pnl[row]
        return view
