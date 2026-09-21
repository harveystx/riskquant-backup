# src/riskquant/solutions/sensitivity_pnl/calculator.py
"""
A compiled book, held across scenario sets.

Why this exists
---------------
`compile_sensitivity_book` costs 293 ms on a 792k-entry portfolio, against a
2,200 ms kernel at S=4000. That ratio is fine for a batch run and absurd for
live risk, where the same book is priced against a new shock matrix every few
seconds and S is small: at S=250 the compile dominates the run it precedes.

The book depends on the factor AXIS, not on the shocks. So it is compiled once,
here, and every subsequent scenario set goes straight to the kernel.

    axis = FactorAxis.from_keys(factor_ids)
    calc = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=book_uids,
        )
    )
    while ticking:
        result = calc.run(axis.shocks(next_shock_matrix()))

What a cached book does NOT get to skip
---------------------------------------
Every scenario set is validated exactly as thoroughly as the first, and not
because `run` remembers to check. A `ScenarioShocks` cannot exist without having
passed dtype, contiguity, shape-against-axis, and finiteness checks in its own
`__post_init__`, so validation is a CONSTRUCTION INVARIANT of the market type
rather than a step on the run path. There is no argument to `run` that skips it,
no fast path that bypasses it, and no way for tick five thousand to be checked
less carefully than tick one.

The axis is the seam
--------------------
The one thing caching can newly get wrong is pricing a book against a market it
was not compiled for — factors reordered, added, or dropped between ticks. The
engine cannot see that: a `(K, S)` matrix with permuted rows returns a plausible
wrong number. So `run` compares the market's axis against the compiled one by
value, and a mismatch is a defect, not a warning. Building markets through
`calc.axis.shocks(...)` makes the mismatch unreachable, because there is then
only one place the row order can come from.

State
-----
The calculator is frozen and holds no mutable per-run state: no cached engine, no
reused output buffer, no counters. Two threads may price two scenario sets
against one calculator concurrently. The compiled book is read-only to the
kernels, and everything that varies per run — grid, engine, scratch — is
constructed inside `run` and dies with it.
"""

from __future__ import annotations

__all__ = [
    "SensitivityPnlCalculator",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.errors import DefectLog
from riskquant.solutions.sensitivity_pnl.market import FactorAxis, MarketScenarios
from riskquant.solutions.sensitivity_pnl.pipeline import (
    build_provenance,
    compile_book,
    make_engine,
    resolve_batch_size,
    scope_factor_meta,
    stream_scenarios,
)
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityBookSpec,
    SensitivityPnlProvenance,
    SensitivityPnlResult,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorDeclaration,
    validate_taylor_completeness,
)
from riskquant.solutions.sensitivity_pnl.validate import (
    validate_book_spec,
    validate_market_against_book,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SensitivityPnlCalculator:
    """A compiled sensitivity book, reusable across scenario sets."""

    book: CompiledSensitivityBook
    axis: FactorAxis
    """The factor axis the book was compiled against. Use it to build every
    market priced through this calculator: `calc.axis.shocks(matrix)`."""
    taylor_declaration: TaylorDeclaration | None = None
    """P1-A contract certified at compile; None = UNCERTIFIED (the production
    entry point refuses it)."""

    @classmethod
    def compile(cls, spec: SensitivityBookSpec) -> SensitivityPnlCalculator:
        """
        Validate and compile a book. The expensive step, paid once.

        Raises
        ------
        SensitivityPnlInputError
            Listing every defect in the spec, before anything is compiled.
        ValueError
            From the compiler, for contracts this layer deliberately does not
            duplicate — duplicate `(position, factor)` keys, a cross term with
            `i == j`.
        """
        factor_meta = scope_factor_meta(spec)
        validate_book_spec(spec, factor_meta=factor_meta)
        if spec.taylor_declaration is not None:
            validate_taylor_completeness(
                declaration=spec.taylor_declaration,
                sensitivities=spec.sensitivities,
                cross_sensitivities=spec.cross_sensitivities,
                position_uids=spec.position_uids,
            )
        return cls(
            book=compile_book(spec, factor_meta=factor_meta),
            axis=spec.factor_axis,
            taylor_declaration=spec.taylor_declaration,
        )

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def run(
        self, market: MarketScenarios, *, book_reused: bool = True
    ) -> SensitivityPnlResult:
        """
        Price the book against one scenario set into a freshly allocated block.

        Returns
        -------
        SensitivityPnlResult
            `.pnl_matrix` is a `PnlMatrix`: `(M,)` int64 position UIDs ascending,
            and an `(M, S)` C-contiguous float32 block whose column order matches
            the market's scenario order. A new block every call — no result ever
            aliases a previous one.

        Raises
        ------
        SensitivityPnlInputError
            If the market's factor axis is not the one the book was compiled
            against.
        ValueError
            From the engine, for a non-positive base level under a ratio
            convention.
        """
        self._require_axis(market)
        validate_market_against_book(book=self.book, axis=self.axis, market=market)
        num_scenarios = market.num_scenarios
        batch_size = resolve_batch_size(
            num_active_factors=self.book.num_active_factors,
            num_scenarios=num_scenarios,
        )
        grid = allocate_pnl_grid(
            block_position_uids=[self.book.position_uids], num_scenarios=num_scenarios
        )
        engine = make_engine(
            book=self.book,
            market=market,
            out=grid.block_view(0),
            batch_size=batch_size,
        )
        stream_scenarios(market=market, engine=engine, num_scenarios=num_scenarios)
        return SensitivityPnlResult(
            pnl_matrix=grid.to_pnl_matrix(),
            provenance=self._provenance(
                market=market, batch_size=batch_size, book_reused=book_reused
            ),
        )

    def run_into_grid(
        self,
        market: MarketScenarios,
        *,
        grid: PnlGrid,
        block_index: int,
        book_reused: bool = True,
    ) -> SensitivityPnlProvenance:
        """
        Price into a block of a grid the caller already owns.

        For the mixed portfolio — a full-reval block and a sensitivity block in
        one allocation — and for a ticking caller who would rather overwrite one
        buffer than allocate a new `(M, S)` block per tick. The caller owns the
        buffer and therefore owns knowing it is being overwritten, which is the
        difference between this and a calculator that silently reused one.

        Returns the provenance stamp; the PnL lands in the caller's grid.
        """
        log = DefectLog()
        if grid.num_scenarios != market.num_scenarios:
            log.add(
                source="market",
                problem=(
                    f"{market.num_scenarios} scenario(s) against a grid of "
                    f"{grid.num_scenarios}."
                ),
                remedy=(
                    "every block of a grid shares one scenario axis; a short "
                    "block would leave uninitialised columns."
                ),
            )
            log.raise_if_any()
        self._require_axis(market)
        validate_market_against_book(book=self.book, axis=self.axis, market=market)

        registered = grid.block_uids(block_index)
        if not bool(np.array_equal(self.book.position_uids, registered)):
            log.add(
                source="position_uids",
                problem=(
                    f"the compiled book has {self.book.num_positions} row(s); "
                    f"block {block_index} was registered with "
                    f"{registered.shape[0]}, and the two UID arrays differ."
                ),
                remedy=(
                    "register the block with this calculator's `position_uids`. "
                    "Rows outside the compiled set would keep the np.empty "
                    "garbage the grid was allocated with, which is "
                    "indistinguishable from a PnL downstream."
                ),
            )
            log.raise_if_any()

        batch_size = resolve_batch_size(
            num_active_factors=self.book.num_active_factors,
            num_scenarios=grid.num_scenarios,
        )
        engine = make_engine(
            book=self.book,
            market=market,
            out=grid.block_view(block_index),
            batch_size=batch_size,
        )
        stream_scenarios(market=market, engine=engine, num_scenarios=grid.num_scenarios)
        return self._provenance(
            market=market, batch_size=batch_size, book_reused=book_reused
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def position_uids(self) -> npt.NDArray[np.int64]:
        """`(M,)` int64 ascending — the row set every result will carry."""
        uids: npt.NDArray[np.int64] = self.book.position_uids
        return uids

    @property
    def factor_keys(self) -> pl.Series:
        """The caller's own factor keys, in scenario-matrix row order."""
        return self.axis.keys

    @property
    def num_positions(self) -> int:
        return self.book.num_positions

    @property
    def num_active_factors(self) -> int:
        """Factors the book touches. Sizes the kernel scratch; not the axis width."""
        return self.book.num_active_factors

    @property
    def num_first_order_entries(self) -> int:
        return self.book.nnz

    @property
    def num_cross_entries(self) -> int:
        return self.book.nnz_cross

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_axis(self, market: MarketScenarios) -> None:
        if self.axis.matches(market.axis):
            return
        log = DefectLog()
        log.add(
            source="market.factor_keys",
            problem=_axis_mismatch(compiled=self.axis, supplied=market.axis),
            remedy=(
                "a cached book is bound to one factor axis; pricing it against "
                "another would gather the wrong scenario rows and return a "
                "plausible, wrong number. Build markets with "
                "`calculator.axis.shocks(...)` / `.levels(...)`, or recompile "
                "the book when the factor universe genuinely changes."
            ),
        )
        log.raise_if_any()

    def _provenance(
        self, *, market: MarketScenarios, batch_size: int, book_reused: bool
    ) -> SensitivityPnlProvenance:
        return build_provenance(
            market=market,
            axis=self.axis,
            book=self.book,
            batch_size=batch_size,
            book_reused=book_reused,
        )


def _axis_mismatch(*, compiled: FactorAxis, supplied: FactorAxis) -> str:
    """
    Say what actually differs, not merely that something does.

    "a 2-factor axis that is not the 2-factor axis this book was compiled
    against" is a tautology, and a caller who has just reordered two factor rows
    learns nothing from it. Length, dtype, and the first differing row are what
    turn the message into a fix.
    """
    if compiled.keys.dtype != supplied.keys.dtype:
        return (
            f"factor keys of dtype {supplied.keys.dtype}; the book was compiled "
            f"against {compiled.keys.dtype} keys."
        )
    if compiled.num_factors != supplied.num_factors:
        return (
            f"{supplied.num_factors} factors; the book was compiled against "
            f"{compiled.num_factors}."
        )
    for row, (want, got) in enumerate(
        zip(compiled.keys.to_list(), supplied.keys.to_list(), strict=True)
    ):
        if want != got:
            return (
                f"the same {compiled.num_factors} factors in a different row "
                f"order: row {row} is {got!r} here and {want!r} in the compiled "
                "book. Row order is the whole contract — the engine gathers by "
                "position, so a reordered axis silently prices the wrong factor."
            )
    return "a factor axis that differs from the compiled one."
