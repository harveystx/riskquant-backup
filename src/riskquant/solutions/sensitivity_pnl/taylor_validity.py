# src/riskquant/solutions/sensitivity_pnl/taylor_validity.py
"""
Taylor-validity, at the seam.

`riskquant.sensitivity_pnl.validity` answers "how much of this book's PnL is
riding on curvature" in the compiled book's own terms — position UIDs (already
caller-facing) and compiled factor slots (not caller-facing under SLOT
encoding, where a slot is a row position, not the string key a desk recognises).
This module is the same translation `SensitivityPnlResult` already performs for
PnL: bind to the SAME compiled book and axis a `SensitivityPnlCalculator` holds,
apply the SAME axis-mismatch and market validation `calculator.run` applies, and
hand back a report keyed in the caller's own terms.

Two entry points, mirroring `validity.py`'s two diagnostics:

    assess_taylor_validity(calculator=calc, market=market, threshold=0.20)
        SCENARIO-DEPENDENT. Same market contract as `calculator.run`; same
        axis-binding guarantee — a market built any other way than
        `calc.axis.shocks(...)` / `.levels(...)` is rejected, not silently
        mispriced.

    assess_characteristic_displacement(calculator=calc)
        SCENARIO-FREE. Needs only the compiled book; translates each entry's
        breakdown displacement back to the caller's own factor key.

Neither performs, or requires, a full revaluation. See
`riskquant.sensitivity_pnl.validity`'s module docstring for exactly what the
underlying metric can and cannot prove.
"""

from __future__ import annotations

__all__ = [
    "CharacteristicDisplacementReport",
    "TaylorValidityReport",
    "assess_characteristic_displacement",
    "assess_taylor_validity",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl.validity import (
    CharacteristicDisplacement,
    TaylorValidityGrid,
    TaylorValiditySummary,
    characteristic_displacement,
    summarize_taylor_validity,
)
from riskquant.solutions.sensitivity_pnl.calculator import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.errors import DefectLog
from riskquant.solutions.sensitivity_pnl.market import MarketScenarios
from riskquant.solutions.sensitivity_pnl.pipeline import (
    build_provenance,
    make_validity_engine,
    resolve_batch_size,
    stream_scenarios,
)
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance
from riskquant.solutions.sensitivity_pnl.validate import validate_market_against_book

# =========================================================================
# SCENARIO-DEPENDENT
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class TaylorValidityReport:
    """The `(M, S)` linear/quadratic split and its per-position rollup, stamped
    with the same audit provenance a PnL result carries."""

    grid: TaylorValidityGrid
    summary: TaylorValiditySummary
    provenance: SensitivityPnlProvenance

    @property
    def position_uids(self) -> npt.NDArray[np.int64]:
        return self.grid.position_uids

    def nonlinearity_fraction_for(self, position_uid: int) -> npt.NDArray[np.float64]:
        """The `(S,)` per-scenario fraction for one position — a fresh array,
        not a view: `nonlinearity_fraction()` computes the full `(M, S)` grid
        and this indexes one row of it, unlike `SensitivityPnlResult.row_for`,
        which has an underlying `(M, S)` array to slice a view from directly."""
        uids = self.grid.position_uids
        row = int(np.searchsorted(uids, position_uid))
        if row >= uids.shape[0] or int(uids[row]) != position_uid:
            raise KeyError(
                f"position_uid {position_uid} is not a row of this report "
                f"({uids.shape[0]} rows, {uids[0]}..{uids[-1]})"
            )
        fraction: npt.NDArray[np.float64] = self.grid.nonlinearity_fraction()[row]
        return fraction


def _require_matching_axis(
    *, calculator: SensitivityPnlCalculator, market: MarketScenarios
) -> None:
    """
    Same guarantee `SensitivityPnlCalculator._require_axis` gives `run`: a
    market built against a different axis is a defect, not a plausible wrong
    number. Kept independent of that private method rather than reaching into
    it — this module owns its own axis check on the same public
    `FactorAxis.matches` calculator.run already relies on.
    """
    if calculator.axis.matches(market.axis):
        return
    log = DefectLog()
    log.add(
        source="market.factor_keys",
        problem=(
            f"a {market.axis.num_factors}-factor axis that does not match the "
            f"{calculator.axis.num_factors}-factor axis this calculator was "
            "compiled against."
        ),
        remedy=(
            "build markets with `calculator.axis.shocks(...)` / `.levels(...)`, "
            "the same axis every other call through this calculator uses."
        ),
    )
    log.raise_if_any()


def assess_taylor_validity(
    *,
    calculator: SensitivityPnlCalculator,
    market: MarketScenarios,
    threshold: float = 0.20,
) -> TaylorValidityReport:
    """
    Run the linear/quadratic split against one scenario set, through the SAME
    compiled book and axis a `SensitivityPnlCalculator` holds.

    Raises
    ------
    SensitivityPnlInputError
        If the market's factor axis is not the one the book was compiled
        against, or the market fails the same checks `calculator.run` applies
        (including the LOG-convention non-positive-shocked-level guard).
    """
    _require_matching_axis(calculator=calculator, market=market)
    validate_market_against_book(
        book=calculator.book, axis=calculator.axis, market=market
    )

    num_scenarios = market.num_scenarios
    batch_size = resolve_batch_size(
        num_active_factors=calculator.book.num_active_factors,
        num_scenarios=num_scenarios,
    )
    linear_out = np.empty(
        (calculator.book.num_positions, num_scenarios), dtype=np.float32, order="C"
    )
    quadratic_out = np.empty(
        (calculator.book.num_positions, num_scenarios), dtype=np.float32, order="C"
    )
    engine = make_validity_engine(
        book=calculator.book,
        market=market,
        linear_out=linear_out,
        quadratic_out=quadratic_out,
        batch_size=batch_size,
    )
    stream_scenarios(market=market, engine=engine, num_scenarios=num_scenarios)

    grid = TaylorValidityGrid(
        position_uids=calculator.book.position_uids,
        linear_term=linear_out,
        quadratic_term=quadratic_out,
    )
    provenance = build_provenance(
        market=market,
        axis=calculator.axis,
        book=calculator.book,
        batch_size=batch_size,
        book_reused=True,
    )
    return TaylorValidityReport(
        grid=grid,
        summary=summarize_taylor_validity(grid, threshold=threshold),
        provenance=provenance,
    )


# =========================================================================
# SCENARIO-FREE
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CharacteristicDisplacementReport:
    """`characteristic_displacement`'s output, with `active_factor_slot`
    translated into the caller's own factor keys."""

    position_uids: npt.NDArray[np.int64]
    factor_keys: list[object]
    x_star: npt.NDArray[np.float64]

    def sorted_by_breakdown(self) -> CharacteristicDisplacementReport:
        """Ascending `x_star`, `inf` last — the entries closest to breaking
        down first, for a report that reads top-to-bottom as a priority list."""
        order = np.argsort(self.x_star, kind="stable")
        return CharacteristicDisplacementReport(
            position_uids=self.position_uids[order],
            factor_keys=[self.factor_keys[i] for i in order],
            x_star=self.x_star[order],
        )


def assess_characteristic_displacement(
    calculator: SensitivityPnlCalculator,
) -> CharacteristicDisplacementReport:
    """
    Scenario-free early-warning scan, in the caller's own factor keys.

    `active_level_uix` — not the compiled `factor_id`, whose VALUE means
    something different under direct vs slot encoding — is the row-index form
    valid under both, and is what makes `axis.keys[...]` the correct lookup
    regardless of which encoding this calculator's axis uses.
    """
    result: CharacteristicDisplacement = characteristic_displacement(calculator.book)
    active_level_uix = calculator.book.active_level_uix
    keys = calculator.axis.keys.to_list()
    factor_keys = [
        keys[int(active_level_uix[slot])] for slot in result.active_factor_slot
    ]
    return CharacteristicDisplacementReport(
        position_uids=result.position_uids,
        factor_keys=factor_keys,
        x_star=result.x_star,
    )
