# scripts/solutions/sensitivity_pnl/demo_e2e.py
"""
End-to-end tour of ``riskquant.solutions.sensitivity_pnl``.

Runs a small, fully hand-checkable book through every public path in turn:

  1. ONE-SHOT      one book, one scenario set, one `PnlMatrix`.
  2. CACHED        one book compiled once, priced against a stream of ticks.
  3. LEVELS        the integrated contract, driven by shocked levels rather than
                   displacements, with RELATIVE and ABSOLUTE conventions mixed.
  4. CO-ALLOCATED  a sensitivity block written into a grid shared with a
                   full-reval block, in one allocation.
  5. REJECTED      what a bad request looks like coming back.

The portfolio is two positions on the same two factors, so every number printed
can be checked against the closed form at the bottom of this file by hand.

    python scripts/solutions/sensitivity_pnl/demo_e2e.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    ScenarioShocks,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
    write_sensitivity_pnl,
)

# --- the book ------------------------------------------------------------
# V = volume * X_rep * S, so second order is zero and all curvature is the
# single cross term d2V/dS dX = volume. See scripts/.../verify_spot.py.
SPOT, FX = 42.50, 0.85
LONG_UID, HEDGE_UID, EMPTY_UID = 101, 202, 303
LONG_VOLUME, HEDGE_VOLUME = 50_000.0, -18_000.0

EQ, FXR = "EQ.ACME.SPOT", "FX.EURGBP"

SENSITIVITIES = pl.DataFrame(
    {
        "position_id": [LONG_UID, LONG_UID, HEDGE_UID, HEDGE_UID],
        "factor_id": [EQ, FXR, EQ, FXR],
        "first_order": [
            LONG_VOLUME * FX,
            LONG_VOLUME * SPOT,
            HEDGE_VOLUME * FX,
            HEDGE_VOLUME * SPOT,
        ],
        "second_order": [0.0, 0.0, 0.0, 0.0],
    }
)
CROSS_SENSITIVITIES = pl.DataFrame(
    {
        "position_id": [LONG_UID, HEDGE_UID],
        "factor_id_i": [EQ, EQ],
        "factor_id_j": [FXR, FXR],
        "cross_second_order": [LONG_VOLUME, HEDGE_VOLUME],
    }
)
#: Deliberately the house-wide table: it carries factors this book never touches,
#: and rows outside the axis are scoped away rather than raising.
FACTOR_META = pl.DataFrame(
    {
        "factor_id": [EQ, FXR, "IR.EUR.5Y", "CO.BRENT.FWD"],
        "displacement_convention_id": [
            int(DisplacementConvention.ABSOLUTE),
            int(DisplacementConvention.ABSOLUTE),
            int(DisplacementConvention.ABSOLUTE),
            int(DisplacementConvention.RELATIVE),
        ],
        "quote_base": [1.0, 1.0, 1.0e4, 1.0],
    }
)
#: EMPTY_UID carries no sensitivities. Declaring it holds its row and emits exact
#: zeros, which is what keeps a downstream join on trade_uid stable.
BOOK_UIDS = [LONG_UID, HEDGE_UID, EMPTY_UID]


def closed_form(volume: float, d_spot: float, d_fx: float) -> float:
    """Full revaluation. Bilinear, so second order is exact for any shock."""
    return volume * (FX + d_fx) * (SPOT + d_spot) - volume * FX * SPOT


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(result_pnl: np.ndarray, uids: np.ndarray, shocks: np.ndarray) -> None:
    print(
        f"  {'position':>10} "
        + "".join(f"{f'scn {i}':>14}" for i in range(shocks.shape[1]))
    )
    for row, uid in enumerate(uids):
        cells = "".join(f"{v:>14,.2f}" for v in result_pnl[row])
        print(f"  {uid:>10} {cells}")


# =========================================================================
# 1. ONE-SHOT
# =========================================================================


def demo_one_shot() -> np.ndarray:
    banner("1. ONE-SHOT — one book, one scenario set")
    shocks = np.array(
        [
            [2.50, -5.00, 0.00],  # EQ.ACME.SPOT
            [-0.03, 0.04, 0.00],  # FX.EURGBP
        ]
    )
    market = ScenarioShocks.from_matrix(factor_keys=[EQ, FXR], displacement=shocks)
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=SENSITIVITIES,
            factor_meta=FACTOR_META,
            cross_sensitivities=CROSS_SENSITIVITIES,
            market=market,
            position_uids=BOOK_UIDS,
        )
    )
    show(result.pnl, result.position_uids, shocks)
    print(
        f"\n  matrix: {result.pnl.shape} {result.pnl.dtype}, "
        f"C-contiguous={result.pnl.flags.c_contiguous}"
    )
    print(
        f"  {EMPTY_UID} carries no sensitivities and holds its row as exact zeros: "
        f"{bool(np.all(result.pnl[2] == 0.0))}"
    )

    for row, volume in ((0, LONG_VOLUME), (1, HEDGE_VOLUME)):
        engine = result.pnl[row].astype(np.float64)
        exact = np.array(
            [
                closed_form(volume, ds, dx)
                for ds, dx in zip(shocks[0], shocks[1], strict=True)
            ]
        )
        print(
            f"  row {row} max error vs full revaluation: {np.abs(engine - exact).max():.3e}"
        )

    p = result.provenance
    print(
        f"\n  provenance: book_reused={p.book_reused} encoding={p.factor_key_encoding} "
        f"Ka={p.num_active_factors}/{p.num_market_factors} batch={p.batch_size} "
        f"silent_positions={p.num_positions_without_sensitivities}"
    )
    return result.pnl


# =========================================================================
# 2. CACHED
# =========================================================================


def demo_cached(one_shot_pnl: np.ndarray) -> None:
    banner("2. CACHED — compile once, price a stream of ticks")
    axis = FactorAxis.from_keys([EQ, FXR])
    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=SENSITIVITIES,
            factor_meta=FACTOR_META,
            factor_axis=axis,
            cross_sensitivities=CROSS_SENSITIVITIES,
            position_uids=BOOK_UIDS,
        )
    )
    print(
        f"  compiled: {calculator.num_positions} positions, "
        f"{calculator.num_active_factors} active factors, "
        f"{calculator.num_first_order_entries} first-order + "
        f"{calculator.num_cross_entries} cross entries"
    )

    rng = np.random.default_rng(20260728)
    print(f"\n  {'tick':>6}{'d_spot':>10}{'d_fx':>10}{'long PnL':>16}{'hedge PnL':>16}")
    for tick in range(5):
        d_spot, d_fx = float(rng.normal(0, 3.0)), float(rng.normal(0, 0.02))
        # The axis is the factory, so a tick cannot be built on the wrong rows.
        result = calculator.run(axis.shocks(np.array([[d_spot], [d_fx]])))
        print(
            f"  {tick:>6}{d_spot:>10.3f}{d_fx:>10.4f}"
            f"{result.pnl[0, 0]:>16,.2f}{result.pnl[1, 0]:>16,.2f}"
        )

    replay = calculator.run(
        axis.shocks(np.array([[2.50, -5.00, 0.00], [-0.03, 0.04, 0.00]]))
    )
    print(
        f"\n  same shocks as the one-shot run give identical numbers: "
        f"{bool(np.array_equal(replay.pnl, one_shot_pnl))}"
    )
    print(f"  provenance book_reused={replay.provenance.book_reused}")

    # The one thing caching can newly get wrong.
    try:
        calculator.run(
            ScenarioShocks.from_matrix(
                factor_keys=[FXR, EQ], displacement=np.ones((2, 1))
            )
        )
    except SensitivityPnlInputError as error:
        print(f"\n  a market on a reordered axis is refused:\n{error}")


# =========================================================================
# 3. LEVELS
# =========================================================================


def demo_levels() -> None:
    banner("3. LEVELS — the integrated contract, mixed conventions")
    # Same payoff, but FX is now quoted per unit return, so its delta is rescaled
    # by the level and the convention says RELATIVE.
    sensitivities = pl.DataFrame(
        {
            "position_id": [LONG_UID, LONG_UID],
            "factor_id": [EQ, FXR],
            "first_order": [LONG_VOLUME * FX, LONG_VOLUME * SPOT * FX],
            "second_order": [0.0, 0.0],
        }
    )
    factor_meta = pl.DataFrame(
        {
            "factor_id": [EQ, FXR],
            "displacement_convention_id": [
                int(DisplacementConvention.ABSOLUTE),
                int(DisplacementConvention.RELATIVE),
            ],
            "quote_base": [1.0, 1.0],
        }
    )
    cross = pl.DataFrame(
        {
            "position_id": [LONG_UID],
            "factor_id_i": [EQ],
            "factor_id_j": [FXR],
            "cross_second_order": [LONG_VOLUME * FX],
        }
    )
    axis = FactorAxis.from_keys([EQ, FXR])
    d_spot = np.array([2.50, -5.00])
    d_fx = np.array([-0.03, 0.04])
    market = axis.levels(
        base_levels=np.array([SPOT, FX]),
        shocked_levels=np.vstack([SPOT + d_spot, FX + d_fx]),
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sensitivities,
            factor_meta=factor_meta,
            cross_sensitivities=cross,
            market=market,
        )
    )
    engine = result.pnl[0].astype(np.float64)
    exact = np.array(
        [closed_form(LONG_VOLUME, ds, dx) for ds, dx in zip(d_spot, d_fx, strict=True)]
    )
    print(f"  engine (levels, mixed conventions): {np.round(engine, 2)}")
    print(f"  full revaluation:                   {np.round(exact, 2)}")
    print(f"  max error: {np.abs(engine - exact).max():.3e}")
    print(f"  provenance market_contract={result.provenance.market_contract}")


# =========================================================================
# 4. CO-ALLOCATED
# =========================================================================


def demo_co_allocated() -> None:
    banner("4. CO-ALLOCATED — one grid shared with a full-reval block")
    full_reval_uids = np.array([1, 2], dtype=np.int64)
    grid = allocate_pnl_grid(
        block_position_uids=[full_reval_uids, np.array(BOOK_UIDS, dtype=np.int64)],
        num_scenarios=3,
    )
    # Stand-in for whatever the full-reval runner would have written.
    grid.block_view(0)[:] = np.array(
        [[10.0, 20.0, 30.0], [-5.0, -6.0, -7.0]], dtype=np.float32
    )

    shocks = np.array([[2.50, -5.00, 0.00], [-0.03, 0.04, 0.00]])
    provenance = write_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=SENSITIVITIES,
            factor_meta=FACTOR_META,
            cross_sensitivities=CROSS_SENSITIVITIES,
            market=ScenarioShocks.from_matrix(
                factor_keys=[EQ, FXR], displacement=shocks
            ),
            position_uids=BOOK_UIDS,
        ),
        grid=grid,
        block_index=1,
    )
    matrix = grid.to_pnl_matrix()
    print(f"  one allocation, {matrix.pnl.shape[0]} rows across 2 blocks:")
    show(matrix.pnl, matrix.trade_uids, shocks)
    print(
        f"\n  block 1 provenance: {provenance.num_positions} positions, "
        f"book_reused={provenance.book_reused}"
    )
    print(f"  to_pnl_matrix() is zero-copy: {np.shares_memory(matrix.pnl, grid.pnl)}")


# =========================================================================
# 5. REJECTED
# =========================================================================


def demo_rejected() -> None:
    banner("5. REJECTED — what a bad request looks like coming back")
    broken = pl.DataFrame(
        {
            "position_id": [101.0, None, 202.0],
            "factor_id": [EQ, FXR, "IR.GBP.10Y"],
            "first_order": [1.0, float("inf"), 3.0],
            "second_order": [0.0, 0.0, 0.0],
        }
    )
    try:
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=broken,
                factor_meta=FACTOR_META,
                market=ScenarioShocks.from_matrix(
                    factor_keys=[EQ, FXR], displacement=np.ones((2, 2))
                ),
            )
        )
    except SensitivityPnlInputError as error:
        print(error)
        print(
            f"\n  {len(error.defects)} defects, each with a source a front-end can "
            "map to a cell:"
        )
        for defect in error.defects:
            print(f"    {defect.source:<34} rows={list(defect.row_indices)}")


def main() -> int:
    one_shot_pnl = demo_one_shot()
    demo_cached(one_shot_pnl)
    demo_levels()
    demo_co_allocated()
    demo_rejected()
    print(f"\n{'=' * 72}\ndemo complete\n{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
