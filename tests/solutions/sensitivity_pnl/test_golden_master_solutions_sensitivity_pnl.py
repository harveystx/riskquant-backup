# tests/solutions/sensitivity_pnl/test_golden_master_solutions_sensitivity_pnl.py
"""
Two-layer golden master for `riskquant.solutions.sensitivity_pnl`.

The core package has `tests/sensitivity_pnl/test_golden_master_sensitivity_pnl.py`.
The solution tier had no pytest coverage at all: the three `_verify/*.py` scripts
live inside `src/`, are hand-run, print rather than assert, and therefore never
fail a build. This file is the CI-visible replacement — the scripts remain useful
as narrated demos, but they are not the regression net.

Layer 1 — structural invariants. Properties that must hold for ANY book and ANY
          market, and which survive every change to the fixtures. This is where
          the facade's real contracts live: axis binding, row-set stability,
          cached-book equivalence, provenance honesty, aliasing, and the
          non-finite guards.

Layer 2 — a tiny test-OWNED frozen book with hand-computed PnL, exercising all
          three displacement conventions, two quote bases, and a cross term.
          Owned here, not shared with the core golden master, so the two cannot
          drift together and mask a facade defect.

The defect this file was written around
---------------------------------------
`test_log_convention_rejects_a_non_positive_shocked_level` is the regression for
a P0 found in audit. A LOG factor shocked through zero produced `ln` of a
non-positive number; `error_model="numpy"` emitted NaN silently, `PnlMatrix`
checks dtype and contiguity but not finiteness, the portfolio row inherited it on
summation, and `np.sort` then placed NaN LAST — so a lower-tail VaR excluded the
scenario that caused it. On a ten-scenario reproduction, poisoning the -100
catastrophe moved 99% VaR from -100 to -10. A loss did not merely mis-price; it
disappeared. Do not weaken these two tests.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    DuplicatePolicy,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    SensitivityPnlInputError,
    SensitivityPnlRequest,
    run_sensitivity_pnl,
    write_sensitivity_pnl,
)

# =========================================================================
# LAYER 2 FIXTURE — test-owned, frozen, hand-computable
# =========================================================================
#
#   key   convention  base   quote_base   (quote_base folds as base**n)
#   "E"   RELATIVE    50.0     100.0      cash delta per 1% move
#   "R"   ABSOLUTE     0.04   10000.0     DV01
#   "X"   LOG          2.0        1.0     log-space sensitivity
#
#   P1 (uid 11):  d1[E]=3.0  d2[E]=-2.0 ;  d1[R]=-5.0  d2[R]=0.0
#   P2 (uid 22):  cross(E, X) = 7.0
#   P3 (uid 33):  declared, no sensitivities -> exact zeros, holds its row
#
#   Scenario A levels = [55.0,  0.0410, 2*exp(0.5)]   -> x = [ 0.10,  0.0010,  0.5]
#   Scenario B levels = [45.0,  0.0385, 2*exp(-0.2)]  -> x = [-0.10, -0.0015, -0.2]
#
#   natural coefficients:  d1[E] = 3.0 * 100 = 300      d2[E] = -2.0 * 100^2 = -20000
#                          d1[R] = -5.0 * 1e4 = -50000
#                          cross(E,X) = 7.0 * 100 * 1 = 700
#
#   P1(A) = 300(0.10) + 0.5(-20000)(0.10^2) + (-50000)(0.0010)
#         =  30.0 - 100.0 - 50.0                                    = -120.0
#   P1(B) = 300(-0.10) + 0.5(-20000)(0.01) + (-50000)(-0.0015)
#         = -30.0 - 100.0 + 75.0                                    =  -55.0
#   P2(A) = 700(0.10)(0.5)                                          =   35.0
#   P2(B) = 700(-0.10)(-0.2)                                        =   14.0

FROZEN_KEYS = ["E", "R", "X"]
FROZEN_BASE_LEVELS = np.array([50.0, 0.04, 2.0], dtype=np.float64)
FROZEN_SHOCKED_LEVELS = np.ascontiguousarray(
    np.array(
        [
            [55.0, 45.0],
            [0.0410, 0.0385],
            [2.0 * math.exp(0.5), 2.0 * math.exp(-0.2)],
        ],
        dtype=np.float64,
    )
)
FROZEN_EXPECTED_PNL = np.array(
    [
        [-120.0, -55.0],
        [35.0, 14.0],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)
FROZEN_UIDS = [11, 22, 33]


def frozen_factor_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "factor_id": FROZEN_KEYS,
            "displacement_convention_id": [
                int(DisplacementConvention.RELATIVE),
                int(DisplacementConvention.ABSOLUTE),
                int(DisplacementConvention.LOG),
            ],
            "quote_base": [100.0, 10_000.0, 1.0],
        }
    )


def frozen_sensitivities() -> pl.DataFrame:
    return pl.DataFrame(
        [(11, "E", 3.0, -2.0), (11, "R", -5.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )


def frozen_cross() -> pl.DataFrame:
    return pl.DataFrame(
        [(22, "E", "X", 7.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.String,
            "factor_id_j": pl.String,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )


def frozen_axis() -> FactorAxis:
    return FactorAxis.from_keys(FROZEN_KEYS)


def frozen_spec(axis: FactorAxis | None = None) -> SensitivityBookSpec:
    return SensitivityBookSpec(
        sensitivities=frozen_sensitivities(),
        factor_meta=frozen_factor_meta(),
        factor_axis=axis if axis is not None else frozen_axis(),
        cross_sensitivities=frozen_cross(),
        position_uids=FROZEN_UIDS,
    )


def frozen_request(axis: FactorAxis) -> SensitivityPnlRequest:
    return SensitivityPnlRequest(
        sensitivities=frozen_sensitivities(),
        factor_meta=frozen_factor_meta(),
        market=axis.levels(
            base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
        ),
        cross_sensitivities=frozen_cross(),
        position_uids=FROZEN_UIDS,
    )


# =========================================================================
# LAYER 2 — frozen values
# =========================================================================


def test_frozen_book_matches_hand_computation_on_the_levels_path() -> None:
    axis = frozen_axis()
    result = run_sensitivity_pnl(frozen_request(axis))
    assert np.array_equal(result.position_uids, np.array(FROZEN_UIDS, dtype=np.int64))
    assert np.allclose(result.pnl, FROZEN_EXPECTED_PNL, rtol=1e-5, atol=1e-4)


def test_frozen_book_matches_hand_computation_on_the_displacement_path() -> None:
    """
    The two market contracts must agree on the same market state.

    Displacements are formed here in each factor's declared space, which is the
    same arithmetic `kernel_build_displacement` performs internally — so this
    also pins that the facade routes RELATIVE, ABSOLUTE and LOG to the arms the
    fixture claims.
    """
    axis = frozen_axis()
    displacement = np.ascontiguousarray(
        np.vstack(
            [
                FROZEN_SHOCKED_LEVELS[0] / FROZEN_BASE_LEVELS[0] - 1.0,
                FROZEN_SHOCKED_LEVELS[1] - FROZEN_BASE_LEVELS[1],
                np.log(FROZEN_SHOCKED_LEVELS[2] / FROZEN_BASE_LEVELS[2]),
            ]
        ).astype(np.float32)
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=frozen_sensitivities(),
            factor_meta=frozen_factor_meta(),
            market=axis.shocks(displacement),
            cross_sensitivities=frozen_cross(),
            position_uids=FROZEN_UIDS,
        )
    )
    assert np.allclose(result.pnl, FROZEN_EXPECTED_PNL, rtol=1e-5, atol=1e-4)


def test_quote_base_folds_as_base_to_the_order() -> None:
    """
    An order-n coefficient scales by `quote_base ** n`, not by `quote_base`.

    Pinned independently of the frozen totals because a linear-only fold would
    still reproduce a first-order-only book exactly, and the error would surface
    a quarter later on the first gamma-heavy portfolio.
    """
    axis = FactorAxis.from_keys(["G"])
    meta = pl.DataFrame(
        {
            "factor_id": ["G"],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)],
            "quote_base": [100.0],
        }
    )
    sens = pl.DataFrame(
        [(1, "G", 0.0, 4.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    displacement = np.ascontiguousarray(np.array([[0.03]], dtype=np.float32))
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens,
            factor_meta=meta,
            market=axis.shocks(displacement),
            position_uids=[1],
        )
    )
    # 0.5 * (4.0 * 100^2) * 0.03^2 = 0.5 * 40000 * 0.0009 = 18.0
    assert np.isclose(float(result.pnl[0, 0]), 18.0, rtol=1e-5)


def test_cross_term_is_the_full_mixed_partial_and_is_not_halved() -> None:
    """`c_ij * x_i * x_j`, listed once. A stray 1/2 halves every vanna book."""
    axis = FactorAxis.from_keys(["A", "B"])
    meta = pl.DataFrame(
        {
            "factor_id": ["A", "B"],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 2,
            "quote_base": [1.0, 1.0],
        }
    )
    empty = pl.DataFrame(
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        }
    )
    cross = pl.DataFrame(
        [(1, "A", "B", 10.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.String,
            "factor_id_j": pl.String,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    displacement = np.ascontiguousarray(np.array([[0.2], [0.3]], dtype=np.float32))
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=empty,
            factor_meta=meta,
            market=axis.shocks(displacement),
            cross_sensitivities=cross,
            position_uids=[1],
        )
    )
    assert np.isclose(float(result.pnl[0, 0]), 10.0 * 0.2 * 0.3, rtol=1e-5)


# =========================================================================
# LAYER 1 — the P0 regression
# =========================================================================


def test_log_convention_rejects_a_non_positive_shocked_level() -> None:
    """
    P0 REGRESSION. A LOG factor shocked through zero used to yield NaN PnL.

    NaN survives `PnlMatrix` (dtype and contiguity only), propagates to the whole
    portfolio row on summation, and `np.sort` places it LAST — so the scenario
    that produced it is excluded from a lower-tail VaR entirely. The loss does
    not mis-price; it vanishes.
    """
    axis = frozen_axis()
    poisoned = FROZEN_SHOCKED_LEVELS.copy()
    poisoned[2, 1] = -0.5  # factor "X" is the LOG one
    with pytest.raises(SensitivityPnlInputError, match="non-positive shocked level"):
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=frozen_sensitivities(),
                factor_meta=frozen_factor_meta(),
                market=axis.levels(
                    base_levels=FROZEN_BASE_LEVELS,
                    shocked_levels=np.ascontiguousarray(poisoned),
                ),
                cross_sensitivities=frozen_cross(),
                position_uids=FROZEN_UIDS,
            )
        )


def test_log_level_defect_names_the_callers_own_key() -> None:
    """
    Under SLOT encoding the compiled `factor_id` is a row position, which tells a
    desk user nothing. The facade must name the key they supplied.
    """
    axis = frozen_axis()
    poisoned = FROZEN_SHOCKED_LEVELS.copy()
    poisoned[2, 0] = 0.0
    with pytest.raises(SensitivityPnlInputError) as excinfo:
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=frozen_sensitivities(),
                factor_meta=frozen_factor_meta(),
                market=axis.levels(
                    base_levels=FROZEN_BASE_LEVELS,
                    shocked_levels=np.ascontiguousarray(poisoned),
                ),
                cross_sensitivities=frozen_cross(),
                position_uids=FROZEN_UIDS,
            )
        )
    assert "'X'" in str(excinfo.value)


def test_every_pnl_value_is_finite() -> None:
    """
    The blanket invariant behind the P0. No admissible input may produce a
    non-finite PnL, because nothing downstream checks for one.
    """
    axis = frozen_axis()
    result = run_sensitivity_pnl(frozen_request(axis))
    assert bool(np.isfinite(result.pnl).all())


def test_relative_convention_tolerates_a_negative_shocked_level() -> None:
    """
    The complement of the P0, pinned so the fix is not over-applied.

    `L/L0 - 1` is finite for any finite `L` once `L0 > 0` — a return below -100%
    is economically odd but arithmetically defined. Rejecting it would silently
    narrow the scenario space a caller is entitled to express.
    """
    axis = FactorAxis.from_keys(["S"])
    meta = pl.DataFrame(
        {
            "factor_id": ["S"],
            "displacement_convention_id": [int(DisplacementConvention.RELATIVE)],
            "quote_base": [1.0],
        }
    )
    sens = pl.DataFrame(
        [(1, "S", 10.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens,
            factor_meta=meta,
            market=FactorAxis.from_keys(["S"]).levels(
                base_levels=np.array([2.0]),
                shocked_levels=np.ascontiguousarray(np.array([[-1.0]])),
            ),
            position_uids=[1],
        )
    )
    assert np.isclose(float(result.pnl[0, 0]), 10.0 * (-1.0 / 2.0 - 1.0), rtol=1e-5)
    assert bool(np.isfinite(result.pnl).all())
    assert axis.num_factors == 1


# =========================================================================
# LAYER 1 — the axis is the seam
# =========================================================================


def test_a_reordered_axis_is_rejected_rather_than_silently_repriced() -> None:
    """
    The failure `market.py` exists to prevent. A permuted `(K, S)` matrix is
    indistinguishable from a correct one to the engine — it returns a plausible,
    wrong number — so the rejection has to happen where the keys are still known.
    """
    calculator = SensitivityPnlCalculator.compile(frozen_spec())
    reordered = FactorAxis.from_keys(["R", "E", "X"])
    with pytest.raises(SensitivityPnlInputError, match="row order"):
        calculator.run(
            reordered.levels(
                base_levels=FROZEN_BASE_LEVELS[[1, 0, 2]],
                shocked_levels=np.ascontiguousarray(FROZEN_SHOCKED_LEVELS[[1, 0, 2]]),
            )
        )


def test_axis_mismatch_message_names_the_first_differing_row() -> None:
    """ "A different axis" is a tautology; the row number is the fix."""
    calculator = SensitivityPnlCalculator.compile(frozen_spec())
    with pytest.raises(SensitivityPnlInputError) as excinfo:
        calculator.run(
            FactorAxis.from_keys(["E", "X", "R"]).shocks(
                np.zeros((3, 2), dtype=np.float32)
            )
        )
    message = str(excinfo.value)
    assert "row 1" in message
    assert "'X'" in message and "'R'" in message


def test_a_rebuilt_axis_with_the_same_keys_is_accepted() -> None:
    """Compared by value, not identity — rebuilding per tick is not an error."""
    calculator = SensitivityPnlCalculator.compile(frozen_spec())
    rebuilt = FactorAxis.from_keys(list(FROZEN_KEYS))
    result = calculator.run(
        rebuilt.levels(
            base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
        )
    )
    assert np.allclose(result.pnl, FROZEN_EXPECTED_PNL, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("num_factors", [2, 4])
def test_a_wrong_width_axis_is_rejected(num_factors: int) -> None:
    calculator = SensitivityPnlCalculator.compile(frozen_spec())
    keys = [f"Z{i}" for i in range(num_factors)]
    with pytest.raises(SensitivityPnlInputError, match="factors"):
        calculator.run(
            FactorAxis.from_keys(keys).shocks(
                np.zeros((num_factors, 2), dtype=np.float32)
            )
        )


# =========================================================================
# LAYER 1 — cached book equivalence
# =========================================================================


def test_cached_book_is_bit_identical_to_the_one_shot_path() -> None:
    """
    `run_sensitivity_pnl` is documented as a two-line composition of the cached
    path. Bit-identical, not merely close — a difference would mean two compile
    paths exist and only one is being reviewed.
    """
    axis = frozen_axis()
    one_shot = run_sensitivity_pnl(frozen_request(axis))
    calculator = SensitivityPnlCalculator.compile(frozen_spec(axis))
    cached = calculator.run(
        axis.levels(
            base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
        )
    )
    assert np.array_equal(one_shot.pnl, cached.pnl)
    assert np.array_equal(one_shot.position_uids, cached.position_uids)


def test_repeated_ticks_are_stable_and_do_not_alias() -> None:
    """
    Every `run` must return a fresh block. A calculator that reused one buffer
    would silently mutate results a caller was still holding — the aliasing bug
    `make_engine` is documented as paying 0.02% to avoid.
    """
    axis = frozen_axis()
    calculator = SensitivityPnlCalculator.compile(frozen_spec(axis))
    market = axis.levels(
        base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
    )
    first = calculator.run(market)
    held = first.pnl.copy()
    second = calculator.run(market)
    assert first.pnl is not second.pnl
    assert np.array_equal(first.pnl, held)
    assert np.array_equal(second.pnl, held)


def test_a_cached_book_validates_every_tick_as_thoroughly_as_the_first() -> None:
    """
    Validation is a construction invariant of the market type, not a step on the
    run path — so there is no argument to `run` that can skip it and no way for
    tick five thousand to be checked less carefully than tick one.
    """
    axis = frozen_axis()
    calculator = SensitivityPnlCalculator.compile(frozen_spec(axis))
    good = np.ascontiguousarray(np.zeros((3, 2), dtype=np.float32))
    for _ in range(3):
        calculator.run(axis.shocks(good))
    poisoned = good.copy()
    poisoned[1, 1] = np.nan
    with pytest.raises(SensitivityPnlInputError, match="non-finite"):
        axis.shocks(poisoned)


def test_batch_size_does_not_change_the_answer() -> None:
    """Batching is a memory decision; it must not be observable in the result."""
    rng = np.random.default_rng(5)
    keys = [f"F{i}" for i in range(6)]
    axis = FactorAxis.from_keys(keys)
    meta = pl.DataFrame(
        {
            "factor_id": keys,
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 6,
            "quote_base": [1.0] * 6,
        }
    )
    rows = [
        (m, keys[f], float(rng.normal(0, 1e3)), float(rng.normal(0, 1e4)))
        for m in range(1, 21)
        for f in range(6)
    ]
    sens = pl.DataFrame(
        rows,
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=list(range(1, 21)),
        )
    )
    displacement = np.ascontiguousarray(
        rng.normal(0.0, 0.02, size=(6, 733)).astype(np.float32)
    )
    reference = calculator.run(axis.shocks(displacement))
    # The engine slices the full matrix by offset, so a narrower scenario set is
    # the honest way to vary batching without reaching past the facade.
    for width in (1, 7, 733):
        partial = calculator.run(
            axis.shocks(np.ascontiguousarray(displacement[:, :width]))
        )
        assert np.array_equal(partial.pnl, reference.pnl[:, :width]), width


# =========================================================================
# LAYER 1 — row-set stability (Contract C)
# =========================================================================


def test_a_declared_position_with_no_sensitivities_holds_its_row() -> None:
    """
    The difference between a drop-in replacement and a near-miss. A fully hedged
    position carries no rows in the feed; without the declared row set it would
    vanish and a downstream join on `trade_uid` would drop it without comment.
    """
    axis = frozen_axis()
    result = run_sensitivity_pnl(frozen_request(axis))
    row = result.row_for(33)
    assert np.array_equal(result.position_uids, np.array([11, 22, 33], dtype=np.int64))
    assert np.all(row == 0.0)


def test_an_undeclared_position_in_the_feed_is_rejected() -> None:
    """
    A stale feed row, or one belonging to another book, must not silently widen
    the output. The row set is the contract.
    """
    axis = frozen_axis()
    extra = pl.concat(
        [
            frozen_sensitivities(),
            pl.DataFrame(
                [(99, "E", 1.0, 0.0)],
                schema=frozen_sensitivities().schema,
                orient="row",
            ),
        ]
    )
    with pytest.raises(SensitivityPnlInputError, match="absent from position_uids"):
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=extra,
                factor_meta=frozen_factor_meta(),
                market=axis.levels(
                    base_levels=FROZEN_BASE_LEVELS,
                    shocked_levels=FROZEN_SHOCKED_LEVELS,
                ),
                cross_sensitivities=frozen_cross(),
                position_uids=FROZEN_UIDS,
            )
        )


def test_row_for_rejects_an_unknown_uid() -> None:
    result = run_sensitivity_pnl(frozen_request(frozen_axis()))
    with pytest.raises(KeyError):
        result.row_for(12_345)


def test_result_rows_are_views_not_copies() -> None:
    """`row_for` is the only row accessor precisely because it does not copy."""
    result = run_sensitivity_pnl(frozen_request(frozen_axis()))
    assert result.row_for(11).base is not None


# =========================================================================
# LAYER 1 — grid handoff
# =========================================================================


def test_write_into_grid_matches_the_allocating_path() -> None:
    axis = frozen_axis()
    allocated = run_sensitivity_pnl(frozen_request(axis))

    grid = allocate_pnl_grid(
        block_position_uids=[
            np.array([900, 901], dtype=np.int64),
            np.array(FROZEN_UIDS, dtype=np.int64),
        ],
        num_scenarios=2,
    )
    grid.block_view(0)[:] = -1.0
    provenance = write_sensitivity_pnl(frozen_request(axis), grid=grid, block_index=1)
    assert np.array_equal(grid.block_view(1), allocated.pnl)
    assert np.all(grid.block_view(0) == -1.0)
    assert provenance.book_reused is False


def test_write_into_grid_rejects_a_row_set_that_does_not_match_the_block() -> None:
    """
    A book one row short would leave the `np.empty` garbage the grid was
    allocated with sitting in the output, indistinguishable from a PnL.
    """
    axis = frozen_axis()
    grid = allocate_pnl_grid(
        block_position_uids=[np.array([11, 22], dtype=np.int64)], num_scenarios=2
    )
    with pytest.raises(SensitivityPnlInputError, match="position_uids"):
        write_sensitivity_pnl(frozen_request(axis), grid=grid, block_index=0)


def test_write_into_grid_rejects_a_scenario_count_mismatch() -> None:
    axis = frozen_axis()
    grid = allocate_pnl_grid(
        block_position_uids=[np.array(FROZEN_UIDS, dtype=np.int64)], num_scenarios=5
    )
    with pytest.raises(SensitivityPnlInputError, match="scenario"):
        write_sensitivity_pnl(frozen_request(axis), grid=grid, block_index=0)


def test_write_into_grid_requires_a_declared_row_set() -> None:
    axis = frozen_axis()
    grid = allocate_pnl_grid(
        block_position_uids=[np.array([11, 22], dtype=np.int64)], num_scenarios=2
    )
    request = SensitivityPnlRequest(
        sensitivities=frozen_sensitivities(),
        factor_meta=frozen_factor_meta(),
        market=axis.levels(
            base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
        ),
        cross_sensitivities=frozen_cross(),
    )
    with pytest.raises(SensitivityPnlInputError, match="required when writing"):
        write_sensitivity_pnl(request, grid=grid, block_index=0)


# =========================================================================
# LAYER 1 — provenance is an audit record, so it must be true
# =========================================================================


def test_provenance_reports_the_facts_of_the_run() -> None:
    axis = frozen_axis()
    result = run_sensitivity_pnl(frozen_request(axis))
    provenance = result.provenance
    assert provenance.market_contract == "levels"
    assert provenance.factor_key_encoding == "slot"  # string keys
    assert provenance.book_reused is False
    assert provenance.num_positions == 3
    assert provenance.num_scenarios == 2
    assert provenance.num_market_factors == 3
    assert provenance.num_active_factors == 3
    assert provenance.num_first_order_entries == 2
    assert provenance.num_cross_entries == 1
    assert provenance.num_positions_without_sensitivities == 1  # uid 33
    assert provenance.batch_size >= 1


def test_provenance_book_reused_distinguishes_a_cached_run() -> None:
    """
    A ticking run showing `book_reused=False` means the cache is not being hit.
    The flag is only useful if it is honest, so it is pinned on both paths.
    """
    axis = frozen_axis()
    assert run_sensitivity_pnl(frozen_request(axis)).provenance.book_reused is False
    calculator = SensitivityPnlCalculator.compile(frozen_spec(axis))
    market = axis.shocks(np.zeros((3, 2), dtype=np.float32))
    assert calculator.run(market).provenance.book_reused is True


def test_provenance_encoding_reports_direct_for_a_sorted_integer_axis() -> None:
    axis = FactorAxis.from_keys([10, 20, 30])
    assert axis.slot_map is None
    meta = pl.DataFrame(
        {
            "factor_id": [10, 20, 30],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 3,
            "quote_base": [1.0] * 3,
        }
    )
    sens = pl.DataFrame(
        [(1, 10, 5.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int64,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens,
            factor_meta=meta,
            market=axis.shocks(np.zeros((3, 1), dtype=np.float32)),
            position_uids=[1],
        )
    )
    assert result.provenance.factor_key_encoding == "direct"


def test_active_factors_is_the_book_not_the_market_universe() -> None:
    """
    Sizes the kernel scratch. Reporting the market width instead would make a
    forty-factor book against a twenty-thousand-factor house axis look like a
    memory problem it is not.
    """
    keys = [f"F{i}" for i in range(50)]
    axis = FactorAxis.from_keys(keys)
    meta = pl.DataFrame(
        {
            "factor_id": keys,
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 50,
            "quote_base": [1.0] * 50,
        }
    )
    sens = pl.DataFrame(
        [(1, "F3", 1.0, 0.0), (1, "F7", 1.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=sens,
            factor_meta=meta,
            market=axis.shocks(np.zeros((50, 1), dtype=np.float32)),
            position_uids=[1],
        )
    )
    assert result.provenance.num_market_factors == 50
    assert result.provenance.num_active_factors == 2


# =========================================================================
# LAYER 1 — input defects are reported together, in the caller's terms
# =========================================================================


def test_the_house_wide_factor_table_may_be_passed_whole() -> None:
    """
    Documented behaviour under BOTH key encodings. Rows for factors outside the
    axis are scoped out before the compiler sees them; without that, SLOT
    encoding would raise on a metadata row the caller was told to ignore.
    """
    axis = frozen_axis()
    house = pl.concat(
        [
            frozen_factor_meta(),
            pl.DataFrame(
                {
                    "factor_id": ["ZZ1", "ZZ2"],
                    "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)]
                    * 2,
                    "quote_base": [1.0, 1.0],
                }
            ),
        ]
    )
    result = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=frozen_sensitivities(),
            factor_meta=house,
            market=axis.levels(
                base_levels=FROZEN_BASE_LEVELS, shocked_levels=FROZEN_SHOCKED_LEVELS
            ),
            cross_sensitivities=frozen_cross(),
            position_uids=FROZEN_UIDS,
        )
    )
    assert np.allclose(result.pnl, FROZEN_EXPECTED_PNL, rtol=1e-5, atol=1e-4)


def test_multiple_defects_are_reported_in_one_pass() -> None:
    """
    Fail-on-first turns a ten-column paste error into ten round trips, and each
    round trip recompiles the book.
    """
    axis = frozen_axis()
    broken = pl.DataFrame(
        [(1, "NOPE", float("nan"), 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    with pytest.raises(SensitivityPnlInputError) as excinfo:
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=broken,
                factor_meta=frozen_factor_meta(),
                market=axis.shocks(np.zeros((3, 1), dtype=np.float32)),
                position_uids=[1],
            )
        )
    assert len(excinfo.value.defects) >= 2


def test_a_sensitivity_to_a_factor_outside_the_axis_is_rejected() -> None:
    """
    The compiler calls this the most dangerous failure in the module: the factor
    would contribute exactly 0.0 to every scenario and pass every smoke test.
    """
    axis = FactorAxis.from_keys(["E", "R"])
    with pytest.raises(SensitivityPnlInputError):
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=frozen_sensitivities(),
                factor_meta=frozen_factor_meta(),
                market=axis.shocks(np.zeros((2, 1), dtype=np.float32)),
                cross_sensitivities=frozen_cross(),  # references "X"
                position_uids=FROZEN_UIDS,
            )
        )


def test_duplicate_policy_sum_adds_coefficients() -> None:
    axis = FactorAxis.from_keys(["E"])
    meta = pl.DataFrame(
        {
            "factor_id": ["E"],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)],
            "quote_base": [1.0],
        }
    )
    doubled = pl.DataFrame(
        [(1, "E", 2.0, 0.0), (1, "E", 3.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    market = axis.shocks(np.ascontiguousarray(np.array([[1.0]], dtype=np.float32)))
    with pytest.raises(ValueError, match="duplicate"):
        run_sensitivity_pnl(
            SensitivityPnlRequest(
                sensitivities=doubled,
                factor_meta=meta,
                market=market,
                position_uids=[1],
            )
        )
    summed = run_sensitivity_pnl(
        SensitivityPnlRequest(
            sensitivities=doubled,
            factor_meta=meta,
            market=market,
            position_uids=[1],
            duplicate_policy=DuplicatePolicy.SUM,
        )
    )
    assert np.isclose(float(summed.pnl[0, 0]), 5.0, rtol=1e-6)


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param([], id="empty"),
        pytest.param([1, 1, 2], id="duplicated"),
        pytest.param([1.5, 2.5], id="float_dtype"),
    ],
)
def test_a_malformed_factor_axis_is_rejected(keys: list[object]) -> None:
    """A float key would truncate and could silently merge two factors."""
    with pytest.raises(SensitivityPnlInputError):
        FactorAxis.from_keys(keys)


def test_a_market_matrix_of_the_wrong_height_is_rejected() -> None:
    axis = frozen_axis()
    with pytest.raises(SensitivityPnlInputError, match="rows against"):
        axis.shocks(np.zeros((2, 4), dtype=np.float32))


def test_a_market_with_no_scenarios_is_rejected() -> None:
    axis = frozen_axis()
    with pytest.raises(SensitivityPnlInputError, match="zero scenarios"):
        axis.shocks(np.zeros((3, 0), dtype=np.float32))


# =========================================================================
# LAYER 1 — concurrency
# =========================================================================


def test_two_threads_may_price_one_calculator_concurrently() -> None:
    """
    The calculator documents itself as frozen and free of per-run state. This
    pins the claim to the extent a test can: identical results from concurrent
    and serial pricing.

    Meaningful only where NUMBA_NUM_THREADS > 1 — with one thread, concurrent
    parallel regions never overlap and the test proves nothing, so it skips
    loudly rather than passing vacuously.
    """
    numba = pytest.importorskip("numba")
    if int(numba.config.NUMBA_NUM_THREADS) < 2:
        pytest.skip(
            "single-thread host; concurrent parallel regions cannot overlap and "
            "the calculator's thread-safety claim is UNVERIFIED here"
        )

    axis = frozen_axis()
    calculator = SensitivityPnlCalculator.compile(frozen_spec(axis))
    rng = np.random.default_rng(2)
    markets = [
        axis.shocks(
            np.ascontiguousarray(rng.normal(0, 0.02, (3, 400)).astype(np.float32))
        )
        for _ in range(4)
    ]
    serial = [calculator.run(market).pnl.copy() for market in markets]

    concurrent: list[np.ndarray | None] = [None] * 4
    failures: list[BaseException] = []

    def price(index: int) -> None:
        try:
            concurrent[index] = calculator.run(markets[index]).pnl.copy()
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=price, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, failures
    for index, block in enumerate(concurrent):
        assert block is not None
        assert np.array_equal(block, serial[index]), index
