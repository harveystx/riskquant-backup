# tests/sensitivity_pnl/unit/test_mathematical_properties.py
"""
Cross-cutting mathematical properties of the compiled book + engine pipeline
as a whole -- run across many randomly generated books rather than one fixed
fixture, so a property that happens to hold on a specific hand-picked example
cannot mask a defect that only shows up on a different book shape.

Each property below follows directly from the Taylor form itself:

    PnL_m(s) = SUM_j [ d1_mj * x_j(s) + 0.5 * d2_mj * x_j(s)^2 ]
             + SUM_{i<j} c_mij * x_i(s) * x_j(s)

and is checked THROUGH the real compiler and the real one-shot engine path,
not against a hand-derived formula re-implemented in the test -- the point is
to catch a defect in the pipeline, not to re-verify arithmetic already pinned
by `test_kernels.py`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import DuplicatePolicy, compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_RELATIVE_ID,
)
from riskquant.sensitivity_pnl.engine import compute_sensitivity_pnl_from_displacement


def _random_book(
    *,
    seed: int,
    num_positions: int = 12,
    num_factors: int = 10,
    entries_per_position: int = 4,
    include_cross: bool = True,
    coefficient_scale: float = 1.0,
):
    rng = np.random.default_rng(seed)
    ids = np.arange(num_factors, dtype=np.int32)
    diag_rows = []
    cross_rows = []
    for position_index in range(num_positions):
        uid = position_index + 1
        chosen = rng.choice(num_factors, entries_per_position, replace=False)
        for factor_id in chosen:
            diag_rows.append(
                (
                    uid,
                    int(factor_id),
                    float(rng.normal(0, 10)) * coefficient_scale,
                    float(rng.normal(0, 3)) * coefficient_scale,
                )
            )
        if include_cross and num_factors >= 2:
            pair = rng.choice(num_factors, 2, replace=False)
            cross_rows.append(
                (
                    uid,
                    int(pair[0]),
                    int(pair[1]),
                    float(rng.normal(0, 2)) * coefficient_scale,
                )
            )
    sensitivities = pl.DataFrame(
        diag_rows,
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int32,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross = pl.DataFrame(
        cross_rows,
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.Int32,
            "factor_id_j": pl.Int32,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    book = compile_sensitivity_book(
        sensitivities=sensitivities,
        factor_meta=meta,
        level_universe_factor_ids=ids,
        cross_sensitivities=cross if include_cross else None,
    )
    return book, num_factors


def _price(book, displacement: np.ndarray) -> np.ndarray:
    return compute_sensitivity_pnl_from_displacement(
        book=book, displacement=np.ascontiguousarray(displacement.astype(np.float32))
    ).astype(np.float64)


@pytest.mark.parametrize("seed", range(5))
def test_zero_displacement_gives_exactly_zero_pnl(seed: int) -> None:
    """PnL(x=0) = d1*0 + 0.5*d2*0 + c*0*0 = 0, for ANY coefficients -- the one
    property that must hold regardless of what a book's greeks are."""
    book, k = _random_book(seed=seed)
    displacement = np.zeros((k, 5), dtype=np.float32)
    pnl = _price(book, displacement)
    assert bool(np.all(pnl == 0.0))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("scale", [2.0, 0.5, -1.0, 10.0])
def test_pnl_is_linear_in_the_book_coefficients(seed: int, scale: float) -> None:
    """Scaling EVERY coefficient (first_order, second_order, cross_second_order)
    by k must scale the resulting PnL by exactly k, for any displacement --
    the Taylor form is linear in the coefficients even though it is NOT
    linear in the displacement."""
    rng = np.random.default_rng(seed + 1000)
    base_book, k = _random_book(seed=seed)
    scaled_book, _ = _random_book(seed=seed, coefficient_scale=scale)
    displacement = rng.normal(0, 0.3, (k, 6)).astype(np.float32)

    base_pnl = _price(base_book, displacement)
    scaled_pnl = _price(scaled_book, displacement)
    assert np.allclose(scaled_pnl, scale * base_pnl, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_negating_every_coefficient_negates_pnl(seed: int) -> None:
    """The scale=-1 case of the linearity property, checked directly as its
    own test: a short position's PnL is the exact negative of the
    corresponding long position's, for every scenario."""
    rng = np.random.default_rng(seed + 2000)
    long_book, k = _random_book(seed=seed)
    short_book, _ = _random_book(seed=seed, coefficient_scale=-1.0)
    displacement = rng.normal(0, 0.4, (k, 8)).astype(np.float32)

    long_pnl = _price(long_book, displacement)
    short_pnl = _price(short_book, displacement)
    assert np.allclose(short_pnl, -long_pnl, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_positions_do_not_interfere_with_each_other(seed: int) -> None:
    """Compiling two disjoint sub-portfolios together must give each position
    exactly the PnL it would have gotten compiled alone -- no cross-position
    leakage through shared scratch or shared factor slots."""
    rng = np.random.default_rng(seed + 3000)
    num_factors = 8
    ids = np.arange(num_factors, dtype=np.int32)
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )

    def make_sens(uid: int) -> pl.DataFrame:
        chosen = rng.choice(num_factors, 3, replace=False)
        return pl.DataFrame(
            {
                "position_id": [uid] * 3,
                "factor_id": chosen.tolist(),
                "first_order": rng.normal(0, 5, 3).tolist(),
                "second_order": rng.normal(0, 2, 3).tolist(),
            }
        )

    sens_a = make_sens(1)
    sens_b = make_sens(2)

    book_a = compile_sensitivity_book(
        sensitivities=sens_a, factor_meta=meta, level_universe_factor_ids=ids
    )
    book_b = compile_sensitivity_book(
        sensitivities=sens_b, factor_meta=meta, level_universe_factor_ids=ids
    )
    book_ab = compile_sensitivity_book(
        sensitivities=pl.concat([sens_a, sens_b]),
        factor_meta=meta,
        level_universe_factor_ids=ids,
    )

    displacement = rng.normal(0, 0.5, (num_factors, 4)).astype(np.float32)
    pnl_a = _price(book_a, displacement)
    pnl_b = _price(book_b, displacement)
    pnl_ab = _price(book_ab, displacement)

    row_a = int(np.searchsorted(book_ab.position_uids, 1))
    row_b = int(np.searchsorted(book_ab.position_uids, 2))
    assert np.allclose(pnl_ab[row_a], pnl_a[0], rtol=1e-5, atol=1e-6)
    assert np.allclose(pnl_ab[row_b], pnl_b[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_combining_two_positions_sensitivities_onto_one_uid_sums_their_pnl(
    seed: int,
) -> None:
    """If two 'legs' with independent sensitivity rows are declared under the
    SAME position_id, that position's PnL must equal the SUM of what each
    leg would contribute alone -- the additive structure of the Taylor sum
    over CSR entries, exercised through the real compiler rather than assumed."""
    rng = np.random.default_rng(seed + 4000)
    num_factors = 6
    ids = np.arange(num_factors, dtype=np.int32)
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    leg_1 = pl.DataFrame(
        {
            "position_id": [1, 1],
            "factor_id": [0, 1],
            "first_order": rng.normal(0, 5, 2).tolist(),
            "second_order": [0.0, 0.0],
        }
    )
    leg_2 = pl.DataFrame(
        {
            "position_id": [1, 1],
            "factor_id": [2, 3],
            "first_order": rng.normal(0, 5, 2).tolist(),
            "second_order": [0.0, 0.0],
        }
    )
    combined_book = compile_sensitivity_book(
        sensitivities=pl.concat([leg_1, leg_2]),
        factor_meta=meta,
        level_universe_factor_ids=ids,
    )
    leg_1_book = compile_sensitivity_book(
        sensitivities=leg_1.with_columns(pl.lit(1).alias("position_id")),
        factor_meta=meta,
        level_universe_factor_ids=ids,
    )
    leg_2_book = compile_sensitivity_book(
        sensitivities=leg_2.with_columns(pl.lit(1).alias("position_id")),
        factor_meta=meta,
        level_universe_factor_ids=ids,
    )
    displacement = rng.normal(0, 0.5, (num_factors, 5)).astype(np.float32)
    combined_pnl = _price(combined_book, displacement)[0]
    leg_1_pnl = _price(leg_1_book, displacement)[0]
    leg_2_pnl = _price(leg_2_book, displacement)[0]
    assert np.allclose(combined_pnl, leg_1_pnl + leg_2_pnl, rtol=1e-5, atol=1e-6)


class TestQuoteBaseUnitConsistency:
    def test_quote_base_folding_matches_a_manual_unit_change(self) -> None:
        """A sensitivity of 5.0 quoted 'per 100 units of displacement' against
        a raw displacement of 0.5 natural units must give the SAME PnL as a
        sensitivity of 500.0 quoted in natural units (quote_base=1) against
        the same 0.5 -- this is the entire point of quote_base folding,
        checked end to end rather than assumed from the compiler's own
        arithmetic."""
        ids = np.array([0], dtype=np.int32)

        quoted_book = compile_sensitivity_book(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id": [0],
                    "first_order": [5.0],
                    "second_order": [0.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [0],
                    "displacement_convention_id": [int(DISPLACEMENT_ABSOLUTE_ID)],
                    "quote_base": [100.0],
                }
            ),
            level_universe_factor_ids=ids,
        )
        natural_book = compile_sensitivity_book(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id": [0],
                    "first_order": [500.0],
                    "second_order": [0.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [0],
                    "displacement_convention_id": [int(DISPLACEMENT_ABSOLUTE_ID)],
                    "quote_base": [1.0],
                }
            ),
            level_universe_factor_ids=ids,
        )
        displacement = np.ascontiguousarray(np.array([[0.5]], dtype=np.float32))
        assert np.allclose(
            _price(quoted_book, displacement), _price(natural_book, displacement)
        )


class TestDuplicatePolicySumMatchesPreAggregation:
    def test_sum_policy_output_matches_a_caller_who_pre_summed_the_legs(self) -> None:
        """DuplicatePolicy.SUM must give bit-for-bit-close the same PnL as if
        the caller had aggregated the duplicate legs themselves before
        compiling -- otherwise the policy would be a convenience that changes
        the answer, not just the input shape."""
        ids = np.array([0], dtype=np.int32)
        meta = pl.DataFrame(
            {
                "factor_id": [0],
                "displacement_convention_id": [int(DISPLACEMENT_ABSOLUTE_ID)],
                "quote_base": [1.0],
            }
        )
        duplicated = pl.DataFrame(
            {
                "position_id": [1, 1, 1],
                "factor_id": [0, 0, 0],
                "first_order": [2.0, 3.0, 5.0],
                "second_order": [0.1, 0.2, 0.3],
            }
        )
        pre_summed = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [0],
                "first_order": [10.0],  # 2+3+5
                "second_order": [0.6],  # 0.1+0.2+0.3
            }
        )
        sum_book = compile_sensitivity_book(
            sensitivities=duplicated,
            factor_meta=meta,
            level_universe_factor_ids=ids,
            duplicate_policy=DuplicatePolicy.SUM,
        )
        reference_book = compile_sensitivity_book(
            sensitivities=pre_summed, factor_meta=meta, level_universe_factor_ids=ids
        )
        displacement = np.ascontiguousarray(
            np.array([[0.3, -0.7, 1.5]], dtype=np.float32)
        )
        assert np.allclose(
            _price(sum_book, displacement), _price(reference_book, displacement)
        )


class TestDisplacementConventionRoundTrip:
    def test_relative_convention_reproduces_a_direct_percent_change(self) -> None:
        """RELATIVE displacement x = L/L0 - 1: a 10% level increase must
        produce exactly the same PnL whether expressed as a levels-path
        10% move or as a pre-computed displacement of 0.10."""
        from riskquant.sensitivity_pnl.engine import compute_sensitivity_pnl_from_levels

        ids = np.array([0], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [1],
                    "factor_id": [0],
                    "first_order": [1000.0],
                    "second_order": [0.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [0],
                    "displacement_convention_id": [int(DISPLACEMENT_RELATIVE_ID)],
                    "quote_base": [1.0],
                }
            ),
            level_universe_factor_ids=ids,
        )
        base_levels = np.array([50.0])
        shocked_levels = np.ascontiguousarray(np.array([[55.0]]))  # +10%
        via_levels = compute_sensitivity_pnl_from_levels(
            book=book, base_levels=base_levels, shocked_levels=shocked_levels
        ).astype(np.float64)
        via_displacement = _price(book, np.array([[0.10]], dtype=np.float32))
        assert np.allclose(via_levels, via_displacement)
