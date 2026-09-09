# tests/solutions/sensitivity_pnl/unit/test_pipeline.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.pipeline`, the shared
stages every entry point (one-shot, calculator, grid handoff) composes.

`resolve_batch_size` gets the most dedicated attention here: it derives the
one number the caller never gets to choose, from a fixed scratch budget, and
its arithmetic (never zero, capped at MAX_BATCH_SCENARIOS, shrinking as active
factors grow) is exactly the kind of boundary math worth pinning directly
rather than only observing through a full run's provenance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.sensitivity_pnl.engine import SensitivityPnlEngine
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.sensitivity_pnl.validity import TaylorValidityEngine
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.pipeline import (
    MAX_BATCH_SCENARIOS,
    SCRATCH_BUDGET_BYTES,
    build_provenance,
    compile_book,
    make_engine,
    make_validity_engine,
    resolve_batch_size,
    scope_factor_meta,
    stream_scenarios,
)
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)


def _spec(*, factor_meta: pl.DataFrame, axis_keys) -> SensitivityBookSpec:
    axis = FactorAxis.from_keys(axis_keys)
    return SensitivityBookSpec(
        sensitivities=pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [axis_keys[0]],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        ),
        factor_meta=factor_meta,
        factor_axis=axis,
        position_uids=[1],
    )


class TestScopeFactorMeta:
    def test_narrows_to_factors_present_in_the_axis(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1, 2, 3],
                "displacement_convention_id": [ABSOLUTE] * 3,
                "quote_base": [1.0, 1.0, 1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=[1])
        scoped = scope_factor_meta(spec)
        assert scoped["factor_id"].to_list() == [1]

    def test_leaves_the_frame_untouched_when_the_key_column_is_missing(self) -> None:
        meta = pl.DataFrame({"not_factor_id": [1], "quote_base": [1.0]})
        spec = _spec(
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [1],
                    "displacement_convention_id": [ABSOLUTE],
                    "quote_base": [1.0],
                }
            ),
            axis_keys=[1],
        )
        # swap in the malformed frame directly, bypassing the constructor default
        from dataclasses import replace

        spec = replace(spec, factor_meta=meta)
        scoped = scope_factor_meta(spec)
        assert scoped is meta

    def test_leaves_the_frame_untouched_when_the_key_dtype_mismatches_the_axis(
        self,
    ) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": ["not_an_int"],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        from dataclasses import replace

        spec = _spec(
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [1],
                    "displacement_convention_id": [ABSOLUTE],
                    "quote_base": [1.0],
                }
            ),
            axis_keys=[1],
        )
        spec = replace(spec, factor_meta=meta)
        scoped = scope_factor_meta(spec)
        assert scoped is meta

    def test_scoping_under_slot_encoding_matches_string_keys(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": ["EQ_A", "EQ_B", "EQ_UNUSED"],
                "displacement_convention_id": [ABSOLUTE] * 3,
                "quote_base": [1.0, 1.0, 1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=["EQ_A", "EQ_B"])
        scoped = scope_factor_meta(spec)
        assert set(scoped["factor_id"].to_list()) == {"EQ_A", "EQ_B"}


class TestCompileBook:
    def test_compiles_a_direct_encoded_book(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=[1])
        book = compile_book(spec, factor_meta=meta)
        assert book.num_positions == 1
        assert book.active_factor_ids.tolist() == [1]

    def test_compiles_a_slot_encoded_book_with_relabelled_keys(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": ["EQ_A"],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=["EQ_A"])
        book = compile_book(spec, factor_meta=meta)
        assert book.active_factor_ids.tolist() == [0]  # relabelled to row position

    def test_extra_position_uids_come_from_the_spec(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=[1])
        from dataclasses import replace

        spec = replace(spec, position_uids=[1, 999])
        book = compile_book(spec, factor_meta=meta)
        assert book.position_uids.tolist() == [1, 999]

    def test_none_position_uids_means_the_row_set_comes_from_the_entries(
        self,
    ) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        spec = _spec(factor_meta=meta, axis_keys=[1])
        from dataclasses import replace

        spec = replace(spec, position_uids=None)
        book = compile_book(spec, factor_meta=meta)
        assert book.position_uids.tolist() == [1]


class TestResolveBatchSize:
    def test_never_zero_even_with_a_huge_active_factor_count(self) -> None:
        assert (
            resolve_batch_size(num_active_factors=10_000_000, num_scenarios=1000) >= 1
        )

    def test_zero_active_factors_uses_scenario_count_capped_by_max(self) -> None:
        assert resolve_batch_size(num_active_factors=0, num_scenarios=10) == 10
        assert (
            resolve_batch_size(num_active_factors=0, num_scenarios=1_000_000)
            == MAX_BATCH_SCENARIOS
        )

    def test_capped_at_max_batch_scenarios(self) -> None:
        result = resolve_batch_size(num_active_factors=1, num_scenarios=1_000_000)
        assert result == MAX_BATCH_SCENARIOS

    def test_capped_at_num_scenarios_when_that_is_smaller(self) -> None:
        result = resolve_batch_size(num_active_factors=1, num_scenarios=5)
        assert result == 5

    def test_shrinks_as_active_factors_grow(self) -> None:
        small = resolve_batch_size(num_active_factors=10, num_scenarios=1_000_000)
        large = resolve_batch_size(num_active_factors=10_000, num_scenarios=1_000_000)
        assert large <= small

    def test_matches_the_documented_scratch_budget_formula(self) -> None:
        num_active_factors = 100
        expected = SCRATCH_BUDGET_BYTES // (num_active_factors * 4)
        result = resolve_batch_size(
            num_active_factors=num_active_factors, num_scenarios=10_000_000
        )
        assert result == min(expected, MAX_BATCH_SCENARIOS)


class TestMakeEngine:
    def _book(self):
        ids = np.array([0], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [0],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [0],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        return compile_sensitivity_book(
            sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
        )

    def test_returns_a_sensitivity_pnl_engine(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0])
        market = axis.shocks(np.zeros((1, 3), dtype=np.float32))
        grid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=3
        )
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        assert isinstance(engine, SensitivityPnlEngine)

    def test_displacement_market_gives_the_engine_no_base_levels(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0])
        market = axis.shocks(np.zeros((1, 3), dtype=np.float32))
        grid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=3
        )
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        assert engine.base_levels.size == 0

    def test_levels_market_gives_the_engine_the_base_levels(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([1.0]), shocked_levels=np.zeros((1, 3))
        )
        grid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=3
        )
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        assert engine.base_levels.tolist() == [1.0]

    def test_a_fresh_engine_is_built_every_call_never_reused(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0])
        market = axis.shocks(np.zeros((1, 3), dtype=np.float32))
        grid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=3
        )
        engine1 = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        engine2 = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        assert engine1 is not engine2


class TestMakeValidityEngine:
    def test_returns_a_taylor_validity_engine(self) -> None:
        ids = np.array([0], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [0],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [0],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        book = compile_sensitivity_book(
            sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
        )
        axis = FactorAxis.from_keys([0])
        market = axis.shocks(np.zeros((1, 3), dtype=np.float32))
        linear_out = np.empty((1, 3), dtype=np.float32)
        quadratic_out = np.empty((1, 3), dtype=np.float32)
        engine = make_validity_engine(
            book=book,
            market=market,
            linear_out=linear_out,
            quadratic_out=quadratic_out,
            batch_size=3,
        )
        assert isinstance(engine, TaylorValidityEngine)


class TestStreamScenarios:
    def _book_and_grid(self, num_scenarios: int):
        ids = np.array([0], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [0],
                "first_order": [1.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [0],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        book = compile_sensitivity_book(
            sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
        )
        grid = allocate_pnl_grid(
            block_position_uids=[book.position_uids], num_scenarios=num_scenarios
        )
        return book, grid

    def test_shocks_market_dispatches_to_absorb_displacement(self) -> None:
        book, grid = self._book_and_grid(5)
        axis = FactorAxis.from_keys([0])
        displacement = np.arange(5, dtype=np.float32).reshape(1, 5)
        market = axis.shocks(displacement)
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=5
        )
        stream_scenarios(market=market, engine=engine, num_scenarios=5)
        assert grid.pnl[0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_levels_market_dispatches_to_absorb_levels(self) -> None:
        book, grid = self._book_and_grid(3)
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([10.0]),
            shocked_levels=np.array([[10.0, 11.0, 12.0]]),
        )
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        stream_scenarios(market=market, engine=engine, num_scenarios=3)
        assert grid.pnl[0].tolist() == pytest.approx([0.0, 1.0, 2.0])

    def test_multiple_batches_are_walked_when_batch_size_is_smaller_than_total(
        self,
    ) -> None:
        book, grid = self._book_and_grid(10)
        axis = FactorAxis.from_keys([0])
        displacement = np.arange(10, dtype=np.float32).reshape(1, 10)
        market = axis.shocks(displacement)
        engine = make_engine(
            book=book, market=market, out=grid.block_view(0), batch_size=3
        )
        stream_scenarios(market=market, engine=engine, num_scenarios=10)
        assert grid.pnl[0].tolist() == pytest.approx(list(range(10)))


class TestBuildProvenance:
    def _book(self):
        ids = np.array([0, 1], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": [1, 2],
                "factor_id": [0, 1],
                "first_order": [5.0, 6.0],
                "second_order": [0.0, 0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [0, 1],
                "displacement_convention_id": [ABSOLUTE, ABSOLUTE],
                "quote_base": [1.0, 1.0],
            }
        )
        return compile_sensitivity_book(
            sensitivities=sens,
            factor_meta=meta,
            level_universe_factor_ids=ids,
            extra_position_uids=np.array([3], dtype=np.int64),
        )

    def test_market_contract_reflects_shocks_vs_levels(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0, 1])
        shocks_market = axis.shocks(np.zeros((2, 1), dtype=np.float32))
        levels_market = axis.levels(
            base_levels=np.array([1.0, 1.0]), shocked_levels=np.zeros((2, 1))
        )
        p1 = build_provenance(
            market=shocks_market, axis=axis, book=book, batch_size=1, book_reused=True
        )
        p2 = build_provenance(
            market=levels_market, axis=axis, book=book, batch_size=1, book_reused=True
        )
        assert p1.market_contract == "displacement"
        assert p2.market_contract == "levels"

    def test_factor_key_encoding_reflects_direct_vs_slot(self) -> None:
        book = self._book()
        direct_axis = FactorAxis.from_keys([0, 1])
        slot_axis = FactorAxis.from_keys(["A", "B"])
        market1 = direct_axis.shocks(np.zeros((2, 1), dtype=np.float32))
        market2 = slot_axis.shocks(np.zeros((2, 1), dtype=np.float32))
        p1 = build_provenance(
            market=market1, axis=direct_axis, book=book, batch_size=1, book_reused=True
        )
        p2 = build_provenance(
            market=market2, axis=slot_axis, book=book, batch_size=1, book_reused=True
        )
        assert p1.factor_key_encoding == "direct"
        assert p2.factor_key_encoding == "slot"

    def test_num_positions_without_sensitivities_counts_declared_empty_rows(
        self,
    ) -> None:
        book = self._book()  # position 3 declared but empty
        axis = FactorAxis.from_keys([0, 1])
        market = axis.shocks(np.zeros((2, 1), dtype=np.float32))
        provenance = build_provenance(
            market=market, axis=axis, book=book, batch_size=1, book_reused=True
        )
        assert provenance.num_positions_without_sensitivities == 1

    def test_book_reused_flag_is_passed_through_verbatim(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0, 1])
        market = axis.shocks(np.zeros((2, 1), dtype=np.float32))
        p_false = build_provenance(
            market=market, axis=axis, book=book, batch_size=1, book_reused=False
        )
        p_true = build_provenance(
            market=market, axis=axis, book=book, batch_size=1, book_reused=True
        )
        assert p_false.book_reused is False
        assert p_true.book_reused is True

    def test_counts_match_the_compiled_book_exactly(self) -> None:
        book = self._book()
        axis = FactorAxis.from_keys([0, 1])
        market = axis.shocks(np.zeros((2, 1), dtype=np.float32))
        provenance = build_provenance(
            market=market, axis=axis, book=book, batch_size=1, book_reused=True
        )
        assert provenance.num_positions == book.num_positions
        assert provenance.num_active_factors == book.num_active_factors
        assert provenance.num_first_order_entries == book.nnz
        assert provenance.num_cross_entries == book.nnz_cross
