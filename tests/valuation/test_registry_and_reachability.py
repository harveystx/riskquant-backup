# tests/valuation/test_registry_and_reachability.py
"""
Production reachability, asserted as a FACT rather than assumed.

    authored kernel  !=  production-supported block

Six block types are authored and all six have registered pricers. Only three
can be placed in a `CompiledPortfolio`, and only SPOT has a registered
instrument compiler. These tests pin the CURRENT buildout state so that
promoting a block becomes a deliberate, visible change (D045/VAL-13).
"""

from __future__ import annotations

import dataclasses

from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.blocks.block_type import NUM_BLOCK_TYPES, BlockType
from riskquant.valuation.pricer_registry import BLOCK_PRICERS

PORTFOLIO_SLOTS = {"spot", "vanilla_forward", "vanilla_european_option"}

BLOCK_TYPE_TO_SLOT = {
    BlockType.SPOT: "spot",
    BlockType.VANILLA_FORWARD: "vanilla_forward",
    BlockType.VANILLA_EUROPEAN_OPTION: "vanilla_european_option",
}


def test_every_authored_block_type_has_a_pricer():
    assert len(BLOCK_PRICERS) == NUM_BLOCK_TYPES
    assert all(p is not None for p in BLOCK_PRICERS)


def test_compiled_portfolio_exposes_exactly_three_block_slots():
    slots = {f.name for f in dataclasses.fields(CompiledPortfolio) if f.init} - {
        "unique_dates_ord32"
    }
    assert slots == PORTFOLIO_SLOTS


def test_cashflow_block_types_are_not_reachable_from_a_compiled_portfolio():
    """
    FIXED / FLOATING / PRINCIPAL_CASHFLOW have kernels, pricers and registry
    entries, and no portfolio slot. They are BUILDOUT STATE, not products.
    Delete this test only when the slots are added.
    """
    unreachable = {
        BlockType.FIXED_CASHFLOW,
        BlockType.FLOATING_CASHFLOW,
        BlockType.PRINCIPAL_CASHFLOW,
    }
    assert unreachable.isdisjoint(BLOCK_TYPE_TO_SLOT)
    assert {bt for bt in BlockType} - unreachable == set(BLOCK_TYPE_TO_SLOT)


def test_reachable_block_types_cover_the_portfolio_slots():
    assert set(BLOCK_TYPE_TO_SLOT.values()) == PORTFOLIO_SLOTS
