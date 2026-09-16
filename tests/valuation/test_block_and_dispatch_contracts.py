# tests/valuation/test_block_and_dispatch_contracts.py
"""
Block dataclass contracts and BLOCK_TYPE_ID dispatch.

Covers the structural invariants that no mathematical test can reach:
keyword-only construction (VAL-08), defined equality/hash (VAL-09), dispatch
density, and the subclass registration guard.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    NUM_BLOCK_TYPES,
    UNSET_BLOCK_TYPE_ID_SENTINEL,
    VALID_BLOCK_TYPE_IDS,
    BlockType,
)
from riskquant.valuation.blocks.discounted_cashflow_block import (
    FixedCashflowBlock,
    FloatingCashflowBlock,
    PrincipalCashflowBlock,
)
from riskquant.valuation.blocks.spot_block import SpotBlock
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionBlock,
)
from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock
from riskquant.valuation.pricer_registry import BLOCK_PRICERS

ALL_BLOCKS = [
    SpotBlock,
    VanillaForwardBlock,
    VanillaEuropeanOptionBlock,
    FixedCashflowBlock,
    FloatingCashflowBlock,
    PrincipalCashflowBlock,
]


def _spot(n=2) -> SpotBlock:
    return SpotBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=np.ones(n, dtype=np.float64),
        effective_spot_price_uix=np.zeros(n, dtype=np.int32),
    )


# --- BLOCK_TYPE_ID ----------------------------------------------------------


def test_block_type_ids_are_dense_and_unique():
    ids = [int(b) for b in BlockType]
    assert ids == list(range(NUM_BLOCK_TYPES))
    assert len(set(ids)) == len(ids)


def test_sentinel_cannot_collide_with_a_real_id():
    assert UNSET_BLOCK_TYPE_ID_SENTINEL not in VALID_BLOCK_TYPE_IDS


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_every_block_declares_a_valid_id(cls):
    assert cls.BLOCK_TYPE_ID in VALID_BLOCK_TYPE_IDS


def test_block_type_ids_are_unique_across_classes():
    ids = [c.BLOCK_TYPE_ID for c in ALL_BLOCKS]
    assert len(set(ids)) == len(ids)


def test_subclass_without_an_id_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="must define BLOCK_TYPE_ID"):

        @dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
        class NoIdBlock(Block):
            pass


def test_subclass_with_an_invalid_id_is_rejected_at_class_creation():
    with pytest.raises(ValueError, match="invalid BLOCK_TYPE_ID"):

        @dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
        class BadIdBlock(Block):
            BLOCK_TYPE_ID = 9_999


def test_leading_underscore_names_bypass_the_id_guard_entirely():
    """
    `__init_subclass__` returns early for any class whose name starts with '_'.
    That exists so the abstract `_DiscountedCashflowBlock` can carry the
    sentinel, but it is a NAME-BASED escape hatch, not an abstractness test: a
    concrete block named with a leading underscore is registered with no id
    check at all. Pinned as current behaviour, not endorsed.
    """

    @dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
    class _SilentlyUnchecked(Block):
        BLOCK_TYPE_ID = 9_999  # invalid, and accepted

    assert _SilentlyUnchecked.BLOCK_TYPE_ID == 9_999


# --- dataclass contracts ----------------------------------------------------


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_every_block_is_keyword_only(cls):
    """VAL-08: runs of adjacent same-dtype arrays must not be positionally constructible."""
    params = dataclasses.fields(cls)
    assert all(f.kw_only for f in params), cls.__name__


def test_option_block_rejects_positional_construction():
    a = np.array([1.0])
    u = np.array([0], dtype=np.int32)
    with pytest.raises(TypeError):
        VanillaEuropeanOptionBlock(
            a,  # type: ignore
            a,
            a,
            np.array([1], dtype=np.int8),
            a,
            a,
            u,
            u,
            u,
            u,
            trade_uids=np.array([1], dtype=np.int64),
        )


def _instance(cls):
    """Minimal valid instance of any authored block."""
    n = 2
    u = np.arange(n, dtype=np.int64)
    f = np.ones(n, dtype=np.float64)
    i32 = np.zeros(n, dtype=np.int32)
    i8 = np.ones(n, dtype=np.int8)
    common = {"trade_uids": u}
    if cls is SpotBlock:
        return cls(**common, qty=f, effective_spot_price_uix=i32)
    if cls is VanillaForwardBlock:
        return cls(
            **common,
            qty=f,
            strike=f,
            effective_forward_price_uix=i32,
            effective_discount_factor_uix=i32,
        )
    if cls is VanillaEuropeanOptionBlock:
        return cls(
            **common,
            qty=f,
            strike=f,
            log_strike=f,
            cp_mult=i8,
            sqrt_tte=f,
            half_tte=f,
            forward_price_strike_ccy_uix=i32,
            option_log_forward_price_strike_ccy_uix=i32,
            vol_uix=i32,
            effective_discount_factor_uix=i32,
        )
    if cls is PrincipalCashflowBlock:
        return cls(**common, amount=f, effective_discount_factor_uix=i32)
    if cls is FixedCashflowBlock:
        return cls(
            **common,
            notional=f,
            accrual_factor=f,
            fixed_rate=f,
            pay_receive_mult=i8,
            effective_discount_factor_uix=i32,
        )
    if cls is FloatingCashflowBlock:
        return cls(
            **common,
            notional=f,
            accrual_factor=f,
            spread=f,
            pay_receive_mult=i8,
            forward_rate_uix=i32,
            effective_discount_factor_uix=i32,
        )
    raise AssertionError(cls)


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_equality_and_hash_are_defined(cls):
    """VAL-09: with the generated __eq__/__hash__ these raise on array fields."""
    b, other = _instance(cls), _instance(cls)
    assert (b == other) is False
    assert b == b
    assert isinstance(hash(b), int)
    assert b in [b]


def test_blocks_are_frozen():
    b = _spot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.qty = np.zeros(2)  # type: ignore[misc]


def test_size_is_the_row_count():
    assert _spot(5).size == 5
    assert _spot(0).size == 0


# --- dispatch ---------------------------------------------------------------


def test_registry_is_dense_and_complete():
    assert len(BLOCK_PRICERS) == NUM_BLOCK_TYPES
    assert all(p is not None for p in BLOCK_PRICERS)


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_each_block_indexes_a_distinct_pricer(cls):
    assert BLOCK_PRICERS[cls.BLOCK_TYPE_ID] is not None


def test_block_to_pricer_mapping_is_a_bijection():
    pricers = [BLOCK_PRICERS[c.BLOCK_TYPE_ID] for c in ALL_BLOCKS]
    assert len({id(p) for p in pricers}) == len(ALL_BLOCKS)
