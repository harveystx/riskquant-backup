# tests/valuation/blocks/test_base_block_slots_regression.py
"""
Regression test for RQ-AUD-016.

``@dataclass(frozen=True, slots=True)`` must rebuild the class object to
attach ``__slots__`` (slots cannot be retrofitted onto an existing class).
This discards the class object that was live when any method body in that
class -- including ``__init_subclass__`` -- was compiled. The zero-arg
``super()`` sugar desugars to ``super(__class__, cls)``, where the implicit
``__class__`` cell still points at that discarded pre-rebuild class, so a
genuine subtype of the *rebuilt* class is not recognised as a subtype of it:

    TypeError: super(type, obj): obj must be an instance or subtype of type

This previously made every concrete ``Block`` subclass (``SpotBlock``,
``DiscountedCashflowBlock``, ``VanillaEuropeanOptionBlock``,
``VanillaForwardBlock``, ...) raise at class-definition time -- i.e. at
*import* time, not first use -- taking down the whole valuation-blocks
package with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


def test_dataclass_slots_subclassing_does_not_raise():
    """Minimal reproduction: a slots+frozen dataclass with __init_subclass__
    must not raise TypeError when subclassed."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class _Base:
        x: int = 0

        def __init_subclass__(cls) -> None:
            object.__init_subclass__()

    @dataclass(frozen=True, slots=True, kw_only=True)
    class _Child(_Base):
        y: int = 0

    assert _Child(x=1, y=2).x == 1


def test_block_base_class_imports():
    from riskquant.valuation.blocks._base_block import Block  # noqa: F401


@pytest.mark.parametrize(
    "module,cls_name",
    [
        ("riskquant.valuation.blocks.spot_block", "SpotBlock"),
        ("riskquant.valuation.blocks.discounted_cashflow_block", "PrincipalCashflowBlock"),
        ("riskquant.valuation.blocks.discounted_cashflow_block", "FixedCashflowBlock"),
        ("riskquant.valuation.blocks.discounted_cashflow_block", "FloatingCashflowBlock"),
        ("riskquant.valuation.blocks.vanilla_european_option_block", "VanillaEuropeanOptionBlock"),
        ("riskquant.valuation.blocks.vanilla_forward_block", "VanillaForwardBlock"),
    ],
)
def test_concrete_block_subclasses_import_and_define(module, cls_name):
    import importlib

    mod = importlib.import_module(module)
    assert hasattr(mod, cls_name), f"{module} does not define {cls_name}"


def test_missing_block_type_id_still_raises_typeerror():
    """The __init_subclass__ validation logic itself must still fire -- the
    fix must not silently swallow this legitimate error. Class name must NOT
    start with '_', or the leading-underscore guard skips validation by
    design (used for internal base classes such as _DiscountedCashflowBlock)."""
    from riskquant.valuation.blocks._base_block import Block

    with pytest.raises(TypeError, match="must define BLOCK_TYPE_ID"):

        @dataclass(frozen=True, slots=True, kw_only=True)
        class MissingIdBlock(Block):
            pass


def test_invalid_block_type_id_still_raises_valueerror():
    from riskquant.valuation.blocks._base_block import Block

    with pytest.raises(ValueError, match="invalid BLOCK_TYPE_ID"):

        @dataclass(frozen=True, slots=True, kw_only=True)
        class BadIdBlock(Block):
            BLOCK_TYPE_ID = 999_999


def test_spot_block_constructs_and_prices():
    from riskquant.valuation.blocks.spot_block import SpotBlock

    block = SpotBlock(
        trade_uids=np.array([1, 2], dtype=np.int64),
        qty=np.array([1.0, -2.0], dtype=np.float64),
        effective_spot_price_uix=np.array([0, 1], dtype=np.int32),
    )
    assert block.size == 2
    assert block.BLOCK_TYPE_ID != -1  # a real, valid dispatch id, not the sentinel
