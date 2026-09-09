# tests/sensitivity_pnl/unit/test_conventions.py
"""
Unit tests for `riskquant.sensitivity_pnl.conventions`.

Small module, but every constant here is load-bearing: `RATIO_DISPLACEMENT_IDS`
and `POSITIVE_LEVEL_DISPLACEMENT_IDS` gate two different ValueError paths
elsewhere in the package (a wrong membership here silently widens or narrows
which factors get which safety check), and the `QUOTE_BASE_*` constants are
the literal numbers a caller multiplies their sensitivities by. These tests
pin the raw values, not just "it doesn't crash."
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_LOG_ID,
    DISPLACEMENT_RELATIVE_ID,
    POSITIVE_LEVEL_DISPLACEMENT_IDS,
    QUOTE_BASE_NATURAL,
    QUOTE_BASE_PER_BASIS_POINT,
    QUOTE_BASE_PER_PERCENT,
    QUOTE_BASE_PER_VOL_POINT,
    RATIO_DISPLACEMENT_IDS,
    DisplacementConvention,
)


class TestDisplacementConventionEnum:
    def test_members_have_the_expected_integer_values(self) -> None:
        # Pinned, not just "distinct": these values are what actually gets
        # written into `displacement_convention_id` columns by every caller.
        assert int(DisplacementConvention.ABSOLUTE) == 1
        assert int(DisplacementConvention.RELATIVE) == 2
        assert int(DisplacementConvention.LOG) == 3

    def test_has_exactly_three_members(self) -> None:
        assert list(DisplacementConvention) == [
            DisplacementConvention.ABSOLUTE,
            DisplacementConvention.RELATIVE,
            DisplacementConvention.LOG,
        ]

    def test_member_values_are_distinct(self) -> None:
        values = [int(c) for c in DisplacementConvention]
        assert len(values) == len(set(values))


class TestIdConstants:
    def test_ids_match_the_enum_and_are_int8(self) -> None:
        for constant, member in (
            (DISPLACEMENT_ABSOLUTE_ID, DisplacementConvention.ABSOLUTE),
            (DISPLACEMENT_RELATIVE_ID, DisplacementConvention.RELATIVE),
            (DISPLACEMENT_LOG_ID, DisplacementConvention.LOG),
        ):
            assert isinstance(constant, np.int8)
            assert int(constant) == int(member)


class TestRatioDisplacementIds:
    def test_contains_exactly_relative_and_log(self) -> None:
        assert {int(x) for x in RATIO_DISPLACEMENT_IDS} == {
            int(DisplacementConvention.RELATIVE),
            int(DisplacementConvention.LOG),
        }

    def test_excludes_absolute(self) -> None:
        """
        ABSOLUTE displacement (`x = L - L0`) never divides by the base level,
        so it must never trigger the base-level-positivity guard. Including it
        here by mistake would reject perfectly valid absolute-convention books
        with a zero or negative base level (e.g. a rate curve at 0%).
        """
        assert int(DISPLACEMENT_ABSOLUTE_ID) not in [
            int(x) for x in RATIO_DISPLACEMENT_IDS
        ]

    def test_every_entry_is_int8(self) -> None:
        assert all(isinstance(x, np.int8) for x in RATIO_DISPLACEMENT_IDS)


class TestPositiveLevelDisplacementIds:
    def test_contains_only_log(self) -> None:
        assert [int(x) for x in POSITIVE_LEVEL_DISPLACEMENT_IDS] == [
            int(DisplacementConvention.LOG)
        ]

    def test_excludes_relative(self) -> None:
        """
        `L/L0 - 1` is finite for any finite, nonzero L, including negative L
        (a return below -100%). RELATIVE must NOT be in this set, or a caller
        expressing a large negative move would be rejected for no
        mathematical reason.
        """
        assert int(DISPLACEMENT_RELATIVE_ID) not in [
            int(x) for x in POSITIVE_LEVEL_DISPLACEMENT_IDS
        ]

    def test_excludes_absolute(self) -> None:
        assert int(DISPLACEMENT_ABSOLUTE_ID) not in [
            int(x) for x in POSITIVE_LEVEL_DISPLACEMENT_IDS
        ]

    def test_is_a_strict_subset_of_ratio_displacement_ids(self) -> None:
        """The module docstring states this relationship explicitly; pin it
        as a structural invariant rather than trusting the prose."""
        positive = {int(x) for x in POSITIVE_LEVEL_DISPLACEMENT_IDS}
        ratio = {int(x) for x in RATIO_DISPLACEMENT_IDS}
        assert positive < ratio  # strict subset: positive != ratio


class TestQuoteBaseConstants:
    @pytest.mark.parametrize(
        "constant,expected",
        [
            (QUOTE_BASE_NATURAL, 1.0),
            (QUOTE_BASE_PER_PERCENT, 100.0),
            (QUOTE_BASE_PER_VOL_POINT, 100.0),
            (QUOTE_BASE_PER_BASIS_POINT, 10_000.0),
        ],
    )
    def test_value(self, constant: float, expected: float) -> None:
        assert constant == expected

    def test_all_constants_are_strictly_positive(self) -> None:
        """
        `compile.py` rejects a non-positive `quote_base` at compile time, and
        every order-n coefficient is scaled by `quote_base ** n` — a zero or
        negative named constant would either divide-by-zero downstream logic
        that assumes positivity, or silently flip every sign for odd n.
        """
        for constant in (
            QUOTE_BASE_NATURAL,
            QUOTE_BASE_PER_PERCENT,
            QUOTE_BASE_PER_VOL_POINT,
            QUOTE_BASE_PER_BASIS_POINT,
        ):
            assert constant > 0.0

    def test_per_percent_and_per_vol_point_are_the_same_scale(self) -> None:
        """
        Not a coincidence worth losing silently: both represent "100 quoted
        units per natural unit" (a 1% move and a 1-vol-point move are both
        1/100 of a full unit move), so the two constants are expected to be
        numerically identical even though they mean different things to a
        caller.
        """
        assert QUOTE_BASE_PER_PERCENT == QUOTE_BASE_PER_VOL_POINT
