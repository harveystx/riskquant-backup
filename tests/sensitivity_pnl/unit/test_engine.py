# tests/sensitivity_pnl/unit/test_engine.py
"""
Focused unit tests for the small, independently-testable pieces of
`riskquant.sensitivity_pnl.engine`: the accumulator padding arithmetic, the
thread-ceiling wrapper, and the `StreamingSensitivityEngine` structural
protocol. `SensitivityPnlEngine`'s absorb/validate behaviour end to end is
already covered extensively by `test_golden_master_sensitivity_pnl.py`; this
file does not repeat that, only the pieces beneath it.
"""

from __future__ import annotations

import numba as nb
import pytest

from riskquant.sensitivity_pnl.engine import (
    SensitivityPnlEngine,
    StreamingSensitivityEngine,
    accumulator_row_width,
    numba_thread_ceiling,
)


class TestAccumulatorRowWidth:
    def test_batch_size_of_zero_still_pads_by_one_full_pad_block(self) -> None:
        """0 rounds up to the nearest multiple of 8, then one more pad block
        is added -- the function must not divide by zero or return 0."""
        assert accumulator_row_width(0) == 8

    def test_exact_multiple_of_pad_size_still_gets_a_pad_block(self) -> None:
        """batch_size=8 is already 8-aligned; the padding is UNCONDITIONALLY
        added on top (a false-sharing guard, not a rounding correction), so
        the result must be 16, not 8."""
        assert accumulator_row_width(8) == 16

    @pytest.mark.parametrize(
        "batch_size,expected",
        [
            (1, 16),  # rounds up to 8, + 8 pad
            (7, 16),
            (9, 24),  # rounds up to 16, + 8 pad
            (500, 512),  # rounds up to 504? -> check exact below
        ],
    )
    def test_rounds_up_to_the_next_multiple_of_eight_then_pads(
        self, batch_size: int, expected: int
    ) -> None:
        # 500 -> ceil(500/8)*8 = 504, + 8 = 512
        assert accumulator_row_width(batch_size) == expected

    def test_result_is_always_a_multiple_of_eight(self) -> None:
        for batch_size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 4096, 1_000_000):
            assert accumulator_row_width(batch_size) % 8 == 0

    def test_result_always_strictly_exceeds_batch_size(self) -> None:
        """The padded width must be able to hold `batch_size` elements AND
        leave the false-sharing gap -- so it must be strictly greater, never
        merely equal, for every batch_size including already-aligned ones."""
        for batch_size in (1, 8, 16, 500, 4096):
            assert accumulator_row_width(batch_size) > batch_size

    def test_is_monotonically_non_decreasing(self) -> None:
        widths = [accumulator_row_width(b) for b in range(1, 200)]
        assert all(widths[i] <= widths[i + 1] for i in range(len(widths) - 1))


class TestNumbaThreadCeiling:
    def test_matches_numba_config_directly(self) -> None:
        assert numba_thread_ceiling() == int(nb.config.NUMBA_NUM_THREADS)

    def test_returns_a_plain_python_int(self) -> None:
        assert isinstance(numba_thread_ceiling(), int)

    def test_is_at_least_one(self) -> None:
        assert numba_thread_ceiling() >= 1


class TestStreamingSensitivityEngineProtocol:
    def test_sensitivity_pnl_engine_satisfies_the_protocol_structurally(
        self,
    ) -> None:
        """isinstance against a runtime_checkable Protocol checks the METHOD
        NAMES are present, not their exact signatures -- this pins that the
        engine has not accidentally renamed or removed absorb_levels /
        absorb_displacement, which would silently break `pipeline.
        stream_scenarios`'s duck-typed dispatch without any import error."""
        assert isinstance(SensitivityPnlEngine, type)
        assert hasattr(SensitivityPnlEngine, "absorb_levels")
        assert hasattr(SensitivityPnlEngine, "absorb_displacement")
        assert callable(SensitivityPnlEngine.absorb_levels)
        assert callable(SensitivityPnlEngine.absorb_displacement)

    def test_protocol_declares_batch_size_as_a_property_not_an_attribute(
        self,
    ) -> None:
        """Regression pin for the mypy fix: a plain `batch_size: int`
        annotation on the Protocol asks for a SETTABLE variable, which a
        frozen dataclass field can never satisfy. The Protocol must declare
        it via `@property` so both concrete engines (frozen dataclasses)
        structurally match."""
        assert isinstance(
            StreamingSensitivityEngine.__dict__.get("batch_size"), property
        )
