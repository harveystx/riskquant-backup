# tests/solutions/sensitivity_pnl/unit/test_errors.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.errors`.

This module is the report format every defect in the package is eventually
rendered through, so its own contract deserves direct, isolated coverage:
`InputDefect.render()`'s exact formatting (including the "+N more" truncation
math), `SensitivityPnlInputError`'s message assembly and pluralisation, and
`DefectLog`'s accumulate-then-raise-together behaviour, which is the reason a
ten-column paste error costs one round trip instead of ten.
"""

from __future__ import annotations

import pytest

from riskquant.solutions.sensitivity_pnl.errors import (
    MAX_REPORTED_ROWS,
    DefectLog,
    InputDefect,
    SensitivityPnlInputError,
)


class TestInputDefectRender:
    def test_renders_source_and_problem_on_the_first_line(self) -> None:
        defect = InputDefect(
            source="sensitivities.position_id", problem="bad thing", remedy="fix it"
        )
        rendered = defect.render()
        assert rendered.startswith("  sensitivities.position_id: bad thing")

    def test_renders_the_remedy_on_its_own_line(self) -> None:
        defect = InputDefect(source="x", problem="y", remedy="do the thing")
        assert "      fix: do the thing" in defect.render()

    def test_no_row_indices_line_when_row_indices_is_empty(self) -> None:
        defect = InputDefect(source="x", problem="y", remedy="z", row_indices=())
        assert "rows (0-based)" not in defect.render()

    def test_row_indices_are_rendered_comma_separated(self) -> None:
        defect = InputDefect(
            source="x", problem="y", remedy="z", row_indices=(1, 3, 5), total_rows=3
        )
        assert "rows (0-based): 1, 3, 5" in defect.render()

    def test_no_plus_more_suffix_when_all_rows_are_shown(self) -> None:
        defect = InputDefect(
            source="x", problem="y", remedy="z", row_indices=(1, 2), total_rows=2
        )
        rendered = defect.render()
        assert "more" not in rendered

    def test_plus_more_suffix_when_total_rows_exceeds_shown_rows(self) -> None:
        defect = InputDefect(
            source="x", problem="y", remedy="z", row_indices=(1, 2), total_rows=10
        )
        assert "(+8 more)" in defect.render()

    def test_lines_appear_in_source_problem_rows_fix_order(self) -> None:
        defect = InputDefect(
            source="src", problem="prob", remedy="rem", row_indices=(0,), total_rows=1
        )
        lines = defect.render().split("\n")
        assert "src" in lines[0] and "prob" in lines[0]
        assert "rows (0-based)" in lines[1]
        assert "fix: rem" in lines[2]


class TestSensitivityPnlInputError:
    def test_message_reports_singular_defect_correctly(self) -> None:
        error = SensitivityPnlInputError(
            [InputDefect(source="a", problem="b", remedy="c")]
        )
        assert "1 defect found" in str(error)
        assert "1 defects" not in str(error)

    def test_message_reports_plural_defects_correctly(self) -> None:
        error = SensitivityPnlInputError(
            [
                InputDefect(source="a", problem="b", remedy="c"),
                InputDefect(source="d", problem="e", remedy="f"),
            ]
        )
        assert "2 defects found" in str(error)

    def test_message_includes_every_defects_rendered_body(self) -> None:
        error = SensitivityPnlInputError(
            [
                InputDefect(source="first", problem="p1", remedy="r1"),
                InputDefect(source="second", problem="p2", remedy="r2"),
            ]
        )
        text = str(error)
        assert "first" in text and "p1" in text
        assert "second" in text and "p2" in text

    def test_defects_property_preserves_order_as_a_tuple(self) -> None:
        d1 = InputDefect(source="a", problem="1", remedy="x")
        d2 = InputDefect(source="b", problem="2", remedy="y")
        error = SensitivityPnlInputError([d1, d2])
        assert error.defects == (d1, d2)
        assert isinstance(error.defects, tuple)

    def test_is_a_value_error_subclass(self) -> None:
        """Callers already catching the engine's plain ValueError must keep
        working without changing their except clause."""
        error = SensitivityPnlInputError(
            [InputDefect(source="a", problem="b", remedy="c")]
        )
        assert isinstance(error, ValueError)

    def test_can_be_caught_as_a_plain_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise SensitivityPnlInputError(
                [InputDefect(source="a", problem="b", remedy="c")]
            )

    def test_zero_defects_still_constructs_a_well_formed_message(self) -> None:
        """Not a state DefectLog ever reaches (raise_if_any checks truthiness
        first), but the class itself should not misbehave if constructed with
        an empty list directly."""
        error = SensitivityPnlInputError([])
        assert "0 defects found" in str(error)


class TestMaxReportedRowsConstant:
    def test_is_a_small_positive_integer(self) -> None:
        assert isinstance(MAX_REPORTED_ROWS, int)
        assert 0 < MAX_REPORTED_ROWS < 100


class TestDefectLog:
    def test_starts_empty(self) -> None:
        log = DefectLog()
        assert log.defects == []

    def test_add_appends_a_defect(self) -> None:
        log = DefectLog()
        log.add(source="a", problem="b", remedy="c")
        assert len(log.defects) == 1
        assert log.defects[0].source == "a"

    def test_add_defaults_row_indices_and_total_rows(self) -> None:
        log = DefectLog()
        log.add(source="a", problem="b", remedy="c")
        assert log.defects[0].row_indices == ()
        assert log.defects[0].total_rows == 0

    def test_multiple_adds_accumulate_in_order(self) -> None:
        log = DefectLog()
        log.add(source="first", problem="p", remedy="r")
        log.add(source="second", problem="p", remedy="r")
        assert [d.source for d in log.defects] == ["first", "second"]

    def test_raise_if_any_does_nothing_when_empty(self) -> None:
        log = DefectLog()
        log.raise_if_any()  # must not raise

    def test_raise_if_any_raises_with_every_accumulated_defect(self) -> None:
        log = DefectLog()
        log.add(source="a", problem="p1", remedy="r1")
        log.add(source="b", problem="p2", remedy="r2")
        with pytest.raises(SensitivityPnlInputError) as excinfo:
            log.raise_if_any()
        assert len(excinfo.value.defects) == 2

    def test_two_independent_logs_do_not_share_state(self) -> None:
        """The dataclass uses field(default_factory=list); a plain mutable
        default would leak defects across instances."""
        log_a = DefectLog()
        log_b = DefectLog()
        log_a.add(source="a", problem="p", remedy="r")
        assert log_b.defects == []
