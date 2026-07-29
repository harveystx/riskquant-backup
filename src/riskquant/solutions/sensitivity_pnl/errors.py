# src/riskquant/solutions/sensitivity_pnl/errors.py
"""
The one exception this solution raises for bad input, and its report format.

A public edge is judged on its worst error message, not its best. The engine's
own messages are correct but written for the person who wrote the engine: they
name the array that failed a bound, not the spreadsheet column that produced
it. This module carries enough structure to point a caller at a cell.

Every defect found in one run is reported TOGETHER. Fail-on-first turns a
ten-column paste error into ten round trips, and each round trip costs a
recompile of the book.
"""

from __future__ import annotations

__all__ = [
    "InputDefect",
    "SensitivityPnlInputError",
]

from dataclasses import dataclass, field

#: Offending rows listed per defect. Enough to see a pattern, few enough that a
#: systematically broken column does not print a million line numbers.
MAX_REPORTED_ROWS: int = 8


@dataclass(frozen=True, slots=True, kw_only=True)
class InputDefect:
    """One thing wrong with one field of one input."""

    source: str
    """Where the caller can look: `sensitivities.position_id`, `market`, ..."""

    problem: str
    """What is wrong, with a count. Written for someone who has never read the engine."""

    remedy: str
    """What to do about it. Never 'check your input'."""

    row_indices: tuple[int, ...] = ()
    """0-based row numbers, truncated to `MAX_REPORTED_ROWS`."""

    total_rows: int = 0
    """True number of offending rows, which may exceed `len(row_indices)`."""

    def render(self) -> str:
        lines = [f"  {self.source}: {self.problem}"]
        if self.row_indices:
            shown = ", ".join(str(i) for i in self.row_indices)
            hidden = self.total_rows - len(self.row_indices)
            suffix = f" (+{hidden} more)" if hidden > 0 else ""
            lines.append(f"      rows (0-based): {shown}{suffix}")
        lines.append(f"      fix: {self.remedy}")
        return "\n".join(lines)


class SensitivityPnlInputError(ValueError):
    """
    Raised before any kernel runs, carrying every defect found in the request.

    Subclasses `ValueError` so callers already catching the engine's errors keep
    working; `defects` is there for a front-end that wants to paint cells rather
    than print a string.
    """

    def __init__(self, defects: list[InputDefect]) -> None:
        self.defects: tuple[InputDefect, ...] = tuple(defects)
        count = len(self.defects)
        plural = "" if count == 1 else "s"
        body = "\n".join(defect.render() for defect in self.defects)
        super().__init__(
            f"sensitivity-PnL request rejected — {count} defect{plural} found:\n{body}"
        )


@dataclass(slots=True)
class DefectLog:
    """Accumulator so one pass over the request reports every defect at once."""

    defects: list[InputDefect] = field(default_factory=list)

    def add(
        self,
        *,
        source: str,
        problem: str,
        remedy: str,
        row_indices: tuple[int, ...] = (),
        total_rows: int = 0,
    ) -> None:
        self.defects.append(
            InputDefect(
                source=source,
                problem=problem,
                remedy=remedy,
                row_indices=row_indices,
                total_rows=total_rows,
            )
        )

    def raise_if_any(self) -> None:
        if self.defects:
            raise SensitivityPnlInputError(self.defects)
