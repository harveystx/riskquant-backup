# _context_gen/completeness.py
"""
Completeness status.

Hard rule (implementation contract §2.3 / §16): a profile must never be
silently incomplete. Every reason a profile could be short of source is
enumerated here and reported explicitly; generation fails on any of them
by default (see generate.py) rather than publishing a plausible-looking
partial artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _context_gen.closure import ClosureResult
from _context_gen.source_read import ReadOutcome


@dataclass(frozen=True, slots=True, kw_only=True)
class CompletenessReport:
    profile_name: str
    missing_configured_paths: tuple[str, ...] = field(default_factory=tuple)
    unreadable_files: tuple[str, ...] = field(default_factory=tuple)  # (path, reason)
    parse_failures: tuple[str, ...] = field(default_factory=tuple)  # (path, reason)
    missing_required_dependencies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_complete(self) -> bool:
        return not (
            self.missing_configured_paths
            or self.unreadable_files
            or self.parse_failures
            or self.missing_required_dependencies
        )

    def render(self) -> str:
        if self.is_complete:
            return "STATUS: COMPLETE"
        lines = ["STATUS: INCOMPLETE"]
        if self.missing_configured_paths:
            lines.append("MISSING CONFIGURED PATH:")
            lines.extend(f"    {p}" for p in self.missing_configured_paths)
        if self.unreadable_files:
            lines.append("UNREADABLE FILE:")
            lines.extend(f"    {p}" for p in self.unreadable_files)
        if self.parse_failures:
            lines.append("PARSE FAILURE:")
            lines.extend(f"    {p}" for p in self.parse_failures)
        if self.missing_required_dependencies:
            lines.append("MISSING REQUIRED INTERNAL DEPENDENCY:")
            lines.extend(f"    {p}" for p in self.missing_required_dependencies)
        return "\n".join(lines)


def build_report(
    *,
    profile_name: str,
    configured_dirs: list[str],
    existing_dirs: set[str],
    read_failures: list[tuple[str, str]],
    closure: ClosureResult,
) -> CompletenessReport:
    """
    `read_failures` is a list of (repo_relative_path, reason) already
    classified via source_read.ReadOutcome — UNREADABLE and PARSE_FAILED
    are both completeness concerns for production `.py` files; NOT_PYTHON
    is not (it's an expected, correctly-handled outcome, not a failure).
    """
    missing_paths = tuple(d for d in configured_dirs if d not in existing_dirs)
    unreadable = tuple(p for p, reason in read_failures if reason == ReadOutcome.UNREADABLE.value)
    parse_failed = tuple(p for p, reason in read_failures if reason == ReadOutcome.PARSE_FAILED.value)
    missing_deps = tuple(sorted(closure.unresolved_external))

    return CompletenessReport(
        profile_name=profile_name,
        missing_configured_paths=missing_paths,
        unreadable_files=unreadable,
        parse_failures=parse_failed,
        missing_required_dependencies=missing_deps,
    )
