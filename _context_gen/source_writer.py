# _context_gen/source_writer.py
"""
Renders one domain/test/demo/core/all source artifact.

Source-heavy, metadata-light (implementation contract §21): one status
block, one line per file boundary, then exact source. No per-file docstring
restatement, no per-file class/function list, no per-file directory tree,
no repeated guidance — all of that lives in the index, once (index_writer.py).

Raw source is never modified beyond the one explicit exception permitted by
the architecture contract §14: normalising line endings to `\\n`. Nothing
else about the source bytes is touched.

Fence length is dynamic (CommonMark's own rule: a closing fence must be
longer than any run of backticks inside the content it closes). Fixed-length
triple-backtick fencing is unsafe here: real RiskQuant modules embed
Markdown-formatted formula blocks in their own docstrings (confirmed via a
full 534-file live-repository fidelity sweep, which is exactly how this was
caught — a naive triple-backtick parse of the generated artifact silently
truncated one file at its first embedded ``` line). Using a fence longer
than the longest embedded run makes the boundary unambiguous for every
file, not just the common case.
"""

from __future__ import annotations

import re

from _context_gen.completeness import CompletenessReport
from _context_gen.fs import TEXT_SUFFIXES


def _fence_language(repo_relative: str) -> str:
    for suffix, lang in TEXT_SUFFIXES.items():
        if repo_relative.endswith(suffix):
            return lang
    return "text"


def _fence_for(content: str) -> str:
    longest = 0
    for run in re.findall(r"`+", content):
        longest = max(longest, len(run))
    return "`" * max(3, longest + 1)


def render_file_block(repo_relative: str, content: str) -> str:
    lang = _fence_language(repo_relative)
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    fence = _fence_for(normalized)
    return f"=== FILE: {repo_relative} ===\n{fence}{lang}\n{normalized}\n{fence}\n"


def render_source_artifact(
    *,
    profile_name: str,
    completeness: CompletenessReport,
    header_note: str | None,
    files: list[tuple[str, str]],  # (repo_relative, content), already sorted
) -> str:
    parts = [f"PROFILE {profile_name}", completeness.render()]
    if header_note:
        parts.append(header_note)
    body = "\n".join(render_file_block(p, c) for p, c in files)
    return "\n\n".join(parts) + "\n\n" + body
