# _context_gen/tests/test_source_writer.py
from __future__ import annotations

import re

from _context_gen.source_writer import render_file_block


def test_plain_content_uses_triple_backtick_fence() -> None:
    block = render_file_block("src/riskquant/x.py", "def f():\n    return 1\n")
    assert block.startswith("=== FILE: src/riskquant/x.py ===\n```python\n")
    assert block.rstrip("\n").endswith("```")


def test_content_with_embedded_triple_backtick_gets_a_longer_fence() -> None:
    # Direct regression test for the real bug found in the release-gate
    # pass: a docstring containing its own ``` fenced formula block (a
    # real, confirmed pattern in RiskQuant source -- e.g. `credit/exposure/
    # evaluators/kernels/current_exposure_method.py`) silently truncated
    # the emitted file under fixed-length ``` fencing, because a naive
    # parser reads the FIRST ``` after the opening fence as the close.
    content = 'x = 1\n"""\n```\nNRC = max(sum(MTM) - C, 0)\n```\n"""\n'
    block = render_file_block("src/riskquant/credit/x.py", content)

    # The opening/closing fence must be longer than any backtick run in
    # the content, so a naive scan for the shorter embedded ``` cannot be
    # mistaken for the real close.
    fence_match = re.match(r"=== FILE: \S+ ===\n(`+)python\n", block)
    assert fence_match is not None
    fence = fence_match.group(1)
    assert len(fence) >= 4  # longer than the embedded 3-backtick run
    assert block.count(fence) == 2  # exactly open + close, nothing else matches this length

    # And the full original content, including its own ``` lines, must be
    # recoverable intact between the two real fences.
    body = block.split(f"{fence}python\n", 1)[1].rsplit(f"\n{fence}\n", 1)[0]
    assert body == content


def test_content_with_longer_embedded_run_gets_even_longer_fence() -> None:
    content = "not python, but hypothetically: ````\nstuff\n````\n"
    block = render_file_block("src/riskquant/x.md", content)
    fence_match = re.match(r"=== FILE: \S+ ===\n(`+)markdown\n", block)
    assert fence_match is not None
    assert len(fence_match.group(1)) >= 5  # longer than the embedded 4-backtick run
