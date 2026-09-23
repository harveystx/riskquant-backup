# _context_gen/tests/test_tests_artifact.py
from __future__ import annotations

from pathlib import Path

from _context_gen.generate import render_domain_profile, render_tests_artifact, build_profile, take_snapshot


def _tree(root: Path, layout: dict[str, str]) -> Path:
    for rel, content in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


BASE = {
    "src/riskquant/__init__.py": "",
    "src/riskquant/valuation/__init__.py": "",
    "src/riskquant/valuation/engine.py": "x = 1\n",
}


def test_tests_artifact_embeds_every_test_file_verbatim(tmp_path: Path) -> None:
    body_a = 'def test_a():\n    """```\nformula\n```"""\n'
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/conftest.py": "import pytest\n",
            "tests/valuation/test_engine.py": body_a,
            "tests/simlab/test_runner.py": "def test_b():\n    pass\n",
        },
    )
    art = render_tests_artifact(take_snapshot(tmp_path), tmp_path)
    assert art.completeness_report.is_complete
    assert art.filename == "context_riskquant_tests.txt"
    for path in ("tests/conftest.py", "tests/valuation/test_engine.py", "tests/simlab/test_runner.py"):
        assert f"=== FILE: {path} ===" in art.content
    assert body_a in art.content  # embedded backticks survive via dynamic fence
    assert "=== FILE: src/riskquant/valuation/engine.py ===" not in art.content  # no production source


def test_tests_artifact_reports_parse_failure_and_still_embeds_text(tmp_path: Path) -> None:
    broken = "def test_x(:\n    pass\n"
    _tree(tmp_path, {**BASE, "tests/valuation/test_broken.py": broken})
    art = render_tests_artifact(take_snapshot(tmp_path), tmp_path)
    assert not art.completeness_report.is_complete
    assert "tests/valuation/test_broken.py" in art.completeness_report.parse_failures
    assert broken in art.content


def test_missing_tests_directory_is_incomplete(tmp_path: Path) -> None:
    _tree(tmp_path, BASE)
    art = render_tests_artifact(take_snapshot(tmp_path), tmp_path)
    assert art.completeness_report.missing_configured_paths == ("tests",)


def test_domain_tests_flag_uses_snapshot_text_not_a_second_disk_read(tmp_path: Path) -> None:
    _tree(tmp_path, {**BASE, "tests/valuation/test_engine.py": "ORIGINAL = 1\n"})
    snap = take_snapshot(tmp_path)
    # Mutate the file after the snapshot: output must still reflect the snapshot.
    (tmp_path / "tests/valuation/test_engine.py").write_text("MUTATED = 1\n", encoding="utf-8")
    art = render_domain_profile(build_profile("valuation", tmp_path), snap, tmp_path, tests=True, demos=False)
    assert "ORIGINAL = 1" in art.content
    assert "MUTATED = 1" not in art.content


def test_tests_flag_does_not_change_production_closure(tmp_path: Path) -> None:
    _tree(tmp_path, {**BASE, "tests/valuation/test_engine.py": "from riskquant.valuation import engine\n"})
    snap = take_snapshot(tmp_path)
    prof = build_profile("valuation", tmp_path)
    plain = render_domain_profile(prof, snap, tmp_path, tests=False, demos=False)
    with_tests = render_domain_profile(prof, snap, tmp_path, tests=True, demos=False)
    strip = lambda c: [ln for ln in c.splitlines() if ln.startswith("=== FILE: src/")]
    assert strip(plain.content) == strip(with_tests.content)
