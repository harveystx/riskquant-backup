# _context_gen/tests/test_tests_artifact.py
"""
Contract tests for the per-domain test artifacts and the domain `--tests` flag.

Each test builds a real tree under `tmp_path` and runs the real snapshot,
so the tests cover the path from disk through `take_snapshot` to the
rendered artifact, not the partition logic alone (see
test_suite_partition.py for that).
"""

from __future__ import annotations

from pathlib import Path

from _context_gen.generate import (
    TESTS_SUBDIR,
    RenderedProfile,
    build_profile,
    render_domain_profile,
    render_tests_artifacts,
    take_snapshot,
)

DOMAINS = frozenset({"valuation", "simlab", "solutions_credit_var"})


def _tree(root: Path, layout: dict[str, str]) -> Path:
    for rel, content in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def _render(root: Path) -> dict[str, RenderedProfile]:
    arts = render_tests_artifacts(take_snapshot(root), root, domain_names=DOMAINS)
    return {a.name: a for a in arts}


def _file_headers(content: str, prefix: str) -> list[str]:
    return [ln for ln in content.splitlines() if ln.startswith(f"=== FILE: {prefix}")]


BASE = {
    "src/riskquant/__init__.py": "",
    "src/riskquant/valuation/__init__.py": "",
    "src/riskquant/valuation/engine.py": "x = 1\n",
}


def test_one_artifact_per_test_directory_with_every_file_verbatim(
    tmp_path: Path,
) -> None:
    body_a = 'def test_a():\n    """```\nformula\n```"""\n'
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/conftest.py": "import pytest\n",
            "tests/valuation/test_engine.py": body_a,
            "tests/simlab/test_runner.py": "def test_b():\n    pass\n",
            "tests/solutions/credit_var/test_api.py": "def test_c():\n    pass\n",
        },
    )
    arts = _render(tmp_path)
    assert sorted(arts) == [
        "tests_root",
        "tests_simlab",
        "tests_solutions_credit_var",
        "tests_valuation",
    ]
    for art in arts.values():
        assert art.completeness_report.is_complete
        assert art.filename == f"{TESTS_SUBDIR}/context_riskquant_{art.name}.txt"
        assert _file_headers(art.content, "src/") == []  # no production source

    valuation = arts["tests_valuation"].content
    assert "=== FILE: tests/valuation/test_engine.py ===" in valuation
    assert body_a in valuation  # embedded backticks survive via dynamic fence
    assert "=== FILE: tests/simlab/test_runner.py ===" not in valuation
    assert "=== FILE: tests/simlab/test_runner.py ===" in arts["tests_simlab"].content
    assert (
        "=== FILE: tests/solutions/credit_var/test_api.py ==="
        in arts["tests_solutions_credit_var"].content
    )


def test_header_names_owning_profile_and_embedded_support(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/conftest.py": "import pytest\n",
            "tests/_support/universe.py": "U = 1\n",
            "tests/valuation/test_engine.py": "from tests._support.universe import U\n",
            "tests/e2e/test_demo.py": "def test_d():\n    pass\n",
        },
    )
    arts = _render(tmp_path)

    valuation = arts["tests_valuation"].content
    assert "TEST DIRECTORY: tests/valuation/" in valuation
    assert "OWNING PROFILE: context_riskquant_valuation.txt" in valuation
    assert "tests/_support/universe.py, tests/conftest.py" in valuation
    assert _file_headers(valuation, "tests/") == [
        "=== FILE: tests/_support/universe.py ===",
        "=== FILE: tests/conftest.py ===",
        "=== FILE: tests/valuation/test_engine.py ===",
    ]

    e2e = arts["tests_e2e"].content
    assert "OWNING PROFILE: none" in e2e
    assert "=== FILE: tests/conftest.py ===" in e2e  # ancestor conftest, no import


def test_parse_failure_is_reported_and_text_still_embedded(tmp_path: Path) -> None:
    broken = "def test_x(:\n    pass\n"
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/valuation/test_broken.py": broken,
            "tests/simlab/test_ok.py": "X = 1\n",
        },
    )
    arts = _render(tmp_path)
    report = arts["tests_valuation"].completeness_report
    assert not report.is_complete
    assert report.parse_failures == ("tests/valuation/test_broken.py",)
    assert broken in arts["tests_valuation"].content
    assert arts["tests_simlab"].completeness_report.is_complete  # failure stays local


def test_parse_failure_in_support_file_marks_the_importing_bucket(
    tmp_path: Path,
) -> None:
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/_support/broken.py": "def f(:\n",
            "tests/valuation/test_engine.py": "from tests._support import broken\n",
        },
    )
    report = _render(tmp_path)["tests_valuation"].completeness_report
    assert report.parse_failures == ("tests/_support/broken.py",)


def test_unreadable_file_is_reported_and_its_bucket_still_exists(
    tmp_path: Path,
) -> None:
    _tree(tmp_path, BASE)
    bad = tmp_path / "tests/reporting/test_fx.py"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\xff\xfe\xfa not utf-8\n")
    arts = _render(tmp_path)
    report = arts["tests_reporting"].completeness_report
    assert report.unreadable_files == ("tests/reporting/test_fx.py",)
    assert _file_headers(arts["tests_reporting"].content, "tests/") == []


def test_unresolved_tests_import_is_a_missing_dependency(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {**BASE, "tests/valuation/test_engine.py": "from tests._support import gone\n"},
    )
    report = _render(tmp_path)["tests_valuation"].completeness_report
    assert not report.is_complete
    assert report.missing_required_dependencies == ("tests._support.gone",)


def test_missing_tests_directory_is_incomplete(tmp_path: Path) -> None:
    _tree(tmp_path, BASE)
    arts = render_tests_artifacts(
        take_snapshot(tmp_path), tmp_path, domain_names=DOMAINS
    )
    assert len(arts) == 1
    assert arts[0].completeness_report.missing_configured_paths == ("tests",)
    assert arts[0].filename == f"{TESTS_SUBDIR}/context_riskquant_tests.txt"


def test_render_is_deterministic(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/conftest.py": "",
            "tests/_support/a.py": "A = 1\n",
            "tests/valuation/test_engine.py": "from tests._support import a\n",
            "tests/simlab/test_runner.py": "from tests._support import a\n",
        },
    )
    first = [(a.filename, a.content) for a in _render(tmp_path).values()]
    second = [(a.filename, a.content) for a in _render(tmp_path).values()]
    assert first == second


def test_domain_tests_flag_uses_snapshot_text_not_a_second_disk_read(
    tmp_path: Path,
) -> None:
    _tree(tmp_path, {**BASE, "tests/valuation/test_engine.py": "ORIGINAL = 1\n"})
    snap = take_snapshot(tmp_path)
    # Mutate the file after the snapshot: output must still reflect the snapshot.
    (tmp_path / "tests/valuation/test_engine.py").write_text(
        "MUTATED = 1\n", encoding="utf-8"
    )
    art = render_domain_profile(
        build_profile("valuation", tmp_path), snap, tmp_path, tests=True, demos=False
    )
    assert "ORIGINAL = 1" in art.content
    assert "MUTATED = 1" not in art.content


def test_solutions_domain_tests_flag_reads_tests_solutions_subdir(
    tmp_path: Path,
) -> None:
    # Regression test: the flag swept tests/solutions_<name>/, which does
    # not exist, so a solutions profile embedded zero tests as COMPLETE.
    _tree(
        tmp_path,
        {
            **BASE,
            "src/riskquant/solutions/credit_var/api.py": "def run(): ...\n",
            "tests/solutions/credit_var/test_api.py": "API_TEST = 1\n",
        },
    )
    snap = take_snapshot(tmp_path)
    art = render_domain_profile(
        build_profile("solutions_credit_var", tmp_path),
        snap,
        tmp_path,
        tests=True,
        demos=False,
    )
    assert _file_headers(art.content, "tests/") == [
        "=== FILE: tests/solutions/credit_var/test_api.py ==="
    ]


def test_tests_flag_does_not_change_production_closure(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            **BASE,
            "tests/valuation/test_engine.py": "from riskquant.valuation import engine\n",
        },
    )
    snap = take_snapshot(tmp_path)
    prof = build_profile("valuation", tmp_path)
    plain = render_domain_profile(prof, snap, tmp_path, tests=False, demos=False)
    with_tests = render_domain_profile(prof, snap, tmp_path, tests=True, demos=False)
    assert _file_headers(plain.content, "src/") == _file_headers(
        with_tests.content, "src/"
    )
