# _context_gen/tests/test_solution_domains.py
from __future__ import annotations

from pathlib import Path

from _context_gen.generate import build_profile, discover_solution_domains


def _make_tree(tmp_path: Path, layout: dict[str, str]) -> Path:
    for rel, content in layout.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path


def test_discovers_a_solutions_subdirectory_with_no_matching_core_package(tmp_path: Path) -> None:
    # Direct regression test for the real gap found on the live-proxy repo:
    # solutions/credit_exposure/ has no src/riskquant/credit_exposure/
    # counterpart at all, so it must still get its own domain.
    _make_tree(
        tmp_path,
        {
            "src/riskquant/solutions/credit_exposure/__init__.py": "",
            "src/riskquant/solutions/credit_exposure/api.py": "def api(): ...\n",
        },
    )
    domains = discover_solution_domains(tmp_path)
    assert "solutions_credit_exposure" in domains


def test_solution_domain_name_does_not_collide_with_core_domain(tmp_path: Path) -> None:
    _make_tree(
        tmp_path,
        {
            "src/riskquant/sensitivity_pnl/engine.py": "x = 1\n",
            "src/riskquant/solutions/sensitivity_pnl/api.py": "def api(): ...\n",
        },
    )
    solution_domains = discover_solution_domains(tmp_path)
    assert solution_domains == ["solutions_sensitivity_pnl"]
    assert "sensitivity_pnl" not in solution_domains  # distinct from the core domain's own name


def test_solution_domain_profile_sweeps_the_whole_solutions_subtree_not_just_api() -> None:
    profile = build_profile("solutions_credit_exposure", Path("/unused"))
    assert profile.production_package_dirs == ("src/riskquant/solutions/credit_exposure",)
    assert profile.extra_production_files == ()


def test_core_domain_still_gets_the_light_api_attachment_when_names_match(tmp_path: Path) -> None:
    _make_tree(
        tmp_path,
        {
            "src/riskquant/sensitivity_pnl/engine.py": "x = 1\n",
            "src/riskquant/solutions/sensitivity_pnl/api.py": "def api(): ...\n",
        },
    )
    profile = build_profile("sensitivity_pnl", tmp_path)
    assert profile.production_package_dirs == ("src/riskquant/sensitivity_pnl",)
    assert profile.extra_production_files == ("src/riskquant/solutions/sensitivity_pnl/api.py",)


def test_no_solutions_directory_returns_empty(tmp_path: Path) -> None:
    _make_tree(tmp_path, {"src/riskquant/valuation/engine.py": "x = 1\n"})
    assert discover_solution_domains(tmp_path) == []
