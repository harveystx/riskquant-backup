# _context_gen/tests/test_publish.py
from __future__ import annotations

from pathlib import Path

import pytest

from _context_gen.completeness import CompletenessReport
from _context_gen.generate import RenderedProfile, publish


def _rp(name: str, content: str, complete: bool = True) -> RenderedProfile:
    report = CompletenessReport(
        profile_name=name,
        missing_required_dependencies=() if complete else ("riskquant.missing",),
    )
    return RenderedProfile(
        name=name,
        filename=f"context_riskquant_{name}.txt",
        content=content,
        completeness_report=report,
        token_estimate=0,
    )


def test_publish_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    publish([_rp("valuation", "V1")], out)
    assert (out / "context_riskquant_valuation.txt").read_text() == "V1"


def test_partial_regeneration_preserves_other_previously_published_artifacts(
    tmp_path: Path,
) -> None:
    # Regression test for a real bug found via testing: the first version
    # of publish() replaced the *entire* output directory with only the
    # artifacts from the current run, silently deleting every other
    # previously-published profile on any partial `--profile X` run.
    out = tmp_path / "_contexts"
    publish([_rp("valuation", "V1"), _rp("simlab", "S1")], out)
    publish([_rp("valuation", "V2")], out)  # only valuation regenerated this time
    assert (out / "context_riskquant_valuation.txt").read_text() == "V2"
    assert (
        out / "context_riskquant_simlab.txt"
    ).read_text() == "S1"  # carried forward untouched


def test_incomplete_profile_refuses_publish_by_default(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        publish([_rp("valuation", "V1", complete=False)], out)
    assert not out.exists()


def test_failed_publish_leaves_previous_good_snapshot_untouched(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    publish([_rp("valuation", "GOOD")], out)
    with pytest.raises(RuntimeError):
        publish([_rp("valuation", "BAD"), _rp("simlab", "X", complete=False)], out)
    assert (out / "context_riskquant_valuation.txt").read_text() == "GOOD"


def test_allow_incomplete_still_publishes_with_status_intact(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    publish(
        [_rp("dependence_stress", "PARTIAL", complete=False)],
        out,
        allow_incomplete=True,
    )
    assert (out / "context_riskquant_dependence_stress.txt").read_text() == "PARTIAL"


def _rp_at(filename: str, content: str) -> RenderedProfile:
    report = CompletenessReport(profile_name=filename)
    return RenderedProfile(
        name=filename,
        filename=filename,
        content=content,
        completeness_report=report,
        token_estimate=0,
    )


def test_artifact_in_subdirectory_is_written(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    publish([_rp_at("tests/context_riskquant_tests_credit.txt", "C1")], out)
    assert (out / "tests" / "context_riskquant_tests_credit.txt").read_text() == "C1"


def test_replace_subdirs_removes_stale_bucket(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    publish(
        [
            _rp_at("tests/context_riskquant_tests_credit.txt", "C1"),
            _rp_at("tests/context_riskquant_tests_gone.txt", "G1"),
            _rp("valuation", "V1"),
        ],
        out,
        replace_subdirs=frozenset({"tests"}),
    )
    publish(
        [_rp_at("tests/context_riskquant_tests_credit.txt", "C2")],
        out,
        replace_subdirs=frozenset({"tests"}),
    )
    assert sorted(p.name for p in (out / "tests").iterdir()) == [
        "context_riskquant_tests_credit.txt"
    ]
    assert (out / "tests" / "context_riskquant_tests_credit.txt").read_text() == "C2"
    assert (out / "context_riskquant_valuation.txt").read_text() == "V1"


def test_run_without_replace_subdirs_carries_the_subdirectory_forward(
    tmp_path: Path,
) -> None:
    # An index-only or single-domain run must not touch the test split.
    out = tmp_path / "_contexts"
    publish(
        [_rp_at("tests/context_riskquant_tests_credit.txt", "C1")],
        out,
        replace_subdirs=frozenset({"tests"}),
    )
    publish([_rp("index", "I1")], out)
    assert (out / "tests" / "context_riskquant_tests_credit.txt").read_text() == "C1"


def test_retired_monolith_is_removed_and_kept_in_previous(tmp_path: Path) -> None:
    out = tmp_path / "_contexts"
    out.mkdir()
    (out / "context_riskquant_tests.txt").write_text("OLD", encoding="utf-8")
    publish([_rp("valuation", "V1")], out)
    assert not (out / "context_riskquant_tests.txt").exists()
    previous = tmp_path / "_contexts.previous" / "context_riskquant_tests.txt"
    assert previous.read_text() == "OLD"


@pytest.mark.parametrize(
    "filename", ["", "../escape.txt", "/abs.txt", "C:x.txt", "tests\\x.txt"]
)
def test_unsafe_artifact_filename_is_rejected_before_any_write(
    tmp_path: Path, filename: str
) -> None:
    out = tmp_path / "_contexts"
    with pytest.raises(ValueError, match="unsafe output-relative name"):
        publish([_rp_at(filename, "X")], out)
    assert not out.exists()
    assert not (tmp_path / "_contexts.staging").exists()


@pytest.mark.parametrize("subdir", ["", ".", "..", "tests/deep", "/tests"])
def test_unsafe_replace_subdir_is_rejected_before_any_delete(
    tmp_path: Path, subdir: str
) -> None:
    out = tmp_path / "_contexts"
    publish([_rp("valuation", "V1")], out)
    with pytest.raises(ValueError, match="unsafe output-relative name"):
        publish([_rp("valuation", "V2")], out, replace_subdirs=frozenset({subdir}))
    assert (out / "context_riskquant_valuation.txt").read_text() == "V1"
