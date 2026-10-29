# _context_gen/tests/test_suite_partition.py
"""
Contract tests for the per-domain test split (suite_partition).

The imports under test come from real syntax via `ast_model.
extract_raw_imports`, so these tests also pin the hand-off between the
syntactic layer and the resolver, not only the resolver alone.
"""

from __future__ import annotations

import ast
import random

import pytest

from _context_gen.ast_model import RawImport, dotted_name_for, extract_raw_imports
from _context_gen.suite_partition import (
    ROOT_BUCKET,
    SuiteBucket,
    bucket_of,
    plan_buckets,
    suite_dir_for,
)


def _imports(sources: dict[str, str]) -> dict[str, tuple[RawImport, ...]]:
    out: dict[str, tuple[RawImport, ...]] = {}
    for rel, src in sources.items():
        dotted = dotted_name_for(rel, "tests/", "tests")
        assert dotted is not None
        out[rel] = extract_raw_imports(ast.parse(src), dotted)
    return out


def _by_key(buckets: tuple[SuiteBucket, ...]) -> dict[str, SuiteBucket]:
    return {b.key: b for b in buckets}


@pytest.mark.parametrize(
    ("path", "key", "directory"),
    [
        ("tests/test_architecture.py", ROOT_BUCKET, "tests"),
        ("tests/conftest.py", ROOT_BUCKET, "tests"),
        ("tests/credit/test_x.py", "credit", "tests/credit"),
        ("tests/credit/jtd/kernels/test_y.py", "credit", "tests/credit"),
        ("tests/credit/solutions/test_z.py", "credit", "tests/credit"),
        ("tests/_support/spot_universe.py", "_support", "tests/_support"),
        ("tests/solutions/conftest.py", "solutions", "tests/solutions"),
        (
            "tests/solutions/credit_var/test_api.py",
            "solutions_credit_var",
            "tests/solutions/credit_var",
        ),
        (
            "tests/solutions/sensitivity_pnl/unit/test_api.py",
            "solutions_sensitivity_pnl",
            "tests/solutions/sensitivity_pnl",
        ),
        (
            "tests/portfolio/ingest/_pandas_era_oracle.json",
            "portfolio",
            "tests/portfolio",
        ),
    ],
)
def test_bucket_rule(path: str, key: str, directory: str) -> None:
    assert bucket_of(path) == (key, directory)


@pytest.mark.parametrize("path", ["tests", "tests/", "src/riskquant/x.py", ""])
def test_bucket_of_rejects_paths_outside_tests(path: str) -> None:
    with pytest.raises(ValueError, match="not a file under tests/"):
        bucket_of(path)


@pytest.mark.parametrize(
    "profile_name",
    ["credit", "valuation", "solutions_credit_var", "solutions_risk_run"],
)
def test_suite_dir_for_is_the_inverse_of_bucket_of(profile_name: str) -> None:
    directory = suite_dir_for(profile_name)
    assert bucket_of(f"{directory}/test_any.py") == (profile_name, directory)


def test_key_collision_fails_loud() -> None:
    paths = [
        "tests/solutions/credit_var/test_a.py",
        "tests/solutions_credit_var/test_b.py",
    ]
    with pytest.raises(ValueError, match="collision"):
        plan_buckets(paths=paths, raw_imports={})


def test_root_bucket_collides_with_a_directory_named_root() -> None:
    with pytest.raises(ValueError, match="collision"):
        plan_buckets(
            paths=["tests/test_top.py", "tests/root/test_x.py"], raw_imports={}
        )


def test_support_closure_is_transitive_and_adds_ancestor_conftests() -> None:
    sources = {
        "tests/conftest.py": "",
        "tests/solutions/conftest.py": "",
        "tests/_support/spot_universe.py": (
            "from tests._support import spot_scenario\n"
        ),
        "tests/_support/spot_scenario.py": "import numpy as np\n",
        "tests/_support/identity_rows.py": "",
        "tests/credit/jtd/test_bridge.py": (
            "from tests._support.spot_universe import build\n"
            "import tests._support.identity_rows\n"
        ),
        "tests/solutions/credit_var/test_api.py": "",
    }
    buckets = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))

    credit = buckets["credit"]
    assert credit.own_files == ("tests/credit/jtd/test_bridge.py",)
    assert credit.support_files == (
        "tests/_support/identity_rows.py",
        "tests/_support/spot_scenario.py",
        "tests/_support/spot_universe.py",
        "tests/conftest.py",
    )
    assert credit.unresolved_imports == ()

    credit_var = buckets["solutions_credit_var"]
    assert credit_var.directory == "tests/solutions/credit_var"
    assert credit_var.support_files == (
        "tests/conftest.py",
        "tests/solutions/conftest.py",
    )

    root = buckets[ROOT_BUCKET]
    assert root.own_files == ("tests/conftest.py",)
    assert root.support_files == ()


def test_in_bucket_imports_are_not_support() -> None:
    sources = {
        "tests/solutions/risk_run/_fixture.py": "",
        "tests/solutions/risk_run/test_run.py": (
            "from tests.solutions.risk_run._fixture import build\n"
        ),
    }
    bucket = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))[
        "solutions_risk_run"
    ]
    assert bucket.support_files == ()
    assert bucket.unresolved_imports == ()


def test_namespace_package_imports_resolve_without_a_file() -> None:
    sources = {
        "tests/credit/jtd/test_a.py": (
            "import tests.credit\nfrom tests import credit\nfrom tests.credit import jtd\n"
        ),
    }
    bucket = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))[
        "credit"
    ]
    assert bucket.support_files == ()
    assert bucket.unresolved_imports == ()


def test_unresolved_tests_imports_are_reported() -> None:
    sources = {
        "tests/_support/spot_universe.py": "",
        "tests/utils/test_dates.py": (
            "from tests._support import nope_helper\n"
            "import tests.missing.module\n"
            "from tests.ghost import thing\n"
        ),
    }
    bucket = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))[
        "utils"
    ]
    assert bucket.unresolved_imports == (
        "tests._support.nope_helper",
        "tests.ghost.thing",
        "tests.missing.module",
    )


def test_unresolved_import_in_a_support_file_is_reported_by_the_importer() -> None:
    sources = {
        "tests/_support/spot_universe.py": "from tests._support import gone\n",
        "tests/credit/test_x.py": "from tests._support import spot_universe\n",
    }
    buckets = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))
    assert buckets["credit"].unresolved_imports == ("tests._support.gone",)
    assert buckets["_support"].unresolved_imports == ("tests._support.gone",)


def test_non_tests_imports_are_ignored() -> None:
    sources = {
        "tests/credit/test_x.py": (
            "import numpy as np\n"
            "from riskquant.credit import api\n"
            "from testsuite import helper\n"
        ),
    }
    bucket = _by_key(plan_buckets(paths=sources, raw_imports=_imports(sources)))[
        "credit"
    ]
    assert bucket.support_files == ()
    assert bucket.unresolved_imports == ()


def test_file_without_parsed_imports_still_creates_its_bucket() -> None:
    # An unreadable or unparsable file has no raw imports in the snapshot.
    # Its bucket must still exist, so the renderer can report the failure.
    buckets = _by_key(
        plan_buckets(
            paths=["tests/reporting/test_fx.py", "tests/reporting/oracle.json"],
            raw_imports={},
        )
    )
    assert buckets["reporting"].own_files == (
        "tests/reporting/oracle.json",
        "tests/reporting/test_fx.py",
    )


def test_support_target_without_parsed_imports_is_still_embedded() -> None:
    # A support file that failed to parse is still a known module: it is
    # embedded (and reported by the renderer), never treated as missing.
    sources = {"tests/credit/test_x.py": "from tests._support import broken\n"}
    raw = _imports(sources)
    buckets = _by_key(
        plan_buckets(
            paths=[*sources, "tests/_support/broken.py"],
            raw_imports=raw,
        )
    )
    assert buckets["credit"].support_files == ("tests/_support/broken.py",)
    assert buckets["credit"].unresolved_imports == ()


def test_plan_is_independent_of_input_order() -> None:
    sources = {
        "tests/conftest.py": "",
        "tests/_support/a.py": "from tests._support import b\n",
        "tests/_support/b.py": "",
        "tests/credit/test_x.py": "from tests._support import a\n",
        "tests/solutions/credit_var/test_y.py": "from tests._support import b\n",
        "tests/valuation/oracle/fixture.json": "",
    }
    raw = _imports({k: v for k, v in sources.items() if k.endswith(".py")})
    expected = plan_buckets(paths=sorted(sources), raw_imports=raw)
    shuffled = list(sources)
    random.Random(7).shuffle(shuffled)
    assert plan_buckets(paths=shuffled, raw_imports=raw) == expected
    assert [b.key for b in expected] == sorted(b.key for b in expected)
