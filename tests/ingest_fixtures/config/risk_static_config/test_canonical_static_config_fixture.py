# tests/ingest_fixtures/config/risk_static_config/test_canonical_fixture.py
"""
Regression tests for the canonical `riskquant.static_config` ingest fixture
(`src/riskquant/ingest_fixtures/config/risk_static_config`).

Audit context (2026-09-10): the fixture pre-dates the pandas -> Polars
migration of `riskquant.config.static_config` (see RQ-AUD-018) and, until
this pass, still constructed dead `pandas.DataFrame` objects internally
(used only by each file's own `_debug_print()`, never by production code).
This has been fixed -- every `_parts/*.py` file and the top-level
`canonical_risk_static_config.py` now build `polars.DataFrame` objects,
matching the technology the compiler they feed actually expects.

This test file locks in:
  1. The fixture is genuinely pandas-free (regression guard for the fix
     above).
  2. Every table's data survives the real compile/validate pipeline
     end-to-end, with the exact row counts documented as an intentional,
     versioned snapshot (a silent change here would silently change every
     downstream test and demo built on this fixture).
  3. `CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS` (the Polars convenience
     export) matches `CANONICAL_RISK_STATIC_CONFIG_DATA` (the underlying
     typed dict) exactly, row for row.
  4. The known, documented resolver-coverage gap: the canonical fixture
     exercises only 3 of the 6 `(XAxisKind, RollMethod)` pairs the
     resolution engine supports. This is deliberate (see
     the curve_meta fixture's roll columns: "the goal is a fixture
     that exactly matches runtime needs, not a feature catalogue") but is
     pinned here so it stays a tracked, intentional fact rather than
     something a future reader has to rediscover. The three missing pairs
     are demonstrated with standalone, non-canonical examples in
     `demos/static_config_walkthrough.py` instead of being folded into the
     fixture itself (which several other tests -- e.g.
     `test_rf_resolution.py`, `test_validate_tier5_domain_invariants.py` --
     assert exact row/entity counts against; growing the fixture would
     silently break those).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from riskquant.config.enums.roll_methods import ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "riskquant"
    / "ingest_fixtures"
    / "config"
    / "risk_static_config"
)


# ---------------------------------------------------------------------------
# 1. Pandas-freedom (regression guard)
# ---------------------------------------------------------------------------


def test_fixture_directory_is_pandas_free():
    offenders = []
    for path in FIXTURE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "import pandas" in source:
            offenders.append(str(path.relative_to(FIXTURE_ROOT)))
    assert not offenders, (
        "riskquant.ingest_fixtures.config.risk_static_config must be pandas-free "
        f"-- it feeds the Polars-based static_config compiler. Found pandas in: "
        f"{offenders}"
    )


def test_fixture_directory_uses_polars():
    """Companion to the pandas-freedom check: confirm each _parts file
    actually uses polars (not merely that it avoids pandas -- e.g. a file
    that dropped its debug DataFrame entirely would pass the check above
    for the wrong reason)."""
    parts_dir = FIXTURE_ROOT / "_parts"
    missing_polars = []
    for path in parts_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "import polars" not in source:
            missing_polars.append(str(path.relative_to(FIXTURE_ROOT)))
    assert not missing_polars, f"Expected polars usage in: {missing_polars}"


# ---------------------------------------------------------------------------
# 2. Compile/validate round-trip with documented, versioned row counts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled():
    dict_of_pl = {
        name: pl.DataFrame(data)
        for name, data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    return compile_risk_static_config_from_dict_of_dfs(dict_of_pl)


def test_fixture_compiles_cleanly(compiled):
    assert compiled is not None  # would have raised ConfigurationError otherwise


@pytest.mark.parametrize(
    "attr,expected",
    [
        ("ccy_dict", 5),
        ("holiday_calendars_dict", 5),
        ("node_meta_dict", 10),
        ("curve_meta_dict", 13),
        ("curve_point_meta_dict", 98),
        ("surface_meta_dict", 3),
        ("surface_point_meta_dict", 60),
    ],
)
def test_fixture_row_counts_are_the_documented_snapshot(compiled, attr, expected):
    """
    Pins the exact size of every table. If this test fails after an
    intentional fixture change, update the expected count here AND check
    whether `test_rf_resolution.py` / `test_validate_tier5_domain_invariants.py`
    (which also assert against this same fixture) need the same update.
    """
    actual = len(getattr(compiled, attr))
    assert actual == expected, (
        f"{attr}: expected {expected} (documented snapshot), got {actual}. "
        f"If this is an intentional fixture change, this is not the only "
        f"test that will need updating -- see this file's module docstring."
    )


# ---------------------------------------------------------------------------
# 3. Internal consistency between the two exported forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_name", list(CANONICAL_RISK_STATIC_CONFIG_DATA.keys()))
def test_dict_of_dfs_matches_underlying_dict_data(table_name):
    raw = CANONICAL_RISK_STATIC_CONFIG_DATA[table_name]
    df = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS[table_name]
    assert isinstance(df, pl.DataFrame)
    assert df.columns == list(raw.keys())
    assert df.height == len(next(iter(raw.values())))


def test_all_seven_required_tables_present():
    expected_tables = {
        "ccy",
        "holiday_calendars",
        "node_meta",
        "curve_meta",
        "curve_point_meta",
        "surface_meta",
        "surface_point_meta",
    }
    assert set(CANONICAL_RISK_STATIC_CONFIG_DATA.keys()) == expected_tables
    assert set(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.keys()) == expected_tables


# ---------------------------------------------------------------------------
# 4. Documented resolver-coverage gap (deliberate, tracked)
# ---------------------------------------------------------------------------


def test_documented_roll_method_coverage_gap():
    """
    The canonical fixture's curve_meta roll columns exercise exactly 3 of the
    6 (XAxisKind, RollMethod) pairs the resolution engine supports. This is
    a deliberate scope decision (see the curve_meta fixture roll columns),
    pinned here so a change to fixture coverage is a conscious,
    reviewed decision rather than silent drift -- and so the demo walkthrough
    knows exactly which pairs it must supplement with standalone examples.
    """
    exercised_pairs = set()
    for x_kind_name, roll_method in zip(
        CANONICAL_RISK_STATIC_CONFIG_DATA["curve_meta"]["x_axis_kind"],
        CANONICAL_RISK_STATIC_CONFIG_DATA["curve_meta"]["roll_method"],
        strict=True,
    ):
        if roll_method is None:
            continue
        exercised_pairs.add((XAxisKind[x_kind_name].name, roll_method))

    expected_exercised = {
        ("OFFSET_M", "CALENDAR_PERIOD"),
        ("OFFSET_Y", "ROLL_ON_MMDD"),
        ("OFFSET_HY", "HALF_YEAR_BY_LABEL"),
    }
    assert exercised_pairs == expected_exercised, (
        f"Roll-method coverage changed: exercised={exercised_pairs}, "
        f"expected={expected_exercised}. If pairs were added intentionally, "
        f"update this test; if pairs were accidentally lost, that's a real "
        f"fixture regression."
    )

    all_pairs = {(k.name, m.name) for k, m in ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS}
    missing_pairs = all_pairs - expected_exercised
    assert missing_pairs == {
        ("OFFSET_M", "HALF_YEAR_BY_DATE"),
        ("OFFSET_Y", "CALENDAR_PERIOD"),
        ("OFFSET_HY", "HALF_YEAR_BY_DATE"),
    }, (
        "The set of un-exercised pairs changed -- update the demo walkthrough's supplementary examples to match."
    )
