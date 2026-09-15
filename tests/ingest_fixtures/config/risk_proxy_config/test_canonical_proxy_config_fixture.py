# tests/ingest_fixtures/config/risk_proxy_config/test_canonical_fixture.py
"""
Regression tests for the canonical `riskquant.proxy_config` ingest fixture
(`src/riskquant/ingest_fixtures/config/risk_proxy_config`).

Mirrors `tests/ingest_fixtures/config/risk_static_config/test_canonical_fixture.py`.

Audit context (2026-09-10): like the static_config fixture before it, this
fixture pre-dated the pandas -> Polars migration of `riskquant.config.proxy_config`
(RQ-AUD-018) and still constructed a dead `pandas.DataFrame`
(`CANONICAL_RISK_PROXY_CONFIG_DF`) -- unlike the static_config case, this
object had FOUR real consumers (three demo scripts plus its own
`_debug_print`), all of which were silently broken by the type mismatch
before this fix (confirmed by direct execution, not assumed -- see the
ledger for the exact tracebacks). This has been fixed: the fixture now
builds a genuine `polars.DataFrame`, matching what
`compile_proxy_topology_from_df` actually expects.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DATA,
    CANONICAL_RISK_PROXY_CONFIG_DF,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "riskquant"
    / "ingest_fixtures"
    / "config"
    / "risk_proxy_config"
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
        "riskquant.ingest_fixtures.config.risk_proxy_config must be pandas-free "
        f"-- it feeds the Polars-based proxy_config compiler. Found pandas in: "
        f"{offenders}"
    )


def test_fixture_uses_polars():
    source = (FIXTURE_ROOT / "canonical_risk_proxy_config.py").read_text()
    assert "import polars" in source


def test_dict_of_dfs_export_is_a_polars_dataframe():
    assert isinstance(CANONICAL_RISK_PROXY_CONFIG_DF, pl.DataFrame)


# ---------------------------------------------------------------------------
# 2. Structural integrity of the raw data
# ---------------------------------------------------------------------------


def test_fixture_has_six_rows_three_container_three_rf():
    levels = CANONICAL_RISK_PROXY_CONFIG_DATA["proxying_level"]
    assert len(levels) == 6
    assert levels.count("CONTAINER") == 3
    assert levels.count("RISK_FACTOR") == 3


def test_dict_of_dfs_matches_underlying_dict_data():
    assert CANONICAL_RISK_PROXY_CONFIG_DF.columns == list(
        CANONICAL_RISK_PROXY_CONFIG_DATA.keys()
    )
    assert CANONICAL_RISK_PROXY_CONFIG_DF.height == len(
        CANONICAL_RISK_PROXY_CONFIG_DATA["proxying_level"]
    )


def test_no_required_name_is_also_a_proxy_name():
    """Single-hop policy, checked at the fixture-data level independent
    of the validator (belt-and-braces: the validator's own chain check
    is tested in test_full_pipeline_equivalence.py; this pins the SAME
    invariant directly against the raw fixture data)."""
    required = set(CANONICAL_RISK_PROXY_CONFIG_DATA["required_name"])
    proxy = set(CANONICAL_RISK_PROXY_CONFIG_DATA["proxy_name"])
    assert required.isdisjoint(proxy), (
        f"Chain risk: names appearing as both required_name and proxy_name: "
        f"{required & proxy}"
    )


def test_no_self_reference():
    for req, prx in zip(
        CANONICAL_RISK_PROXY_CONFIG_DATA["required_name"],
        CANONICAL_RISK_PROXY_CONFIG_DATA["proxy_name"],
        strict=True,
    ):
        assert req != prx


# ---------------------------------------------------------------------------
# 3. Compile round-trip against the real static config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_static_config():
    dict_of_pl = {
        name: pl.DataFrame(table_data)
        for name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    return compile_risk_static_config_from_dict_of_dfs(dict_of_pl)


def test_fixture_compiles_cleanly(real_static_config):
    topology = compile_proxy_topology_from_df(
        df=CANONICAL_RISK_PROXY_CONFIG_DF, static_config=real_static_config
    )
    assert topology.num_container_proxies == 3
    assert topology.num_rf_proxies == 3


# ---------------------------------------------------------------------------
# 4. Documented proxy-level coverage gap (deliberate, tracked)
# ---------------------------------------------------------------------------


def test_documented_proxy_coverage_gap(real_static_config):
    """
    The canonical fixture's container-level rows all proxy CURVE
    containers; no SURFACE container is proxied, and no surface-point
    rf_id is used as an rf-level required_name. Both are genuinely
    supported (see `audit.py`'s kind_label mappings, which include
    SURFACE and SURFACE_POINT, and
    `tests/config/proxy_config/test_cross_level_conflict.py`, which
    exercises the surface path directly) but are not exercised by this
    fixture. Pinned here, mirroring
    `test_canonical_fixture.py::test_documented_roll_method_coverage_gap`
    in the static_config fixture tests, so a change to fixture coverage
    is a conscious, reviewed decision rather than silent drift.
    """
    from riskquant.config.enums.container_topology import ContainerTopologyType

    name_to_cid = real_static_config.container_name_to_id
    topology_type_arr = real_static_config.container_topology_type

    proxied_kinds = set()
    for name in CANONICAL_RISK_PROXY_CONFIG_DATA["required_name"]:
        if name not in name_to_cid:
            continue  # rf-level required_name, not a container
        cid = int(name_to_cid[name])
        kind = int(topology_type_arr[real_static_config.container_slot(cid)])
        proxied_kinds.add(ContainerTopologyType(kind).name)

    assert proxied_kinds == {"CURVE"}, (
        f"Container-level proxy coverage changed: {proxied_kinds}. If SURFACE "
        f"coverage was added intentionally, update this test and the "
        f"demo walkthrough's 'coverage gap' section to match."
    )
