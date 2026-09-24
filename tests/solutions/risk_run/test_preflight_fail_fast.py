# tests/solutions/risk_run/test_preflight_fail_fast.py
"""
PR-04: composition preflight of `run_scenario_var_es`.

Three contracts were statically knowable from the run's inputs but were first
checked after identity, base valuation, scenario generation and P&L
allocation (or never):

    R1  confidence_level in the open interval (0, 1)      VaR/ES owner's check
    R2  every settlement -> reporting route is quoted      run-derived route (D085)

(R3, the check of caller-supplied FX rows against their names, lost its premise
when the reporting FX rows became run-derived, D085.)

Each rule is pinned by: a valid neighbour that still runs, the exact invalid
composition, a no-numerics proof (the first expensive stages are patched to
raise, so reaching one fails the test with AssertionError), and, where the
rule is the only barrier, a guard-disabled run proving the defect is real.

Fixture: the D084 hostile book (`test_inverted_fx_taylor_admission`), run
axis [USDEUR, EURGBP, EQ_USD, EQ_GBP], reporting USD through USDEUR.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_inverted_fx_taylor_admission import _inputs, all_full, world  # noqa: F401

import riskquant.solutions.risk_run.api as api
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    ReportingSpec,
)

#: The first materially expensive stages of the run, in execution order.
_EXPENSIVE = (
    "build_run_manifest",
    "_value_positions",
    "build_scenario_generator",
    "allocate_pnl_grid",
    "compute_var_es",
)


@pytest.fixture
def no_numerics(monkeypatch):
    """Any expensive stage reached raises AssertionError, not the rule's error."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight let the run reach identity or numerics")

    for stage in _EXPENSIVE:
        monkeypatch.setattr(api, stage, boom)


def _full(w: SimpleNamespace, **over: Any) -> dict[str, Any]:
    kw = _inputs(w, derived=[], block=None)
    kw.update(over)
    return kw


# ---- R1: confidence_level --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [0.0, 1.0, 1.5, -0.1, math.nan, math.inf], ids=lambda v: repr(v)
)
def test_r1_invalid_confidence_fails_before_identity(world, no_numerics, bad) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        run_scenario_var_es(**_full(world, confidence_level=bad))


@pytest.mark.parametrize(
    "edge", [math.nextafter(1.0, 0.0), 5e-324], ids=["below_one", "min_subnormal"]
)
def test_r1_open_interval_edges_are_admitted(world, edge) -> None:
    run = run_scenario_var_es(**_full(world, confidence_level=edge))
    assert run.manifest.risk_measure.confidence_level == edge


def test_r1_without_preflight_the_run_reaches_allocation(world, monkeypatch) -> None:
    """D059: the preflight is the only early barrier; without it an invalid
    level is found only by `compute_var_es`, after the P&L grid exists."""
    monkeypatch.setattr(api, "_validate_confidence_level", lambda value: value)
    reached: list[str] = []
    real = api.allocate_pnl_grid

    def spy(**kwargs: Any):
        reached.append("allocate_pnl_grid")
        return real(**kwargs)

    monkeypatch.setattr(api, "allocate_pnl_grid", spy)
    with pytest.raises(ValueError, match="confidence_level"):
        run_scenario_var_es(**_full(world, confidence_level=1.5))
    assert reached == ["allocate_pnl_grid"]


# ---- R2: the settlement -> reporting route (run-derived, D085) ------------------------


@pytest.mark.parametrize(
    "reporting_ccy, match",
    [
        ("JPY", "no configured FX pair"),
        ("SEK", "not configured"),
    ],
    ids=["configured_ccy_without_a_pair", "unconfigured_ccy"],
)
def test_r2_unroutable_reporting_fails_before_identity(
    world, no_numerics, reporting_ccy, match
) -> None:
    with pytest.raises(CompositionContractError, match=match):
        run_scenario_var_es(
            **_full(world, reporting=ReportingSpec(reporting_ccy=reporting_ccy))
        )


def test_r2_malformed_reporting_ccy_is_refused_at_the_edge() -> None:
    with pytest.raises(ValueError, match="exactly 3 letters"):
        ReportingSpec(reporting_ccy="EU")


def test_r2_neighbouring_route_is_admitted(world, all_full) -> None:
    """EUR reporting through the direct USDEUR quote (USD -> EUR) runs."""
    run = run_scenario_var_es(
        **_full(world, reporting=ReportingSpec(reporting_ccy="EUR"))
    )
    assert run.reporting_ccy == "EUR"
    assert run.identity != all_full.identity
