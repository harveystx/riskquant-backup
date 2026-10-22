# tests/e2e/test_factor_regression_notebook.py
"""
The factor-regression demo notebook, run headless as a regression gate.

Pattern of `test_demo_scripts_are_runnable.py`: the demo is executed, not
inspected. `notebooks/factor_regression_demo.py` is loaded by path (the
`notebooks` folder is not a package) and run with `app.run()`, which executes
every cell in dependency order without a browser or a kernel. The last
section of the notebook is the canonical catalog self-check: one row per
canonical analysis with its recorded and observed outcome. The gate asserts
that this table holds every analysis of the catalog and that every row
matches.

The notebook extras (`marimo`, `matplotlib`) are optional dependencies. The
gate skips, it does not fail, when they are absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from riskquant.ingest_fixtures.factor_regression import analyses

marimo = pytest.importorskip("marimo")
matplotlib = pytest.importorskip("matplotlib")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _REPO_ROOT / "notebooks" / "factor_regression_demo.py"

_SELF_CHECK_COLUMNS = ("kind", "name", "expected", "observed", "match")


def _load_notebook() -> ModuleType:
    assert _NOTEBOOK.is_file(), f"notebook not found: {_NOTEBOOK}"
    spec = importlib.util.spec_from_file_location("factor_regression_demo", _NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _self_check_tables(outputs: tuple[object, ...]) -> list[pl.DataFrame]:
    return [
        o
        for o in outputs
        if isinstance(o, pl.DataFrame) and tuple(o.columns) == _SELF_CHECK_COLUMNS
    ]


@pytest.fixture(scope="module")
def notebook_outputs() -> tuple[object, ...]:
    # Headless: figures are built but never shown.
    matplotlib.use("Agg")
    module = _load_notebook()
    outputs, _defs = module.app.run()
    return tuple(outputs)


def test_notebook_runs_and_every_catalog_outcome_matches(
    notebook_outputs: tuple[object, ...],
) -> None:
    tables = _self_check_tables(notebook_outputs)
    assert len(tables) == 1, f"expected one self-check table, found {len(tables)}"
    table = tables[0]
    expected_rows = (
        len(analyses.CANONICAL_OLS_ANALYSES)
        + len(analyses.CANONICAL_WALD_HYPOTHESES)
        + len(analyses.CANONICAL_UNIT_ROOT_ANALYSES)
    )
    assert expected_rows == 33, expected_rows
    assert table.height == expected_rows, table
    mismatched = table.filter(~pl.col("match"))
    assert mismatched.height == 0, mismatched
