# tests/market_data/remediation/test_structural_invariants.py
"""Grep-level and API-level invariants that tests of behaviour cannot prove."""

from __future__ import annotations

import re
from pathlib import Path

import hostile_fixture as hf
import pytest

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)

SRC = Path(__file__).resolve().parents[3] / "src" / "riskquant"


def _py_files():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_fx_rf_required_has_exactly_one_definition():
    pattern = re.compile(r"^\s*_?NO_FX[A-Z_]*\s*(?::[^=]*)?=", re.MULTILINE)
    hits = [
        (p.relative_to(SRC).as_posix(), m.group(0).strip())
        for p in _py_files()
        for m in pattern.finditer(p.read_text(encoding="utf-8"))
    ]
    assert hits == [("constants.py", "NO_FX_RF_REQUIRED =")], hits


def test_array_bearing_market_data_containers_use_identity_equality():
    rsc = compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())
    _, m, matrix, state, _ = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    _, m2, matrix2, state2, _ = hf.run_spot_book(
        static_config=rsc, book=hf.SPOT_BOOK, levels=hf.LEVELS, num_scenarios=hf.S
    )
    for a, b in ((m, m2), (matrix, matrix2), (state, state2)):
        assert (a == a) is True
        assert (a == b) is False
        assert isinstance(hash(a), int)
    assert m.fingerprint == m2.fingerprint


@pytest.mark.parametrize(
    "module", ["market_data/builder.py", "market_data/compile.py", "universe.py"]
)
def test_no_raw_numba_decorators_introduced(module):
    text = (SRC / module).read_text()
    assert "@nb.njit" not in text and "@numba.njit" not in text
