# tests/solutions/credit_var/test_pandas_free_results.py
"""
WS_CREDIT_POLARS S1: credit_var results stay equal to v0.2.0 (C.4) and the
pandas cap is gone (C.18).

S1.T3 runs the request set Q and compares each result JSON, without
``provenance.generated_at`` and ``provenance.engine_version`` (F-27), with
the record that `_gates/record_s1_goldens.py` made at tag v0.2.0. Q is
`test_result_contract._book(key)` for each key of `_BOOK_KEYS`, plus the
WS_CREDIT_VAR_API S2.T6 accepted case (one obligor at sum(loading^2) =
1 + 0.5e-9). The record also holds each request JSON, so a change of Q fails
here as a Q change and not as a result change.

Reproducibility is environment-relative (D-C15): the comparison runs only at
the numpy, scipy and numba versions of the record. At other versions the test
fails and names each difference; a new record needs an owner decision.

Every later slice runs this file unchanged (S3 to S6).
"""

from __future__ import annotations

import ast
import functools
import json
import tomllib
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
import scipy

from riskquant.solutions.credit_var.api import CreditVarRequest, run_credit_var
from tests.solutions.credit_var.test_request_contract import (
    _book_with_obligor_20_loadings,
    _two_factor_loadings,
)
from tests.solutions.credit_var.test_result_contract import _BOOK_KEYS, _book

_HERE = Path(__file__).resolve().parent
_GOLDEN = _HERE / "golden" / "credit_var_results_v0_2_0.json"
_PYPROJECT = _HERE.parents[2] / "pyproject.toml"
_REQUEST_CONTRACT = _HERE / "test_request_contract.py"
_EXTRA_CASE = "mf_gaussian_sum_sq_1_plus_0p5e-9"
_EXCLUDE: dict[str, Any] = {"provenance": {"generated_at", "engine_version"}}
_CATALOG_KEYS = frozenset({
    "forced_default",
    "independent_bernoulli",
    "stress_regime_bernoulli",
    "stress_regime_bernoulli_grouped",
    "beta_binomial",
    "beta_binomial_grouped",
    "sf_gaussian",
    "sf_gaussian_basel",
    "sf_students_t",
    "sf_students_t_basel",
    "mf_gaussian",
    "marshall_olkin",
    "mf_students_t",
})  # F-20, written out here so that a shrunk _BOOK_KEYS fails


@functools.cache
def _golden() -> dict[str, Any]:
    with _GOLDEN.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _q() -> dict[str, dict[str, Any]]:
    books = {key: _book(key) for key in _BOOK_KEYS}
    books[_EXTRA_CASE] = _book_with_obligor_20_loadings(
        *_two_factor_loadings(1.0 + 0.5e-9)
    )
    return books


def _version_differences() -> list[str]:
    golden = _golden()
    runtime = {
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
    }
    return [
        f"{name}: recorded {golden[name]}, runtime {value}"
        for name, value in runtime.items()
        if golden[name] != value
    ]


# ---------------------------------------------------------------------------
# S1.T3 (C.4)
# ---------------------------------------------------------------------------


def test_s1_t3_q_holds_the_13_catalog_keys_and_the_s2_t6_case() -> None:
    assert frozenset(_BOOK_KEYS) == _CATALOG_KEYS
    assert set(_golden()["cases"]) == _CATALOG_KEYS | {_EXTRA_CASE}


@pytest.mark.parametrize("case", sorted(_CATALOG_KEYS | {_EXTRA_CASE}))
def test_s1_t3_result_json_equals_the_v0_2_0_record(case: str) -> None:
    recorded = _golden()["cases"][case]
    request = CreditVarRequest.model_validate(_q()[case])
    assert request.model_dump_json() == recorded["request"], (
        f"{case}: the request of Q differs from the recorded request"
    )
    differences = _version_differences()
    if differences:
        pytest.fail(
            "C.4 compares only at the recorded numpy, scipy and numba versions "
            f"(D-C15): {'; '.join(differences)}"
        )
    result = run_credit_var(request)
    assert result.model_dump_json(exclude=_EXCLUDE) == recorded["result"], case


def test_s1_t3_version_check_names_each_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__version__", "0.0.1")
    monkeypatch.setattr(scipy, "__version__", "0.0.2")
    differences = _version_differences()
    assert len(differences) == 2
    assert differences[0].startswith("numpy_version: recorded ")
    assert differences[0].endswith("runtime 0.0.1")
    assert differences[1].startswith("scipy_version: recorded ")


# ---------------------------------------------------------------------------
# S1.T5 (C.18)
# ---------------------------------------------------------------------------


def test_s1_t5_pandas_has_no_upper_bound_and_stubs_are_unchanged() -> None:
    with _PYPROJECT.open("rb") as handle:
        dependencies: list[str] = tomllib.load(handle)["project"]["dependencies"]
    pandas_specs = [
        d for d in dependencies if d.split(">")[0].split("<")[0] == "pandas"
    ]
    assert pandas_specs == ["pandas>=2.0"]
    assert "pandas-stubs>=2.0" in dependencies


# ---------------------------------------------------------------------------
# S1.T6: the S2.T6 oracle depends on neither pandas nor the code under test
# ---------------------------------------------------------------------------


def test_s1_t6_engine_grouped_sum_oracle_is_the_golden_record() -> None:
    tree = ast.parse(_REQUEST_CONTRACT.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_engine_grouped_sum"
    ]
    assert len(functions) == 1
    names = {
        node.id for node in ast.walk(functions[0]) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(functions[0]) if isinstance(node, ast.Attribute)
    }
    assert not names & {"pd", "pandas", "groupby"}, names
    assert "loading_sum_sq" not in names
    assert not [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
