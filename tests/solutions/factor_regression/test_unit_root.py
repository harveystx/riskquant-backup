# tests/solutions/factor_regression/test_unit_root.py
"""
Tests for S4: unit-root contracts, digests and entry point.

Test IDs (slice table)
----------------------
S4.T1  - CombinationTerm and LinearCombination type and domain checks
S4.T2  - UnitRootSpec type and domain checks
S4.T3  - UnitRootResult fields: no p-value, no verdict field
S4.T4  - UnitRootResult arrays: read-only; inputs unchanged
S4.T5  - LOG level 0.0 on a named date -> ScaleDomainError with context
S4.T6  - 21 aligned dates lag 2 -> InsufficientObservationsError
S4.T7  - digests in two fresh processes equal
S4.T8  - one mutation per field -> digest changes as expected
S4.T9  - no statsmodels import in the slice packages; pyproject.toml has no
         statsmodels.  The facade allow-list scan is S6.T14 of test_digest.py.
S4.T10 - existing S6.T9 tests of test_digest.py and S7 tests of test_api.py
         (they run in the slice; no test here)
S4.T11 - existing S7.T7 tests of test_api.py (they run in the slice; no test
         here)
S4.T12 - Scale and Deterministic values match core codes
S4.T13 - independent hashlib/numpy oracle for both new digests
S4.T14 - one case per C.10 mapping

Extra tests named in decision 10
---------------------------------
- api.py does not import hashing
- unit_root_test checks argument types in the declared order
- masked-array and ndarray-subclass rejection for every array arg
- caller arrays unchanged after align_levels and unit_root_test
- two equal calls give equal results

Fix round 2
-----------
- R-001: a lag far above m raises the C.6 size error at once (child process
  with a timeout); exact labels for lag 3 with CONSTANT_TREND
- R-002: row dates (A.1), labels (A.3), units, window, exclusions, critical
  levels, rank_rtol reaches the core, float storage of rank_rtol
- R-003: the three digests of a result against a hashlib and numpy oracle
  built from the input levels; the A.2 and A.4 texts typed in this file
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy import ma
from numpy.typing import NDArray

from riskquant.exceptions import RiskQuantError
from riskquant.factor_regression import codes
from riskquant.solutions.factor_regression import digest, labels
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    CombinationTerm,
    DegenerateFitError,
    Deterministic,
    InsufficientObservationsError,
    LinearCombination,
    ModelSpecError,
    RankDeficiencyError,
    Scale,
    ScaleDomainError,
    UnitRootResult,
    UnitRootSpec,
    align_levels,
    unit_root_test,
)
from riskquant.solutions.factor_regression.contracts import (
    ENGINE_VERSION,
    ObservedSeries,
)

# tests/solutions/factor_regression/test_unit_root.py: parents[3] is the
# repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FACADE_DIR = _REPO_ROOT / "src" / "riskquant" / "solutions" / "factor_regression"


# ------------------------------------------------------------------ helpers


def _dates(n: int, start: int = 0) -> NDArray[np.int32]:
    return np.arange(start, start + n, dtype=np.int32)


def _vals(n: int, seed: int = 0) -> NDArray[np.float64]:
    return np.asarray(
        np.cumsum(np.random.default_rng(seed).standard_normal(n)) + 50.0,
        dtype=np.float64,
    )


def _series(name: str = "P", n: int = 40, seed: int = 0) -> ObservedSeries:
    return ObservedSeries(
        name=name,
        unit="USD",
        date_ord=_dates(n),
        values=_vals(n, seed),
    )


def _aligned(series: list[ObservedSeries] | None = None, n: int = 40) -> AlignedLevels:
    if series is None:
        series = [_series("P", n)]
    return align_levels(series, start_ord=0, end_ord=n)


def _spec(
    names: list[str] | None = None,
    weights: list[float] | None = None,
    scale: Scale = Scale.LEVEL,
    det: Deterministic = Deterministic.CONSTANT,
    lag: int = 0,
    rtol: float = 1e-10,
) -> UnitRootSpec:
    if names is None:
        names = ["P"]
    if weights is None:
        weights = [1.0] * len(names)
    terms = tuple(
        CombinationTerm(series=n, weight=w) for n, w in zip(names, weights, strict=True)
    )
    return UnitRootSpec(
        combination=LinearCombination(terms=terms, scale=scale),
        deterministic=det,
        lag=lag,
        rank_rtol=rtol,
    )


def _run_ok(
    n: int = 40, lag: int = 0, det: Deterministic = Deterministic.CONSTANT
) -> UnitRootResult:
    levels = _aligned(n=n)
    spec = _spec(lag=lag, det=det)
    return unit_root_test(levels, spec)


# ================================================================ S4.T1
# CombinationTerm and LinearCombination: type and domain checks.


@pytest.mark.parametrize(
    ("kw", "exc"),
    [
        ({"series": 1, "weight": 1.0}, TypeError),  # series not str
        ({"series": "P", "weight": True}, TypeError),  # weight bool
        ({"series": "P", "weight": np.bool_(True)}, TypeError),  # weight np.bool_
        ({"series": "P", "weight": "1"}, TypeError),  # weight str
        ({"series": "P", "weight": None}, TypeError),  # weight None
        ({"series": "P", "weight": Fraction(1)}, TypeError),  # weight Fraction
    ],
    ids=[
        "series-int",
        "weight-True",
        "weight-npbool",
        "weight-str",
        "weight-None",
        "weight-Fraction",
    ],
)
def test_t1_term_type_error(kw: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        CombinationTerm(**kw)


@pytest.mark.parametrize(
    ("kw", "exc"),
    [
        ({"series": "", "weight": 1.0}, ModelSpecError),  # empty series
        ({"series": "P\x85Q", "weight": 1.0}, ModelSpecError),  # control char C1
        ({"series": "P", "weight": 0.0}, ModelSpecError),  # zero weight
        ({"series": "P", "weight": float("nan")}, ModelSpecError),  # nan
        ({"series": "P", "weight": float("inf")}, ModelSpecError),  # inf
        ({"series": "P", "weight": -float("inf")}, ModelSpecError),  # -inf
        ({"series": "P", "weight": 10**400}, ModelSpecError),  # overflow int
    ],
    ids=[
        "empty-series",
        "ctrl-char",
        "zero-weight",
        "nan",
        "inf",
        "-inf",
        "overflow-int",
    ],
)
def test_t1_term_model_spec_error(kw: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        CombinationTerm(**kw)


def test_t1_weight_int_stored_as_float() -> None:
    t = CombinationTerm(series="P", weight=2)
    assert t.weight == 2.0
    assert isinstance(t.weight, float)


@pytest.mark.parametrize(
    ("terms", "scale", "exc"),
    [
        ("not-a-tuple", Scale.LEVEL, TypeError),  # terms not tuple
        (
            (CombinationTerm(series="P", weight=1.0),),
            "LEVEL",
            TypeError,
        ),  # scale not Scale
    ],
    ids=["terms-not-tuple", "scale-str"],
)
def test_t1_linear_combination_type_error(
    terms: object, scale: object, exc: type[Exception]
) -> None:
    with pytest.raises(exc):
        LinearCombination(terms=cast(Any, terms), scale=cast(Any, scale))


@pytest.mark.parametrize(
    ("terms", "exc"),
    [
        ((), ModelSpecError),  # empty tuple
        (
            (
                CombinationTerm(series="P", weight=1.0),
                CombinationTerm(series="P", weight=2.0),
            ),
            ModelSpecError,
        ),  # repeated series
    ],
    ids=["empty-terms", "repeated-series"],
)
def test_t1_linear_combination_model_spec_error(
    terms: tuple[CombinationTerm, ...], exc: type[Exception]
) -> None:
    with pytest.raises(exc):
        LinearCombination(terms=terms, scale=Scale.LEVEL)


def test_t1_combination_term_unknown_series_raises_model_spec_error() -> None:
    """A series not in the aligned levels raises ModelSpecError from unit_root_test."""
    levels = _aligned()
    spec = _spec(names=["UNKNOWN"])
    with pytest.raises(ModelSpecError):
        unit_root_test(levels, spec)


# ================================================================ S4.T2
# UnitRootSpec type and domain checks.


@pytest.mark.parametrize(
    ("kw",),
    [
        (
            {
                "combination": _spec().combination,
                "deterministic": Deterministic.CONSTANT,
                "lag": True,
                "rank_rtol": 1e-10,
            },
        ),  # lag bool
        (
            {
                "combination": _spec().combination,
                "deterministic": Deterministic.CONSTANT,
                "lag": np.bool_(True),
                "rank_rtol": 1e-10,
            },
        ),  # lag np.bool_
        (
            {
                "combination": _spec().combination,
                "deterministic": Deterministic.CONSTANT,
                "lag": 1.5,
                "rank_rtol": 1e-10,
            },
        ),  # lag float
        (
            {
                "combination": _spec().combination,
                "deterministic": Deterministic.CONSTANT,
                "lag": "2",
                "rank_rtol": 1e-10,
            },
        ),  # lag str
        (
            {
                "combination": _spec().combination,
                "deterministic": "NONE",
                "lag": 0,
                "rank_rtol": 1e-10,
            },
        ),  # deterministic str
    ],
    ids=["lag-True", "lag-npbool", "lag-float", "lag-str", "det-str"],
)
def test_t2_spec_type_error(kw: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        UnitRootSpec(**kw)


def test_t2_spec_omitted_field_raises_type_error() -> None:
    # The field rank_rtol is omitted.  The call goes through Any, because a
    # static checker rejects the call that this test must make.
    kw: dict[str, Any] = {
        "combination": _spec().combination,
        "deterministic": Deterministic.CONSTANT,
        "lag": 0,
    }
    with pytest.raises(TypeError):
        UnitRootSpec(**kw)


@pytest.mark.parametrize(
    ("lag", "rtol", "exc"),
    [
        (-1, 1e-10, ModelSpecError),  # lag negative
        (2**63, 1e-10, ModelSpecError),  # lag above int64
        (0, 1.0, ModelSpecError),  # rank_rtol 1.0 (not in open interval)
        (0, float("nan"), ModelSpecError),  # rank_rtol nan
    ],
    ids=["lag-neg", "lag-too-big", "rtol-1.0", "rtol-nan"],
)
def test_t2_spec_domain_error(lag: int, rtol: float, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        UnitRootSpec(
            combination=_spec().combination,
            deterministic=Deterministic.CONSTANT,
            lag=lag,
            rank_rtol=rtol,
        )


def test_t2_lag_max_accepted() -> None:
    spec = UnitRootSpec(
        combination=_spec().combination,
        deterministic=Deterministic.CONSTANT,
        lag=2**63 - 1,
        rank_rtol=1e-10,
    )
    assert spec.lag == 2**63 - 1


# ================================================================ S4.T3
# UnitRootResult: no p-value and no verdict field.


def test_t3_no_verdict_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(UnitRootResult)}
    for forbidden in (
        "p_value",
        "rejected",
        "accepted",
        "stationary",
        "valid",
        "approved",
    ):
        assert forbidden not in field_names, (
            f"UnitRootResult must not have field {forbidden!r}"
        )


# ================================================================ S4.T4
# UnitRootResult arrays read-only; caller arrays unchanged.


def test_t4_arrays_read_only() -> None:
    result = _run_ok()
    for fname in (
        "coef",
        "se",
        "critical_values",
        "row_start_date_ord",
        "row_end_date_ord",
    ):
        arr = getattr(result, fname)
        assert not arr.flags["WRITEABLE"], f"UnitRootResult.{fname} must be read-only"


@pytest.mark.parametrize("scale", [Scale.LEVEL, Scale.LOG], ids=["LEVEL", "LOG"])
def test_t4_caller_arrays_unchanged(scale: Scale) -> None:
    """
    No caller array changes: the ObservedSeries inputs and the AlignedLevels.

    The raw input arrays stay writeable, so a write by align_levels or by
    unit_root_test into a retained view changes them and this test fails.
    """
    n = 40
    names = ("P", "Z")
    raw_dates = [_dates(n), _dates(n)]
    raw_values = [_vals(n, seed=1), _vals(n, seed=2)]
    date_copies = [arr.copy() for arr in raw_dates]
    value_copies = [arr.copy() for arr in raw_values]
    series = [
        ObservedSeries(name=name, unit="USD", date_ord=dates, values=values)
        for name, dates, values in zip(names, raw_dates, raw_values, strict=True)
    ]
    levels = align_levels(series, start_ord=0, end_ord=n)
    levels_before = levels.levels.copy()
    dates_before = levels.date_ord.copy()

    unit_root_test(levels, _spec(names=list(names), weights=[1.0, -0.5], scale=scale))

    for date_original, date_kept in zip(raw_dates, date_copies, strict=True):
        np.testing.assert_array_equal(date_original, date_kept, strict=True)
    for value_original, value_kept in zip(raw_values, value_copies, strict=True):
        np.testing.assert_array_equal(value_original, value_kept, strict=True)
    np.testing.assert_array_equal(levels.levels, levels_before, strict=True)
    np.testing.assert_array_equal(levels.date_ord, dates_before, strict=True)


# ================================================================ S4.T5
# LOG level 0.0 on a named date -> ScaleDomainError.


def test_t5_log_zero_level() -> None:
    n = 40
    vals = np.ones(n, dtype=np.float64) * 2.0
    vals[7] = 0.0
    series = ObservedSeries(
        name="P",
        unit="EUR",
        date_ord=_dates(n),
        values=vals,
    )
    levels = align_levels([series], start_ord=0, end_ord=n)
    spec = _spec(scale=Scale.LOG)
    with pytest.raises(ScaleDomainError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert err.series == "P"
    assert err.date_ord == 7  # date ordinal = 7 (date at aligned index 7)
    assert err.level == 0.0
    assert err.__cause__ is not None


def test_t5_scale_domain_error_subclasses_riskquant_error() -> None:
    assert issubclass(ScaleDomainError, RiskQuantError)


# ================================================================ S4.T6
# 21 aligned dates lag 2 -> InsufficientObservationsError n=18.


def test_t6_insufficient_observations() -> None:
    n = 21
    levels = _aligned(n=n)
    spec = _spec(lag=2)
    with pytest.raises(InsufficientObservationsError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert err.n_available == 18
    assert err.n_required == 20
    assert err.reason == "SAMPLE_ROWS_BELOW_TABLE_MINIMUM"


# ================================================================ S4.T7
# Digests in two fresh processes equal.


# The child gets the parent's sys.path through the script text, as in
# test_t7_cross_process_digests_equal of test_digest.py.  The inputs equal
# those of _t7_result.
_SUBPROCESS_SCRIPT = """
import sys
sys.path[0:0] = {paths!r}
import json
import numpy as np
from riskquant.solutions.factor_regression.api import (
    CombinationTerm,
    Deterministic,
    LinearCombination,
    Scale,
    UnitRootSpec,
    align_levels,
    unit_root_test,
)
from riskquant.solutions.factor_regression.contracts import ObservedSeries

rng = np.random.default_rng(7)
n = 50
vals = np.cumsum(rng.standard_normal(n)) + 100.0
s = ObservedSeries(
    name="P",
    unit="USD",
    date_ord=np.arange(n, dtype=np.int32),
    values=vals.astype(np.float64),
)
levels = align_levels([s], start_ord=0, end_ord=n)
spec = UnitRootSpec(
    combination=LinearCombination(
        terms=(CombinationTerm(series="P", weight=1.0),),
        scale=Scale.LEVEL,
    ),
    deterministic=Deterministic.CONSTANT,
    lag=1,
    rank_rtol=1e-10,
)
result = unit_root_test(levels, spec)
print(json.dumps({{
    "sample": result.sample_digest,
    "spec": result.spec_digest,
    "row": result.row_digest,
}}))
"""


def _t7_result() -> UnitRootResult:
    rng = np.random.default_rng(7)
    n = 50
    vals = np.cumsum(rng.standard_normal(n)) + 100.0
    s = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=np.arange(n, dtype=np.int32),
        values=vals.astype(np.float64),
    )
    levels = align_levels([s], start_ord=0, end_ord=n)
    return unit_root_test(levels, _spec(lag=1, rtol=1e-10))


def _run_child() -> dict[str, str]:
    script = _SUBPROCESS_SCRIPT.format(paths=sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    out: dict[str, str] = json.loads(result.stdout.strip())
    return out


def test_t7_cross_process_digests_equal() -> None:
    """
    Two fresh processes give the same three digests, equal to this process
    (C.9).
    """
    first = _run_child()
    second = _run_child()
    in_proc = _t7_result()
    assert first == second, "digests differ between two fresh processes"
    assert first == {
        "sample": in_proc.sample_digest,
        "spec": in_proc.spec_digest,
        "row": in_proc.row_digest,
    }, "digests of a fresh process differ from this process"
    assert all(len(d) == 64 for d in first.values())


# ================================================================ S4.T8
# Mutation sensitivity matrix.
#
# weight: spec yes, sample yes, row no
# scale:  spec yes, sample yes, row no
# lag:    spec yes, sample yes, row yes
# case:   spec yes, sample yes, row no
# one level: spec no, sample yes, row no
# one interior date ordinal: spec no, sample yes, row yes
# rank_rtol: spec yes, sample no, row no


def _digests(levels: AlignedLevels, spec: UnitRootSpec) -> tuple[str, str, str]:
    r = unit_root_test(levels, spec)
    return r.spec_digest, r.sample_digest, r.row_digest


def _make_t8_base() -> tuple[AlignedLevels, UnitRootSpec]:
    rng = np.random.default_rng(99)
    n = 50
    vals = np.cumsum(rng.standard_normal(n)) + 80.0
    s = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=np.arange(n, dtype=np.int32),
        values=vals.astype(np.float64),
    )
    levels = align_levels([s], start_ord=0, end_ord=n)
    spec = _spec(lag=1, det=Deterministic.CONSTANT, rtol=1e-9)
    return levels, spec


def test_t8_mutation_matrix() -> None:
    levels, spec = _make_t8_base()
    base_sp, base_sam, base_row = _digests(levels, spec)

    # --- weight changed -> spec yes, sample yes, row no
    spec_w = UnitRootSpec(
        combination=LinearCombination(
            terms=(CombinationTerm(series="P", weight=2.0),),
            scale=Scale.LEVEL,
        ),
        deterministic=Deterministic.CONSTANT,
        lag=1,
        rank_rtol=1e-9,
    )
    sp, sam, row = _digests(levels, spec_w)
    assert sp != base_sp, "weight: spec unchanged (expected change)"
    assert sam != base_sam, "weight: sample unchanged (expected change)"
    assert row == base_row, "weight: row changed (expected unchanged)"

    # --- scale changed -> spec yes, sample yes, row no
    vals_pos = np.abs(levels.levels[0]) + 1.0
    s_pos = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=np.arange(50, dtype=np.int32),
        values=vals_pos.astype(np.float64),
    )
    levels_pos = align_levels([s_pos], start_ord=0, end_ord=50)
    spec_log = _spec(scale=Scale.LOG, lag=1, rtol=1e-9)
    sp_l, sam_l, row_l = _digests(levels_pos, spec_log)
    spec_lev = _spec(scale=Scale.LEVEL, lag=1, rtol=1e-9)
    sp_lv, sam_lv, row_lv = _digests(levels_pos, spec_lev)
    assert sp_l != sp_lv, "scale: spec unchanged (expected change)"
    assert sam_l != sam_lv, "scale: sample unchanged (expected change)"
    assert row_l == row_lv, "scale: row changed (expected unchanged)"

    # --- lag changed -> spec yes, sample yes, row yes
    spec_lag0 = UnitRootSpec(
        combination=spec.combination,
        deterministic=Deterministic.CONSTANT,
        lag=0,
        rank_rtol=1e-9,
    )
    sp0, sam0, row0 = _digests(levels, spec_lag0)
    assert sp0 != base_sp, "lag: spec unchanged (expected change)"
    assert sam0 != base_sam, "lag: sample unchanged (expected change)"
    assert row0 != base_row, "lag: row unchanged (expected change)"

    # --- case changed -> spec yes, sample yes, row no
    spec_ct = UnitRootSpec(
        combination=spec.combination,
        deterministic=Deterministic.CONSTANT_TREND,
        lag=1,
        rank_rtol=1e-9,
    )
    sp_ct, sam_ct, row_ct = _digests(levels, spec_ct)
    assert sp_ct != base_sp, "case: spec unchanged (expected change)"
    assert sam_ct != base_sam, "case: sample unchanged (expected change)"
    assert row_ct == base_row, "case: row changed (expected unchanged)"

    # --- one level changed -> spec no, sample yes, row no
    vals2 = levels.levels[0].copy()
    vals2[10] += 5.0
    s2 = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=np.arange(50, dtype=np.int32),
        values=vals2.astype(np.float64),
    )
    levels2 = align_levels([s2], start_ord=0, end_ord=50)
    sp2, sam2, row2 = _digests(levels2, spec)
    assert sp2 == base_sp, "level: spec changed (expected unchanged)"
    assert sam2 != base_sam, "level: sample unchanged (expected change)"
    assert row2 == base_row, "level: row changed (expected unchanged)"

    # --- one interior date ordinal changed -> spec no, sample yes, row yes
    # Use even-spaced dates [0, 2, 4, ..., 98] within window [0, 100).
    # Change the date at position 5 from 10 to 11 (still strictly increasing).
    # Both base and mutated cases use the same window [0, 100), so the window
    # field of the spec digest does not change.
    n_t8 = 50
    dates_even = np.arange(0, 2 * n_t8, 2, dtype=np.int32)  # [0,2,4,...,98]
    dates_mutd = dates_even.copy()
    dates_mutd[5] = 11  # was 10; neighbours 8 and 12 -> still strictly increasing
    rng_t8 = np.random.default_rng(77)
    vals_t8 = np.cumsum(rng_t8.standard_normal(n_t8)) + 100.0
    s_base3 = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=dates_even.astype(np.int32),
        values=vals_t8.astype(np.float64),
    )
    s_mutd3 = ObservedSeries(
        name="P",
        unit="USD",
        date_ord=dates_mutd.astype(np.int32),
        values=vals_t8.astype(np.float64),
    )
    spec_t8 = _spec(lag=1, rtol=1e-9)
    levels_base3 = align_levels([s_base3], start_ord=0, end_ord=100)
    levels_mutd3 = align_levels([s_mutd3], start_ord=0, end_ord=100)
    sp_b3, sam_b3, row_b3 = _digests(levels_base3, spec_t8)
    sp3, sam3, row3 = _digests(levels_mutd3, spec_t8)
    assert sp3 == sp_b3, "date: spec changed (expected unchanged)"
    assert sam3 != sam_b3, "date: sample unchanged (expected change)"
    assert row3 != row_b3, "date: row unchanged (expected change)"

    # --- rank_rtol changed -> spec yes, sample no, row no
    spec_rtol = UnitRootSpec(
        combination=spec.combination,
        deterministic=Deterministic.CONSTANT,
        lag=1,
        rank_rtol=1e-6,
    )
    sp_rt, sam_rt, row_rt = _digests(levels, spec_rtol)
    assert sp_rt != base_sp, "rtol: spec unchanged (expected change)"
    assert sam_rt == base_sam, "rtol: sample changed (expected unchanged)"
    assert row_rt == base_row, "rtol: row changed (expected unchanged)"


# ================================================================ S4.T9
# Facade import scan + pyproject.toml has no statsmodels.


def test_t9_pyproject_has_no_statsmodels() -> None:
    pyproject = _REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
    text = pyproject.read_text(encoding="utf-8").lower()
    assert "statsmodels" not in text, "pyproject.toml must not contain statsmodels"


# The facade allow-list scan (D104) is S6.T14 of test_digest.py.  It runs in
# the slice.  This scan adds the statsmodels rule of O-3 for the core and
# facade packages and their tests.
_STATSMODELS_SCAN_DIRS = (
    "src/riskquant/factor_regression",
    "src/riskquant/solutions/factor_regression",
    "tests/factor_regression",
    "tests/solutions/factor_regression",
)


def _names_statsmodels(node: ast.AST) -> bool:
    """True when an Import or ImportFrom node names statsmodels in any segment."""
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""] + [alias.name for alias in node.names]
    return any("statsmodels" in name.split(".") for name in names)


@pytest.mark.parametrize(
    "source",
    [
        "import statsmodels\n",
        "import statsmodels.api as sm\n",
        "from statsmodels.tsa.stattools import adfuller\n",
        "from x import statsmodels\n",
        "def f() -> None:\n    import statsmodels\n",
    ],
)
def test_t9_statsmodels_detector_flags_every_import_form(source: str) -> None:
    assert any(_names_statsmodels(node) for node in ast.walk(ast.parse(source)))


def test_t9_no_statsmodels_import_in_slice_packages() -> None:
    violations: list[str] = []
    for rel in _STATSMODELS_SCAN_DIRS:
        directory = _REPO_ROOT / rel
        assert directory.is_dir(), f"directory not found: {directory}"
        files = sorted(directory.rglob("*.py"))
        assert files, f"no .py file under {directory}"
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                and _names_statsmodels(node)
            )
    assert violations == [], "statsmodels imports:\n" + "\n".join(violations)


# ================================================================ S4.T12
# Scale and Deterministic member values match core codes.


def test_t12_scale_values_match_codes() -> None:
    assert int(Scale.LEVEL.value) == int(codes.SCALE_LEVEL)
    assert int(Scale.LOG.value) == int(codes.SCALE_LOG)


def test_t12_deterministic_values_match_codes() -> None:
    assert int(Deterministic.CONSTANT.value) == int(codes.DET_CONSTANT)
    assert int(Deterministic.CONSTANT_TREND.value) == int(codes.DET_CONSTANT_TREND)


# ================================================================ S4.T13
# Independent oracle for unit_root_spec_digest and unit_root_sample_digest.


def _oracle_digest_array(arr: NDArray[Any]) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def _oracle_digest_named(**arrays: NDArray[Any]) -> str:
    h = hashlib.sha256()
    for name, arr in arrays.items():
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(_oracle_digest_array(arr).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _text_uint8(text: str) -> NDArray[np.uint8]:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _oracle_unit_root_spec_digest(
    *,
    engine_version: str,
    term_names: list[str],
    term_units: list[str],
    weights: NDArray[np.float64],
    scale: int,
    deterministic: int,
    lag: int,
    rank_rtol: float,
    start_ord: int,
    end_ord: int,
    critical_value_source: str,
) -> str:
    # A.3 field order exactly.
    terms_text = "\x1e".join(
        f"T\x1f{n}\x1f{u}" for n, u in zip(term_names, term_units, strict=True)
    )
    return _oracle_digest_named(
        schema_version=np.array([1], dtype=np.int64),
        engine_version=_text_uint8(engine_version),
        test_kind=np.array([int(codes.TEST_ADF)], dtype=np.int8),
        terms=_text_uint8(terms_text),
        weights=weights,
        scale=np.array([scale], dtype=np.int8),
        deterministic=np.array([deterministic], dtype=np.int8),
        lag=np.array([lag], dtype=np.int64),
        rank_rtol=np.array([rank_rtol], dtype=np.float64),
        window=np.array([start_ord, end_ord], dtype=np.int32),
        critical_value_source=_text_uint8(critical_value_source),
    )


def _oracle_unit_root_sample_digest(
    *,
    row_start_date_ord: NDArray[np.int32],
    row_end_date_ord: NDArray[np.int32],
    regressor_names: list[str],
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> str:
    # A.3 column_names grammar: Y\x1fdz, C\x1f, then X\x1f{label} for each label.
    parts = ["Y\x1fdz", "C\x1f"] + [f"X\x1f{n}" for n in regressor_names]
    col_bytes = _text_uint8("\x1e".join(parts))
    return _oracle_digest_named(
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
        column_names=col_bytes,
        y=y,
        X=x,
    )


@pytest.mark.parametrize("det", [Deterministic.CONSTANT, Deterministic.CONSTANT_TREND])
def test_t13_spec_digest_oracle(det: Deterministic) -> None:
    n = 50
    w = np.array([1.0, -0.5], dtype=np.float64)
    cv = labels.CRITICAL_VALUE_SOURCES[int(det.value)]
    actual = digest.unit_root_spec_digest(
        engine_version=ENGINE_VERSION,
        test_kind=int(codes.TEST_ADF),
        term_names=["P", "Z"],
        term_units=["EUR", "EUR"],
        weights=w,
        scale=int(codes.SCALE_LEVEL),
        deterministic=int(det.value),
        lag=2,
        rank_rtol=1e-8,
        start_ord=0,
        end_ord=n,
        critical_value_source=cv,
    )
    oracle = _oracle_unit_root_spec_digest(
        engine_version=ENGINE_VERSION,
        term_names=["P", "Z"],
        term_units=["EUR", "EUR"],
        weights=w,
        scale=int(codes.SCALE_LEVEL),
        deterministic=int(det.value),
        lag=2,
        rank_rtol=1e-8,
        start_ord=0,
        end_ord=n,
        critical_value_source=cv,
    )
    assert actual == oracle, f"spec_digest mismatch for {det.name}"


@pytest.mark.parametrize("det", [Deterministic.CONSTANT, Deterministic.CONSTANT_TREND])
def test_t13_sample_digest_oracle(det: Deterministic) -> None:
    rng = np.random.default_rng(13)
    n, k = 30, 3
    start = np.arange(0, n, dtype=np.int32)
    end = np.arange(1, n + 1, dtype=np.int32)
    y = np.asarray(rng.standard_normal(n), dtype=np.float64)
    x = np.asarray(rng.standard_normal((n, k)), dtype=np.float64)
    reg_names = ["z_lag1", "dz_lag1", "dz_lag2"]
    if det == Deterministic.CONSTANT_TREND:
        reg_names = ["trend", *reg_names]
        x = np.ascontiguousarray(np.column_stack([np.arange(n), x]), dtype=np.float64)

    actual = digest.unit_root_sample_digest(
        row_start_date_ord=start,
        row_end_date_ord=end,
        y=y,
        x=x,
        regressor_names=reg_names,
    )
    oracle = _oracle_unit_root_sample_digest(
        row_start_date_ord=start,
        row_end_date_ord=end,
        regressor_names=reg_names,
        y=y,
        x=x,
    )
    assert actual == oracle, f"sample_digest mismatch for {det.name}"


# ================================================================ S4.T14
# One case per C.10 error mapping.


def _make_linear_levels(n: int = 40) -> AlignedLevels:
    """Linear series z = 1 + 0.5*t (same for rank and zero-variance tests)."""
    t = np.arange(n, dtype=float)
    vals = 1.0 + 0.5 * t
    s = ObservedSeries(
        name="S",
        unit="USD",
        date_ord=_dates(n),
        values=vals.astype(np.float64),
    )
    return align_levels([s], start_ord=0, end_ord=n)


def test_t14_rank_deficiency() -> None:
    """CONSTANT_TREND, lag 0, linear series -> RankDeficiencyError naming ADF columns."""
    levels = _make_linear_levels()
    spec = _spec(names=["S"], det=Deterministic.CONSTANT_TREND, lag=0)
    with pytest.raises(RankDeficiencyError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert err.reason == "RANK_DEFICIENT"
    # The ADF column labels of A.3, in design order.
    assert err.column_names == ("const", "trend", "z_lag1")
    assert err.__cause__ is not None


def test_t14_zero_response_variance() -> None:
    """CONSTANT, lag 0, linear series -> DegenerateFitError ZERO_RESPONSE_VARIANCE."""
    levels = _make_linear_levels()
    spec = _spec(names=["S"], det=Deterministic.CONSTANT, lag=0)
    with pytest.raises(DegenerateFitError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert "ZERO_RESPONSE_VARIANCE" in err.reason
    assert err.__cause__ is not None


def test_t14_exact_fit() -> None:
    """CONSTANT, lag 0, z=0.5**t (40 pts) -> DegenerateFitError EXACT_FIT."""
    n = 40
    t = np.arange(n, dtype=float)
    vals = 0.5**t
    s = ObservedSeries(
        name="S",
        unit="USD",
        date_ord=_dates(n),
        values=vals.astype(np.float64),
    )
    levels = align_levels([s], start_ord=0, end_ord=n)
    spec = _spec(names=["S"], det=Deterministic.CONSTANT, lag=0)
    with pytest.raises(DegenerateFitError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert "EXACT_FIT" in err.reason
    assert err.__cause__ is not None


def test_t14_overflow() -> None:
    """Two series 1e308, weights 10/-10, LEVEL -> DegenerateFitError INPUT_NON_FINITE."""
    n = 40
    big = np.full(n, 1e308, dtype=np.float64)
    s_a = ObservedSeries(name="A", unit="USD", date_ord=_dates(n), values=big)
    s_b = ObservedSeries(name="B", unit="USD", date_ord=_dates(n), values=big)
    levels = align_levels([s_a, s_b], start_ord=0, end_ord=n)
    spec = UnitRootSpec(
        combination=LinearCombination(
            terms=(
                CombinationTerm(series="A", weight=10.0),
                CombinationTerm(series="B", weight=-10.0),
            ),
            scale=Scale.LEVEL,
        ),
        deterministic=Deterministic.CONSTANT,
        lag=0,
        rank_rtol=1e-10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(DegenerateFitError) as exc_info:
            unit_root_test(levels, spec)
    err = exc_info.value
    assert "INPUT_NON_FINITE" in err.reason
    assert err.__cause__ is not None


def test_t14_parameter_error_patched() -> None:
    """Patched adf_core raising CoreParameterError -> ModelSpecError."""
    from riskquant.factor_regression.errors import CoreParameterError

    levels = _aligned()
    spec = _spec()
    with patch("riskquant.factor_regression.adf.adf_core") as mock_adf:
        mock_adf.side_effect = CoreParameterError(
            reason=codes.PARAM_NEGATIVE_LAG, value=-1.0
        )
        with pytest.raises(ModelSpecError) as exc_info:
            unit_root_test(levels, spec)
    assert exc_info.value.__cause__ is not None


def test_t14_internal_core_input_error_propagates() -> None:
    """CoreInputError with reason INPUT_NOT_NDARRAY from patched adf_core propagates unchanged."""
    from riskquant.factor_regression.errors import CoreInputError

    levels = _aligned()
    spec = _spec()
    core_err = CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=0, positions=())
    with patch("riskquant.factor_regression.adf.adf_core") as mock_adf:
        mock_adf.side_effect = core_err
        with pytest.raises(CoreInputError) as exc_info:
            unit_root_test(levels, spec)
    assert exc_info.value is core_err


def test_t14_log_domain_scale_domain_error() -> None:
    """LOG scale with level 0.0 -> ScaleDomainError with core as __cause__ (same as T5)."""
    n = 40
    vals = np.ones(n, dtype=np.float64) * 5.0
    vals[3] = 0.0
    s = ObservedSeries(name="P", unit="EUR", date_ord=_dates(n), values=vals)
    levels = align_levels([s], start_ord=0, end_ord=n)
    spec = _spec(scale=Scale.LOG)
    with pytest.raises(ScaleDomainError) as exc_info:
        unit_root_test(levels, spec)
    assert exc_info.value.__cause__ is not None


def test_t14_size_sample_rows_below_table_minimum() -> None:
    """21 aligned dates lag 2 -> InsufficientObservationsError (same as T6)."""
    levels = _aligned(n=21)
    spec = _spec(lag=2)
    with pytest.raises(InsufficientObservationsError) as exc_info:
        unit_root_test(levels, spec)
    err = exc_info.value
    assert err.n_available == 18
    assert err.n_required == 20
    assert err.__cause__ is not None


# ================================================================ Additional tests
# (decision 10)


def test_api_does_not_import_hashing() -> None:
    src = (_FACADE_DIR / "api.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{a.name}" for a in node.names] + (
                    [node.module or ""] if node.names else []
                )
            for name in names:
                assert "hashing" not in name, (
                    f"api.py must not import hashing (line {node.lineno})"
                )


def test_type_check_order_spec_first() -> None:
    """TypeError for wrong spec type is raised before wrong levels type."""
    with pytest.raises(TypeError) as exc_info:
        unit_root_test(cast(Any, "not_levels"), cast(Any, "not_spec"))
    assert "UnitRootSpec" in str(exc_info.value)


def test_type_check_levels_second() -> None:
    """TypeError for wrong levels type with valid spec."""
    spec = _spec()
    with pytest.raises(TypeError) as exc_info:
        unit_root_test(cast(Any, "not_levels"), spec)
    assert "AlignedLevels" in str(exc_info.value)


class _PlainSubclass(np.ndarray[Any, Any]):
    """A trivial subclass of numpy.ndarray with no new behaviour."""


_ARRAY_KINDS = ("masked", "subclass")


def _as_kind(arr: NDArray[Any], kind: str) -> NDArray[Any]:
    """Return ``arr`` as a numpy.ma.MaskedArray or as a view of _PlainSubclass."""
    if kind == "masked":
        return ma.array(arr)
    return arr.view(_PlainSubclass)


def test_array_kinds_are_ndarray_subclasses() -> None:
    base = np.ones(3, dtype=np.float64)
    for kind in _ARRAY_KINDS:
        arr = _as_kind(base, kind)
        assert isinstance(arr, np.ndarray)
        assert type(arr) is not np.ndarray
        assert arr.dtype == np.float64


@pytest.mark.parametrize("kind", _ARRAY_KINDS)
@pytest.mark.parametrize(
    "arg_name", ["row_start_date_ord", "row_end_date_ord", "y", "x"]
)
def test_ndarray_subclass_raises_for_sample_digest(arg_name: str, kind: str) -> None:
    n, k = 10, 2
    kwargs: dict[str, Any] = {
        "row_start_date_ord": np.arange(n, dtype=np.int32),
        "row_end_date_ord": np.arange(1, n + 1, dtype=np.int32),
        "y": np.ones(n, dtype=np.float64),
        "x": np.ones((n, k), dtype=np.float64),
        "regressor_names": ["z_lag1", "dz_lag1"],
    }
    digest.unit_root_sample_digest(**kwargs)  # control: the plain inputs pass
    kwargs[arg_name] = _as_kind(kwargs[arg_name], kind)
    with pytest.raises(TypeError):
        digest.unit_root_sample_digest(**kwargs)


@pytest.mark.parametrize("kind", _ARRAY_KINDS)
def test_ndarray_subclass_raises_for_spec_digest_weights(kind: str) -> None:
    kwargs: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "test_kind": int(codes.TEST_ADF),
        "term_names": ["P"],
        "term_units": ["USD"],
        "weights": np.array([1.0], dtype=np.float64),
        "scale": int(codes.SCALE_LEVEL),
        "deterministic": int(codes.DET_CONSTANT),
        "lag": 0,
        "rank_rtol": 1e-10,
        "start_ord": 0,
        "end_ord": 100,
        "critical_value_source": "src",
    }
    digest.unit_root_spec_digest(**kwargs)  # control: the plain inputs pass
    kwargs["weights"] = _as_kind(kwargs["weights"], kind)
    with pytest.raises(TypeError):
        digest.unit_root_spec_digest(**kwargs)


@pytest.mark.parametrize("kind", _ARRAY_KINDS)
@pytest.mark.parametrize(
    "field_name",
    ["critical_values", "coef", "se", "row_start_date_ord", "row_end_date_ord"],
)
def test_ndarray_subclass_raises_for_unit_root_result_field(
    field_name: str, kind: str
) -> None:
    result = _run_ok()
    kwargs: dict[str, Any] = {
        f.name: getattr(result, f.name) for f in dataclasses.fields(UnitRootResult)
    }
    UnitRootResult(**kwargs)  # control: the fields of a real result pass
    kwargs[field_name] = _as_kind(np.array(kwargs[field_name]), kind)
    with pytest.raises(TypeError):
        UnitRootResult(**kwargs)


def test_two_equal_calls_give_equal_results() -> None:
    levels = _aligned()
    spec = _spec()
    r1 = unit_root_test(levels, spec)
    r2 = unit_root_test(levels, spec)
    assert r1.spec_digest == r2.spec_digest
    assert r1.sample_digest == r2.sample_digest
    assert r1.row_digest == r2.row_digest
    assert r1.statistic == r2.statistic


# ================================================================ Fix round 2
# R-001: the dz label loop is bounded. R-002, R-003: end-to-end checks of
# unit_root_test results. Expected values come from A.1 to A.4, A.8 and the
# test inputs, not from facade or digest code.

# A.2 and A.4 texts, typed from the appendix.
_TAU_C_SOURCE = "MacKinnon (2010) QED WP 1227, Table 2 tau_c, N=1, declared weights"
_TAU_CT_SOURCE = "MacKinnon (2010) QED WP 1227, Table 3 tau_ct, N=1, declared weights"
_H0_CONSTANT = (
    "H0: the combination has a unit root (gamma = 0) and no drift. "
    "H1: it is stationary around a constant."
)
_H0_CONSTANT_TREND = (
    "H0: the combination has a unit root (gamma = 0). "
    "H1: it is stationary around a linear trend."
)

# A.8 code values, typed from the appendix.
_A8_SCALE_LEVEL = 0
_A8_SCALE_LOG = 1
_A8_DET_CONSTANT = 0
_A8_DET_CONSTANT_TREND = 1
_A8_TEST_ADF = 0

_A8_DET_CODE = {
    Deterministic.CONSTANT: _A8_DET_CONSTANT,
    Deterministic.CONSTANT_TREND: _A8_DET_CONSTANT_TREND,
}
_SOURCE_TEXT = {
    Deterministic.CONSTANT: _TAU_C_SOURCE,
    Deterministic.CONSTANT_TREND: _TAU_CT_SOURCE,
}
_H0_TEXT = {
    Deterministic.CONSTANT: _H0_CONSTANT,
    Deterministic.CONSTANT_TREND: _H0_CONSTANT_TREND,
}

_CHILD_TIMEOUT_S = 20
_CHILD_ADDRESS_SPACE_CAP = 2 * 1024**3

# The child gets the parent's sys.path through the script text, as in
# test_t7_cross_process_digests_equal of test_digest.py.  The address-space
# cap is set after the imports, so that it bounds the unit_root_test call
# only.  A regression to an unbounded label loop then fails with MemoryError
# (POSIX) or with the parent timeout (every platform), and does not hang.
_HUGE_LAG_SCRIPT = """
import sys
sys.path[0:0] = {paths!r}
import json
import numpy as np
from riskquant.solutions.factor_regression.api import (
    CombinationTerm,
    Deterministic,
    InsufficientObservationsError,
    LinearCombination,
    Scale,
    UnitRootSpec,
    align_levels,
    unit_root_test,
)
from riskquant.solutions.factor_regression.contracts import ObservedSeries

m = 40
lag = {lag!r}
series = ObservedSeries(
    name="P",
    unit="USD",
    date_ord=np.arange(m, dtype=np.int32),
    values=np.cumsum(np.random.default_rng(3).standard_normal(m)) + 50.0,
)
levels = align_levels([series], start_ord=0, end_ord=m)
spec = UnitRootSpec(
    combination=LinearCombination(
        terms=(CombinationTerm(series="P", weight=1.0),),
        scale=Scale.LEVEL,
    ),
    deterministic=Deterministic.CONSTANT,
    lag=lag,
    rank_rtol=1e-10,
)
try:
    import resource
except ImportError:
    resource = None
if resource is not None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = {cap!r}
    if hard != resource.RLIM_INFINITY:
        cap = min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
try:
    unit_root_test(levels, spec)
except InsufficientObservationsError as err:
    cause = err.__cause__
    print(json.dumps({{
        "error": type(err).__name__,
        "reason": err.reason,
        "n_available": err.n_available,
        "n_required": err.n_required,
        "cause": None if cause is None else type(cause).__name__,
    }}))
else:
    print(json.dumps({{"error": None}}))
"""


@pytest.mark.parametrize("lag", [2**63 - 1, 10**9], ids=["lag-int64-max", "lag-1e9"])
def test_r001_huge_lag_raises_size_error_in_child(lag: int) -> None:
    """
    A lag far above m raises the C.6 size error at once (R-001, S4.T2, C.6).

    S4.T2 accepts lag ``2**63 - 1``.  The column labels must not be built for
    every lag up to the declared lag before the core size check runs.
    """
    m = 40
    script = _HUGE_LAG_SCRIPT.format(
        paths=sys.path, lag=lag, cap=_CHILD_ADDRESS_SPACE_CAP
    )
    # One BLAS thread keeps the child address space small under the cap.
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"unit_root_test with lag {lag} did not stop in {_CHILD_TIMEOUT_S} s"
        )
    assert completed.returncode == 0, completed.stderr
    out = json.loads(completed.stdout.strip().splitlines()[-1])
    assert out == {
        "error": "InsufficientObservationsError",
        "reason": "SAMPLE_ROWS_BELOW_TABLE_MINIMUM",
        "n_available": m - 1 - lag,
        "n_required": 20,
        "cause": "CoreSampleSizeError",
    }


def test_r001_column_names_lag3_constant_trend() -> None:
    result = _run_ok(n=40, lag=3, det=Deterministic.CONSTANT_TREND)
    assert result.column_names == (
        "const",
        "trend",
        "z_lag1",
        "dz_lag1",
        "dz_lag2",
        "dz_lag3",
    )


# ---- R-002, R-003 fixtures


def _step_series(
    name: str, unit: str, dates: NDArray[np.int32], seed: int
) -> ObservedSeries:
    rng = np.random.default_rng(seed)
    values = np.asarray(
        np.exp(np.cumsum(0.01 * rng.standard_normal(dates.shape[0])) + 4.0),
        dtype=np.float64,
    )
    return ObservedSeries(name=name, unit=unit, date_ord=dates, values=values)


def test_r002_row_dates_follow_a1_with_non_unit_step() -> None:
    m, lag, step, first = 40, 2, 7, 1000
    dates = np.arange(first, first + step * m, step, dtype=np.int32)
    levels = align_levels(
        [_step_series("P", "USD", dates, seed=4)],
        start_ord=first,
        end_ord=first + step * m,
    )
    result = unit_root_test(levels, _spec(lag=lag))
    t = np.arange(lag + 1, m)  # A.1: t = p+1 .. m-1
    np.testing.assert_array_equal(
        result.row_start_date_ord, dates[t - 1].astype(np.int32), strict=True
    )
    np.testing.assert_array_equal(
        result.row_end_date_ord, dates[t].astype(np.int32), strict=True
    )
    assert result.n_obs == m - 1 - lag


@pytest.mark.parametrize(
    ("lag", "expected"),
    [
        (2, ("const", "trend", "z_lag1", "dz_lag1", "dz_lag2")),
        (
            12,
            (
                "const",
                "trend",
                "z_lag1",
                "dz_lag1",
                "dz_lag2",
                "dz_lag3",
                "dz_lag4",
                "dz_lag5",
                "dz_lag6",
                "dz_lag7",
                "dz_lag8",
                "dz_lag9",
                "dz_lag10",
                "dz_lag11",
                "dz_lag12",
            ),
        ),
    ],
    ids=["lag-2", "lag-12"],
)
def test_r002_column_names_constant_trend(lag: int, expected: tuple[str, ...]) -> None:
    result = _run_ok(n=60, lag=lag, det=Deterministic.CONSTANT_TREND)
    assert result.column_names == expected


def test_r002_units_in_declared_term_order() -> None:
    n = 40
    dates = _dates(n)
    levels = align_levels(
        [
            _step_series("A", "USD", dates, seed=11),
            _step_series("B", "EUR", dates, seed=12),
            _step_series("C", "GBP", dates, seed=13),
        ],
        start_ord=0,
        end_ord=n,
    )
    result = unit_root_test(levels, _spec(names=["C", "A"], weights=[1.0, -1.0]))
    assert result.units == ("GBP", "USD")


def test_r002_window_is_the_align_levels_window() -> None:
    n = 40
    dates = _dates(n, start=10)
    levels = align_levels(
        [_step_series("P", "USD", dates, seed=5)], start_ord=0, end_ord=100
    )
    result = unit_root_test(levels, _spec())
    assert result.window == (0, 100)
    assert result.window == (levels.start_ord, levels.end_ord)


def test_r002_exclusions_are_the_alignment_exclusions() -> None:
    n = 45
    full = _dates(n)
    gap = np.delete(full, 5)
    levels = align_levels(
        [_step_series("P", "USD", full, seed=6), _step_series("Q", "USD", gap, seed=7)],
        start_ord=0,
        end_ord=n,
    )
    assert len(levels.exclusions) >= 1
    result = unit_root_test(levels, _spec(names=["P", "Q"], weights=[1.0, -1.0]))
    assert result.exclusions == levels.exclusions


def test_r002_critical_levels() -> None:
    assert _run_ok().critical_levels == (0.01, 0.05, 0.10)


def test_r002_rank_rtol_reaches_adf_core() -> None:
    from riskquant.factor_regression import adf

    real_adf_core = adf.adf_core
    with patch.object(adf, "adf_core", side_effect=real_adf_core) as recorder:
        unit_root_test(_aligned(), _spec(rtol=3e-7))
    assert recorder.call_count == 1
    passed = recorder.call_args.args[3]
    assert type(passed) is float
    assert passed == 3e-7


def test_r002_unit_root_spec_stores_float32_rank_rtol_as_float() -> None:
    spec = UnitRootSpec(
        combination=_spec().combination,
        deterministic=Deterministic.CONSTANT,
        lag=0,
        rank_rtol=cast(float, np.float32(1e-6)),
    )
    assert type(spec.rank_rtol) is float
    assert spec.rank_rtol == float(np.float32(1e-6))
    assert type(unit_root_test(_aligned(), spec).rank_rtol) is float


# ---- R-003: digest oracle built from the input levels


def _oracle_adf(
    log_levels: list[NDArray[np.float64]],
    weights: list[float],
    lag: int,
    has_trend: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """
    Rebuild the A.1 response, design and regressor labels from the levels.

    ``z`` is the left-to-right sum of ``w_i * log(v_i)`` (C.2).  The design
    columns are ones, the trend ``t - (p + 1)``, ``z_{t-1}``, then
    ``dz_{t-1} .. dz_{t-p}`` (A.1).  The labels follow A.3.
    """
    z = weights[0] * np.log(log_levels[0])
    for w, v in zip(weights[1:], log_levels[1:], strict=True):
        z = z + w * np.log(v)
    m = int(z.shape[0])
    rows = np.arange(lag + 1, m)  # t = p+1 .. m-1
    dz_at = {t: z[t] - z[t - 1] for t in range(1, m)}
    y = np.array([dz_at[int(t)] for t in rows], dtype=np.float64)
    columns: list[NDArray[np.float64]] = [np.ones(rows.shape[0], dtype=np.float64)]
    names: list[str] = []
    if has_trend:
        columns.append((rows - (lag + 1)).astype(np.float64))
        names.append("trend")
    columns.append(np.array([z[int(t) - 1] for t in rows], dtype=np.float64))
    names.append("z_lag1")
    for q in range(1, lag + 1):
        columns.append(np.array([dz_at[int(t) - q] for t in rows], dtype=np.float64))
        names.append(f"dz_lag{q}")
    x = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    return y, x, names


@pytest.mark.parametrize("det", [Deterministic.CONSTANT, Deterministic.CONSTANT_TREND])
def test_r003_result_digests_equal_the_oracle(det: Deterministic) -> None:
    m, lag, step, first = 48, 2, 3, 500
    start_ord, end_ord = 400, 700
    rank_rtol = 2e-9
    dates = np.arange(first, first + step * m, step, dtype=np.int32)
    p_series = _step_series("P", "USD", dates, seed=21)
    q_series = _step_series("Q", "EUR", dates, seed=22)
    levels = align_levels([p_series, q_series], start_ord=start_ord, end_ord=end_ord)
    weights = [-1.0, -0.75]
    spec = _spec(
        names=["Q", "P"],
        weights=weights,
        scale=Scale.LOG,
        det=det,
        lag=lag,
        rtol=rank_rtol,
    )
    result = unit_root_test(levels, spec)

    has_trend = det == Deterministic.CONSTANT_TREND
    y, x, names = _oracle_adf(
        [q_series.values, p_series.values], weights, lag, has_trend
    )
    t = np.arange(lag + 1, m)
    row_start = dates[t - 1].astype(np.int32)
    row_end = dates[t].astype(np.int32)

    expected_sample = _oracle_unit_root_sample_digest(
        row_start_date_ord=row_start,
        row_end_date_ord=row_end,
        regressor_names=names,
        y=y,
        x=x,
    )
    expected_spec = _oracle_unit_root_spec_digest_a8(
        term_names=["Q", "P"],
        term_units=["EUR", "USD"],
        weights=np.array(weights, dtype=np.float64),
        scale=_A8_SCALE_LOG,
        deterministic=_A8_DET_CODE[det],
        lag=lag,
        rank_rtol=rank_rtol,
        start_ord=start_ord,
        end_ord=end_ord,
        critical_value_source=_SOURCE_TEXT[det],
    )
    expected_row = _oracle_digest_named(
        row_start_date_ord=row_start, row_end_date_ord=row_end
    )
    assert result.sample_digest == expected_sample
    assert result.spec_digest == expected_spec
    assert result.row_digest == expected_row


def _oracle_unit_root_spec_digest_a8(
    *,
    term_names: list[str],
    term_units: list[str],
    weights: NDArray[np.float64],
    scale: int,
    deterministic: int,
    lag: int,
    rank_rtol: float,
    start_ord: int,
    end_ord: int,
    critical_value_source: str,
) -> str:
    # A.3 field order, with the A.8 TEST_ADF value typed in this file.
    terms_text = "\x1e".join(
        f"T\x1f{n}\x1f{u}" for n, u in zip(term_names, term_units, strict=True)
    )
    return _oracle_digest_named(
        schema_version=np.array([1], dtype=np.int64),
        engine_version=_text_uint8(ENGINE_VERSION),
        test_kind=np.array([_A8_TEST_ADF], dtype=np.int8),
        terms=_text_uint8(terms_text),
        weights=weights,
        scale=np.array([scale], dtype=np.int8),
        deterministic=np.array([deterministic], dtype=np.int8),
        lag=np.array([lag], dtype=np.int64),
        rank_rtol=np.array([rank_rtol], dtype=np.float64),
        window=np.array([start_ord, end_ord], dtype=np.int32),
        critical_value_source=_text_uint8(critical_value_source),
    )


# ---- R-003: fixed texts of A.2 and A.4


def test_r003_label_texts_equal_a2_and_a4() -> None:
    assert labels.CRITICAL_VALUE_SOURCES == {
        _A8_DET_CONSTANT: _TAU_C_SOURCE,
        _A8_DET_CONSTANT_TREND: _TAU_CT_SOURCE,
    }
    assert labels.NULL_HYPOTHESIS_TEXTS == {
        _A8_DET_CONSTANT: _H0_CONSTANT,
        _A8_DET_CONSTANT_TREND: _H0_CONSTANT_TREND,
    }


@pytest.mark.parametrize("det", [Deterministic.CONSTANT, Deterministic.CONSTANT_TREND])
def test_r003_result_texts_equal_a2_and_a4(det: Deterministic) -> None:
    result = _run_ok(det=det)
    assert result.critical_value_source == _SOURCE_TEXT[det]
    assert result.null_hypothesis == _H0_TEXT[det]


def test_r003_a8_codes_used_by_the_oracle() -> None:
    # The oracle types the A.8 values; this pins them to the core codes.
    assert int(codes.SCALE_LEVEL) == _A8_SCALE_LEVEL
    assert int(codes.SCALE_LOG) == _A8_SCALE_LOG
    assert int(codes.DET_CONSTANT) == _A8_DET_CONSTANT
    assert int(codes.DET_CONSTANT_TREND) == _A8_DET_CONSTANT_TREND
    assert int(codes.TEST_ADF) == _A8_TEST_ADF
