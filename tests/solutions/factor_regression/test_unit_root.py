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
S4.T9  - facade import scan + pyproject.toml has no statsmodels
S4.T10 - existing golden digest tests and S7 tests pass (pytest call)
S4.T11 - _EXPECTED_PUBLIC matches api.__all__
S4.T12 - Scale and Deterministic values match core codes
S4.T13 - independent hashlib/numpy oracle for both new digests
S4.T14 - one case per C.10 mapping

Extra tests named in decision 10
---------------------------------
- api.py does not import hashing
- unit_root_test checks argument types in the declared order
- masked-array and ndarray-subclass rejection for every array arg
- caller arrays unchanged after unit_root_test
- two equal calls give equal results
"""

# mypy: ignore-errors
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import pathlib
import subprocess
import sys
import textwrap
import warnings
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from numpy import ma

from riskquant.exceptions import RiskQuantError
from riskquant.factor_regression import codes
from riskquant.solutions.factor_regression import api, digest, labels
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

_FACADE_DIR = pathlib.Path(
    importlib.util.find_spec("riskquant.solutions.factor_regression.api").origin  # type: ignore[union-attr]
).parent


# ------------------------------------------------------------------ helpers


def _dates(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int32)


def _vals(n: int, seed: int = 0) -> np.ndarray:
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
    terms = tuple(CombinationTerm(series=n, weight=w) for n, w in zip(names, weights))
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
def test_t1_term_type_error(kw: dict, exc: type) -> None:
    with pytest.raises(exc):
        CombinationTerm(**kw)  # type: ignore[arg-type]


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
def test_t1_term_model_spec_error(kw: dict, exc: type) -> None:
    with pytest.raises(exc):
        CombinationTerm(**kw)  # type: ignore[arg-type]


def test_t1_weight_int_stored_as_float() -> None:
    t = CombinationTerm(series="P", weight=2)  # type: ignore[arg-type]
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
    terms: object, scale: object, exc: type
) -> None:
    with pytest.raises(exc):
        LinearCombination(terms=terms, scale=scale)  # type: ignore[arg-type]


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
def test_t1_linear_combination_model_spec_error(terms: tuple, exc: type) -> None:
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
def test_t2_spec_type_error(kw: dict) -> None:
    with pytest.raises(TypeError):
        UnitRootSpec(**kw)  # type: ignore[arg-type]


def test_t2_spec_omitted_field_raises_type_error() -> None:
    with pytest.raises(TypeError):
        UnitRootSpec(  # type: ignore[call-arg]
            combination=_spec().combination,
            deterministic=Deterministic.CONSTANT,
            lag=0,
        )


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
def test_t2_spec_domain_error(lag: int, rtol: float, exc: type) -> None:
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


def test_t4_caller_levels_unchanged() -> None:
    levels = _aligned()
    orig_levels = levels.levels.copy()
    orig_dates = levels.date_ord.copy()
    spec = _spec()
    unit_root_test(levels, spec)
    np.testing.assert_array_equal(levels.levels, orig_levels)
    np.testing.assert_array_equal(levels.date_ord, orig_dates)


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


_PROC_SCRIPT = textwrap.dedent("""\
    import numpy as np
    import sys
    sys.path.insert(0, {src_path!r})
    from riskquant.solutions.factor_regression.api import (
        align_levels, unit_root_test,
        CombinationTerm, LinearCombination, Scale, Deterministic, UnitRootSpec,
    )
    from riskquant.solutions.factor_regression.contracts import ObservedSeries
    rng = np.random.default_rng(7)
    n = 50
    vals = np.cumsum(rng.standard_normal(n)) + 100.0
    s = ObservedSeries(name='P', unit='USD',
                       date_ord=np.arange(n, dtype=np.int32),
                       values=vals.astype(np.float64))
    levels = align_levels([s], start_ord=0, end_ord=n)
    spec = UnitRootSpec(
        combination=LinearCombination(
            terms=(CombinationTerm(series='P', weight=1.0),),
            scale=Scale.LEVEL,
        ),
        deterministic=Deterministic.CONSTANT,
        lag=1,
        rank_rtol=1e-10,
    )
    result = unit_root_test(levels, spec)
    print(result.sample_digest)
    print(result.spec_digest)
    print(result.row_digest)
""")


def _run_proc(src_path: str) -> list[str]:
    script = _PROC_SCRIPT.format(src_path=src_path)
    out = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
    )
    return out.strip().split("\n")


def test_t7_cross_process_digests_equal() -> None:
    src_path = str(pathlib.Path(__file__).parents[3] / "src")
    d1 = _run_proc(src_path)
    d2 = _run_proc(src_path)
    assert d1 == d2
    assert all(len(d) == 64 for d in d1)


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
    pyproject = pathlib.Path(__file__).parents[4] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not found")
    text = pyproject.read_text(encoding="utf-8")
    assert "statsmodels" not in text, "pyproject.toml must not contain statsmodels"


_FACADE_PACKAGE = "riskquant.solutions.factor_regression"
_HASHING_MODULE = "riskquant.solutions.run_identity.hashing"
_DIGEST_MODULE = f"{_FACADE_PACKAGE}.digest"
_FACADE_ALLOWED = (
    "numpy",
    "riskquant.factor_regression",
    "riskquant.exceptions",
    _FACADE_PACKAGE,
)


def _module_identity(module_path: str) -> tuple[str, tuple[str, ...]]:
    path = pathlib.PurePosixPath(module_path)
    assert path.parts[0] == "src" and path.suffix == ".py"
    parts = path.with_suffix("").parts[1:]
    if parts[-1] == "__init__":
        package = parts[:-1]
        return ".".join(package), package
    return ".".join(parts), parts[:-1]


def _resolved_imports(source: str, module_path: str) -> list[tuple[int, str]]:
    _, package = _module_identity(module_path)
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if node.level - 1 >= len(package):
                    found.append(
                        (node.lineno, "relative import climbs above the top package")
                    )
                    continue
                base_parts = package[: len(package) - (node.level - 1)]
                base = ".".join(
                    (*base_parts, node.module) if node.module else base_parts
                )
            else:
                base = node.module or ""
            for alias in node.names:
                name = base if alias.name == "*" else f"{base}.{alias.name}"
                found.append((node.lineno, name))
    return found


def _is_allowed(name: str, *, is_digest: bool) -> bool:
    if name.split(".")[0] in sys.stdlib_module_names:
        return True
    allowed = (*_FACADE_ALLOWED, _HASHING_MODULE) if is_digest else _FACADE_ALLOWED
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed)


def _c21_violations(source: str, module_path: str) -> list[str]:
    module_name, _ = _module_identity(module_path)
    is_digest = module_name == _DIGEST_MODULE
    return [
        f"{module_path}:{lineno}: {name}"
        for lineno, name in _resolved_imports(source, module_path)
        if not _is_allowed(name, is_digest=is_digest)
    ]


def _facade_modules() -> list[tuple[str, str]]:
    pkg = _FACADE_DIR
    src_root = pkg.parents[2]
    return [
        (
            f"src/{py_file.relative_to(src_root).as_posix()}",
            py_file.read_text(encoding="utf-8"),
        )
        for py_file in sorted(pkg.rglob("*.py"))
    ]


def test_t9_facade_imports_obey_c21() -> None:
    modules = _facade_modules()
    violations = [
        msg for path, source in modules for msg in _c21_violations(source, path)
    ]
    assert violations == [], "C.21 import violations:\n" + "\n".join(violations)


def test_t9_only_digest_imports_hashing() -> None:
    modules = _facade_modules()
    hashing_users = sorted(
        path
        for path, source in modules
        for _, name in _resolved_imports(source, path)
        if name == _HASHING_MODULE or name.startswith(f"{_HASHING_MODULE}.")
    )
    assert hashing_users == ["src/riskquant/solutions/factor_regression/digest.py"]


# ================================================================ S4.T11
# _EXPECTED_PUBLIC matches api.__all__.


def test_t11_public_surface() -> None:
    from tests.solutions.factor_regression.test_api import _EXPECTED_PUBLIC  # type: ignore[import]

    assert set(api.__all__) == _EXPECTED_PUBLIC
    module = importlib.import_module("riskquant.solutions.factor_regression.api")
    for name in api.__all__:
        assert getattr(module, name, None) is not None, f"api.{name} resolves to None"


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


def _oracle_digest_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def _oracle_digest_named(**arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, arr in arrays.items():
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(_oracle_digest_array(arr).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _text_uint8(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _oracle_unit_root_spec_digest(
    *,
    engine_version: str,
    term_names: list[str],
    term_units: list[str],
    weights: np.ndarray,
    scale: int,
    deterministic: int,
    lag: int,
    rank_rtol: float,
    start_ord: int,
    end_ord: int,
    critical_value_source: str,
) -> str:
    # A.3 field order exactly.
    terms_text = "\x1e".join(f"T\x1f{n}\x1f{u}" for n, u in zip(term_names, term_units))
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
    row_start_date_ord: np.ndarray,
    row_end_date_ord: np.ndarray,
    regressor_names: list[str],
    y: np.ndarray,
    x: np.ndarray,
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
    X = np.asarray(rng.standard_normal((n, k)), dtype=np.float64)
    reg_names = ["z_lag1", "dz_lag1", "dz_lag2"]
    if det == Deterministic.CONSTANT_TREND:
        reg_names = ["trend"] + reg_names
        X = np.ascontiguousarray(np.column_stack([np.arange(n), X]), dtype=np.float64)

    actual = digest.unit_root_sample_digest(
        row_start_date_ord=start,
        row_end_date_ord=end,
        y=y,
        x=X,
        regressor_names=reg_names,
    )
    oracle = _oracle_unit_root_sample_digest(
        row_start_date_ord=start,
        row_end_date_ord=end,
        regressor_names=reg_names,
        y=y,
        x=X,
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
    # Column names must come from ADF labels (const, trend, z_lag1)
    adf_labels = {"const", "trend", "z_lag1"}
    assert set(err.column_names).issubset(adf_labels), (
        f"Unexpected column names: {err.column_names}"
    )
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
        unit_root_test("not_levels", "not_spec")  # type: ignore[arg-type]
    assert "UnitRootSpec" in str(exc_info.value)


def test_type_check_levels_second() -> None:
    """TypeError for wrong levels type with valid spec."""
    spec = _spec()
    with pytest.raises(TypeError) as exc_info:
        unit_root_test("not_levels", spec)  # type: ignore[arg-type]
    assert "AlignedLevels" in str(exc_info.value)


@pytest.mark.parametrize(
    "arg_name",
    ["row_start_date_ord", "row_end_date_ord", "y", "x"],
)
def test_masked_array_raises_for_sample_digest(arg_name: str) -> None:
    n, k = 10, 2
    start = np.arange(n, dtype=np.int32)
    end = np.arange(1, n + 1, dtype=np.int32)
    y = np.ones(n, dtype=np.float64)
    X = np.ones((n, k), dtype=np.float64)
    kwargs: dict = {
        "row_start_date_ord": start,
        "row_end_date_ord": end,
        "y": y,
        "x": X,
        "regressor_names": ["z_lag1", "dz_lag1"],
    }
    if arg_name in ("row_start_date_ord", "row_end_date_ord"):
        kwargs[arg_name] = ma.array(start)
    elif arg_name == "y":
        kwargs["y"] = ma.array(y)
    else:
        kwargs["x"] = ma.array(X)
    with pytest.raises(TypeError):
        digest.unit_root_sample_digest(**kwargs)


def test_masked_array_raises_for_spec_digest_weights() -> None:
    w = ma.array(np.array([1.0], dtype=np.float64))
    with pytest.raises(TypeError):
        digest.unit_root_spec_digest(
            engine_version=ENGINE_VERSION,
            term_names=["P"],
            term_units=["USD"],
            weights=w,  # type: ignore[arg-type]
            scale=int(codes.SCALE_LEVEL),
            deterministic=int(codes.DET_CONSTANT),
            lag=0,
            rank_rtol=1e-10,
            start_ord=0,
            end_ord=100,
            critical_value_source="src",
        )


def test_two_equal_calls_give_equal_results() -> None:
    levels = _aligned()
    spec = _spec()
    r1 = unit_root_test(levels, spec)
    r2 = unit_root_test(levels, spec)
    assert r1.spec_digest == r2.spec_digest
    assert r1.sample_digest == r2.sample_digest
    assert r1.row_digest == r2.row_digest
    assert r1.statistic == r2.statistic
