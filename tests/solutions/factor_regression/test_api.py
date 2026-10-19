# tests/solutions/factor_regression/test_api.py
"""
Tests for the S7 facade entry points (S7.T1 to S7.T12).

Test IDs
--------
S7.T1  - align_levels on three named series: exclusion records against a set oracle
S7.T2  - LOG term with a level 0.0: TransformDomainError, __cause__ is the core error
S7.T3  - r_Q with r_Z and r_S: RankDeficiencyError names the three columns
S7.T4  - equal row_digest for Z-only, S-only, Z+S; different for another step
S7.T5  - wald_test: restriction digest, copied digests, no input mutation
S7.T6  - motivating case and renamed series: bitwise equal numbers; LEVEL diagnostic
S7.T7  - api.__all__: the public surface, every name resolves
S7.T8  - a term names an absent series: ModelSpecError naming it
S7.T9  - fit_ols on the log-return case: every C.15 field, read-only arrays, digests
S7.T10 - two series with one name: AlignmentError before any core call
S7.T11 - HAC lag above R - 1: ModelSpecError
S7.T12 - intercept True and False: r_squared against exact Fraction values (OD-9)

Also: one test per core-to-facade mapping with a __cause__ check, the check
order of align_levels, fit_ols and wald_test, the masked-array and
ndarray-subclass cases of restriction and rhs, and the rebuild of the core
alignment from AlignedLevels, field by field and bit for bit.

Oracles share no code with api.py (A.7): Python sets for the exclusions,
fractions.Fraction for the OLS values, explicit numpy expressions for the
rows and the design that S7.T9 digests.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.align import CoreAlignedLevels, align_core
from riskquant.factor_regression.design import assemble_design
from riskquant.factor_regression.errors import (
    CoreAlignmentError,
    CoreDegenerateFitError,
    CoreDomainError,
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreRestrictionError,
    CoreSampleSizeError,
)
from riskquant.factor_regression.fit import fit_core
from riskquant.solutions.factor_regression import api, digest
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    AlignmentError,
    CovarianceKind,
    CovarianceSpec,
    DegenerateFitError,
    InsufficientObservationsError,
    ModelSpec,
    ModelSpecError,
    ObservedSeries,
    OlsResult,
    RankDeficiencyError,
    RestrictionError,
    Term,
    Transform,
    TransformDomainError,
    align_levels,
    fit_ols,
    wald_test,
)
from riskquant.solutions.factor_regression.contracts import ENGINE_VERSION

_BASE_ORD = 19_000
_IID = CovarianceSpec(kind=CovarianceKind.IID, lag=None)
_HC0 = CovarianceSpec(kind=CovarianceKind.HC0, lag=None)
_FACADE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "riskquant"
    / "solutions"
    / "factor_regression"
)


# ------------------------------------------------------------------ helpers


def _series(
    name: str, unit: str, dates: NDArray[np.int32], values: NDArray[np.float64]
) -> ObservedSeries:
    return ObservedSeries(name=name, unit=unit, date_ord=dates, values=values)


def _ords(*offsets: int) -> NDArray[np.int32]:
    return (_BASE_ORD + np.array(offsets, dtype=np.int64)).astype(np.int32)


def _spec(
    response: str,
    regressors: tuple[str, ...],
    *,
    transform: Transform = Transform.LOG_RETURN,
    intercept: bool = True,
    step: int = 1,
    stride: int = 1,
    covariance: CovarianceSpec = _IID,
    rank_rtol: float = 1e-10,
    condition_threshold: float | None = None,
) -> ModelSpec:
    return ModelSpec(
        response=Term(series=response, transform=transform),
        regressors=tuple(Term(series=r, transform=transform) for r in regressors),
        intercept=intercept,
        step=step,
        stride=stride,
        covariance=covariance,
        rank_rtol=rank_rtol,
        condition_threshold=condition_threshold,
    )


def _motivating_arrays(
    seed: int = 11, n: int = 280, vol: float = 0.01
) -> dict[str, tuple[NDArray[np.int32], NDArray[np.float64]]]:
    """
    Synthetic motivating case: P in EUR, Z in USD, S in EUR per USD, Q = Z*S.

    Every series has its own calendar. Q is formed on the dates common to Z
    and S. P follows Q with a small independent noise.
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n, dtype=np.int64)
    z = 1.1 * np.exp(np.cumsum(vol * rng.standard_normal(n)))
    s = 0.9 * np.exp(np.cumsum(vol * rng.standard_normal(n)))
    q = z * s
    p = 1.3 * q * np.exp(0.002 * rng.standard_normal(n))

    def keep(drop: tuple[int, ...]) -> NDArray[np.int64]:
        mask = np.ones(n, dtype=np.bool_)
        mask[list(drop)] = False
        return days[mask]

    z_days = keep((3, 40, 41, 200))
    s_days = keep((7, 41, 150))
    q_days = np.intersect1d(z_days, s_days)
    p_days = keep((0, 99, 250, 251))
    out: dict[str, tuple[NDArray[np.int32], NDArray[np.float64]]] = {}
    for name, d, v in (
        ("P", p_days, p),
        ("Z", z_days, z),
        ("S", s_days, s),
        ("Q", q_days, q),
    ):
        out[name] = ((_BASE_ORD + d).astype(np.int32), np.array(v[d], dtype=np.float64))
    return out


_UNITS = {"P": "EUR", "Z": "USD", "S": "EUR/USD", "Q": "EUR"}


def _motivating_series(
    rename: dict[str, str] | None = None,
) -> list[ObservedSeries]:
    arrays = _motivating_arrays()
    names = rename or {k: k for k in arrays}
    return [
        _series(names[k], _UNITS[k], arrays[k][0], arrays[k][1])
        for k in ("P", "Z", "S", "Q")
    ]


def _motivating_levels(rename: dict[str, str] | None = None) -> AlignedLevels:
    return align_levels(
        _motivating_series(rename),
        start_ord=_BASE_ORD,
        end_ord=_BASE_ORD + 270,
    )


def _simple_pair(
    y: list[float], x: list[float]
) -> tuple[ObservedSeries, ObservedSeries]:
    dates = _ords(*range(len(y)))
    return (
        _series("Y", "u", dates, np.array(y, dtype=np.float64)),
        _series("X", "u", dates.copy(), np.array(x, dtype=np.float64)),
    )


def _bits(a: NDArray[np.generic], b: NDArray[np.generic]) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes(order="C") == b.tobytes(order="C")
    )


# ================================================================ S7.T1


def test_t1_exclusions_carry_names_dates_and_reasons() -> None:
    a = _ords(0, 1, 2, 3, 5, 6, 9)
    b = _ords(1, 2, 3, 4, 5, 6, 7, 8)
    c = _ords(-2, 1, 3, 5, 6, 8)
    inputs = {"alpha": a, "beta": b, "gamma": c}
    series = [
        _series(name, "u", d, np.arange(d.shape[0], dtype=np.float64) + 1.0)
        for name, d in inputs.items()
    ]
    start, end = _BASE_ORD + 1, _BASE_ORD + 8
    levels = align_levels(series, start_ord=start, end_ord=end)

    sets = {name: {int(x) for x in d} for name, d in inputs.items()}
    window = set(range(start, end))
    common = set.intersection(*(s & window for s in sets.values()))
    assert levels.date_ord.tolist() == sorted(common)
    oracle = set()
    for name, dates in sets.items():
        for d in dates:
            if d not in window:
                oracle.add((name, d, "OUTSIDE_WINDOW"))
            elif d not in common:
                oracle.add((name, d, "NOT_COMMON_DATE"))
    records = [(e.series, e.date_ord, e.reason) for e in levels.exclusions]
    assert len(records) == len(set(records))
    assert set(records) == oracle

    core = align_core(
        [s.date_ord for s in series],
        [s.values for s in series],
        start_ord=start,
        end_ord=end,
    )
    assert len(levels.exclusions) == int(core.excluded_series.shape[0])
    for name, dates in sets.items():
        excluded = sum(1 for e in levels.exclusions if e.series == name)
        assert excluded + levels.date_ord.shape[0] == len(dates)
    assert levels.names == ("alpha", "beta", "gamma")
    assert levels.units == ("u", "u", "u")
    assert (levels.start_ord, levels.end_ord) == (start, end)


def test_t1_levels_are_input_values_bit_for_bit() -> None:
    series = _motivating_series()
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + 270)
    for j, s in enumerate(series):
        position = np.searchsorted(s.date_ord, levels.date_ord)
        assert _bits(levels.levels[j], s.values[position])
    assert not levels.levels.flags.writeable
    assert not levels.date_ord.flags.writeable


# ================================================================ S7.T2


def test_t2_log_level_zero_names_series_transform_date_and_level() -> None:
    dates = _ords(0, 1, 2, 3, 4, 5)
    y = _series("Y", "u", dates, np.array([1.0, 1.1, 0.0, 1.2, 1.3, 1.25]))
    x = _series("X", "u", dates.copy(), np.array([2.0, 2.1, 2.2, 2.0, 2.3, 2.4]))
    levels = align_levels([y, x], start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10)
    with pytest.raises(TransformDomainError) as info:
        fit_ols(levels, _spec("Y", ("X",)))
    error = info.value
    assert error.series == "Y"
    assert error.transform == "LOG_RETURN"
    assert error.date_ord == _BASE_ORD + 2
    assert error.level == 0.0
    assert isinstance(error.__cause__, CoreDomainError)
    assert error.__cause__.aligned_index == 2
    assert "'Y'" in str(error) and "LOG_RETURN" in str(error)


def test_t2_simple_return_zero_level_in_a_regressor() -> None:
    dates = _ords(0, 1, 2, 3, 4, 5)
    y = _series("Y", "u", dates, np.array([1.0, 1.1, 1.2, 1.2, 1.3, 1.25]))
    x = _series("X", "u", dates.copy(), np.array([2.0, 2.1, 2.2, 2.0, 0.0, 2.4]))
    levels = align_levels([y, x], start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10)
    with pytest.raises(TransformDomainError) as info:
        fit_ols(levels, _spec("Y", ("X",), transform=Transform.SIMPLE_RETURN))
    assert info.value.series == "X"
    assert info.value.transform == "SIMPLE_RETURN"
    assert info.value.date_ord == _BASE_ORD + 4
    assert isinstance(info.value.__cause__, CoreDomainError)


# ================================================================ S7.T3


def test_t3_rank_deficiency_names_the_three_columns() -> None:
    levels = _motivating_levels()
    with pytest.raises(RankDeficiencyError) as info:
        fit_ols(levels, _spec("P", ("Q", "Z", "S")))
    error = info.value
    assert set(error.column_names) == {"Q", "Z", "S"}
    assert error.reason == "RANK_DEFICIENT"
    assert error.n_columns == 4
    assert error.rank == 3
    assert isinstance(error.__cause__, CoreRankError)
    assert error.__cause__.involved.tolist() == [1, 2, 3]


def test_t3_zero_column_names_the_column_and_states_no_rank() -> None:
    dates = _ords(*range(8))
    y = _series("Y", "u", dates, np.array([1.0, 1.4, 0.9, 1.3, 1.1, 1.6, 1.2, 1.5]))
    flat = _series("FLAT", "u", dates.copy(), np.full(8, 3.0))
    x = _series(
        "X", "u", dates.copy(), np.array([2.0, 2.2, 2.1, 2.5, 2.4, 2.2, 2.9, 2.6])
    )
    levels = align_levels([y, flat, x], start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10)
    with pytest.raises(RankDeficiencyError) as info:
        fit_ols(levels, _spec("Y", ("X", "FLAT"), transform=Transform.DIFFERENCE))
    error = info.value
    assert error.reason == "RANK_ZERO_COLUMN"
    assert error.column_names == ("FLAT",)
    assert "FLAT" in str(error)
    assert "rank" not in str(error).replace("RANK_ZERO_COLUMN", "")
    assert isinstance(error.__cause__, CoreRankError)
    assert error.__cause__.reason == int(codes.RANK_ZERO_COLUMN)


# ================================================================ S7.T4


def test_t4_row_digest_shared_by_models_on_one_alignment() -> None:
    levels = _motivating_levels()
    z_only = fit_ols(levels, _spec("P", ("Z",)))
    s_only = fit_ols(levels, _spec("P", ("S",)))
    both = fit_ols(levels, _spec("P", ("Z", "S")))
    other_step = fit_ols(levels, _spec("P", ("Z", "S"), step=2))
    assert z_only.row_digest == s_only.row_digest == both.row_digest
    assert other_step.row_digest != both.row_digest
    assert z_only.response == both.response


# ================================================================ S7.T5


def test_t5_wald_binds_and_does_not_mutate() -> None:
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    restriction = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rhs = np.array([1.0, 1.0])
    restriction_before = restriction.copy()
    rhs_before = rhs.copy()

    first = wald_test(result, restriction, rhs)
    assert _bits(restriction, restriction_before)
    assert _bits(rhs, rhs_before)
    assert restriction.flags.writeable and rhs.flags.writeable

    second = wald_test(result, restriction, np.array([1.0, 0.9]))
    assert first.restriction_digest != second.restriction_digest
    assert first.w != second.w
    for wald in (first, second):
        assert wald.sample_digest == result.sample_digest
        assert wald.spec_digest == result.spec_digest
        assert wald.row_digest == result.row_digest
        assert wald.engine_version == ENGINE_VERSION
        assert wald.n_restrictions == 2
        assert wald.dof == result.dof
        assert wald.f == wald.w / 2
    assert first.restriction_digest == digest.restriction_digest(restriction, rhs)


def test_t5_wald_uses_the_result_covariance() -> None:
    # Oracle: one restriction beta_j = 0 gives W = t_j ** 2 (S5.T3), with the
    # HC0 covariance of the result.
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S"), covariance=_HC0))
    wald = wald_test(result, np.array([[0.0, 0.0, 1.0]]), np.array([0.0]))
    assert wald.w == pytest.approx(float(result.t_stats[2]) ** 2, rel=1e-12)


# ================================================================ S7.T6


def test_t6_renamed_series_give_bitwise_equal_numbers() -> None:
    rename = {"P": "A", "Z": "B", "S": "C", "Q": "D"}
    levels = _motivating_levels()
    renamed = _motivating_levels(rename)
    pairs = [
        (_spec("P", ("Z", "S")), _spec("A", ("B", "C"))),
        (
            _spec("P", ("Q",), transform=Transform.LEVEL, step=0),
            _spec("A", ("D",), transform=Transform.LEVEL, step=0),
        ),
    ]
    for original_spec, renamed_spec in pairs:
        one = fit_ols(levels, original_spec)
        two = fit_ols(renamed, renamed_spec)
        for name in (
            "coef",
            "se",
            "t_stats",
            "p_values",
            "cov",
            "resid",
            "fitted",
            "singular_values",
            "row_start_date_ord",
            "row_end_date_ord",
        ):
            assert _bits(getattr(one, name), getattr(two, name)), name
        for name in (
            "dof",
            "n_obs",
            "rss",
            "r_squared",
            "durbin_watson",
            "condition_number",
        ):
            assert getattr(one, name) == getattr(two, name), name
        assert [(d.code, d.value) for d in one.diagnostics] == [
            (d.code, d.value) for d in two.diagnostics
        ]
        assert one.row_digest == two.row_digest
        assert one.sample_digest != two.sample_digest


def test_t6_level_fit_carries_level_diagnostic() -> None:
    result = fit_ols(
        _motivating_levels(), _spec("P", ("Q",), transform=Transform.LEVEL, step=0)
    )
    found = [d for d in result.diagnostics if d.code == "LEVEL_TERM_PRESENT"]
    assert len(found) == 1
    assert found[0].value == 1.0
    assert found[0].message
    log_fit = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    assert all(d.code != "LEVEL_TERM_PRESENT" for d in log_fit.diagnostics)


def test_t6_intercept_only_log_ratio_built_by_the_caller() -> None:
    levels = _motivating_levels()
    p = levels.levels[levels.names.index("P")]
    q = levels.levels[levels.names.index("Q")]
    ratio = _series("LN_P_Q", "1", levels.date_ord.copy(), np.log(p / q))
    aligned = align_levels([ratio], start_ord=levels.start_ord, end_ord=levels.end_ord)
    result = fit_ols(aligned, _spec("LN_P_Q", (), transform=Transform.LEVEL, step=0))
    assert result.column_names == ("const",)
    assert result.coef.shape == (1,)
    assert result.coef[0] == pytest.approx(float(np.mean(np.log(p / q))), rel=1e-12)


# ================================================================ S7.T7


_EXPECTED_PUBLIC = {
    "align_levels",
    "fit_ols",
    "wald_test",
    "ObservedSeries",
    "AlignedLevels",
    "Exclusion",
    "Term",
    "Transform",
    "CovarianceKind",
    "CovarianceSpec",
    "ModelSpec",
    "Diagnostic",
    "OlsResult",
    "WaldResult",
    "SeriesContractError",
    "AlignmentError",
    "ModelSpecError",
    "TransformDomainError",
    "RankDeficiencyError",
    "InsufficientObservationsError",
    "DegenerateFitError",
    "RestrictionError",
}


def test_t7_all_lists_the_public_surface_and_resolves() -> None:
    assert len(api.__all__) == len(set(api.__all__))
    assert set(api.__all__) == _EXPECTED_PUBLIC
    module = importlib.import_module("riskquant.solutions.factor_regression.api")
    for name in api.__all__:
        assert getattr(module, name) is not None


def test_t7_helpers_are_private_and_no_other_module_is_public() -> None:
    tree = ast.parse((_FACADE_DIR / "api.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name in _EXPECTED_PUBLIC or node.name.startswith("_"), node.name
    for path in sorted(_FACADE_DIR.glob("*.py")):
        if path.name == "api.py":
            continue
        other = ast.parse(path.read_text(encoding="utf-8"))
        for found in ast.walk(other):
            if isinstance(found, ast.Assign):
                for target in found.targets:
                    assert not (
                        isinstance(target, ast.Name) and target.id == "__all__"
                    ), path.name


def test_t7_api_does_not_import_hashing() -> None:
    tree = ast.parse((_FACADE_DIR / "api.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "run_identity" not in (node.module or "")
            assert all(alias.name != "hashing" for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all("run_identity" not in alias.name for alias in node.names)


# ================================================================ S7.T8


@pytest.mark.parametrize("missing_role", ["response", "regressor"])
def test_t8_absent_series_raises_model_spec_error(missing_role: str) -> None:
    levels = _motivating_levels()
    spec = (
        _spec("NOPE", ("Z",))
        if missing_role == "response"
        else _spec("P", ("Z", "NOPE"))
    )
    with pytest.raises(ModelSpecError) as info:
        fit_ols(levels, spec)
    assert "'NOPE'" in str(info.value)
    assert info.value.__cause__ is None


# ================================================================ S7.T9


_C15_FIELDS = (
    "coef",
    "se",
    "t_stats",
    "p_values",
    "cov",
    "resid",
    "fitted",
    "dof",
    "n_obs",
    "rss",
    "r_squared",
    "durbin_watson",
    "condition_number",
    "singular_values",
    "diagnostics",
    "column_names",
    "units",
    "transforms",
    "response",
    "window",
    "step",
    "stride",
    "overlap_order",
    "row_start_date_ord",
    "row_end_date_ord",
    "exclusions",
    "covariance",
    "rank_rtol",
    "condition_threshold",
    "sample_digest",
    "spec_digest",
    "row_digest",
    "engine_version",
)


def test_t9_result_holds_every_field_and_the_digests() -> None:
    levels = _motivating_levels()
    spec = _spec("P", ("Z", "S"), condition_threshold=50.0)
    result = fit_ols(levels, spec)

    assert tuple(f.name for f in dataclasses.fields(OlsResult)) == _C15_FIELDS
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, f.name
    assert result.column_names == ("const", "Z", "S")
    assert result.units == ("", "USD", "EUR/USD")
    assert result.transforms == (None, Transform.LOG_RETURN, Transform.LOG_RETURN)
    assert result.response == spec.response
    assert result.window == (levels.start_ord, levels.end_ord)
    assert (result.step, result.stride, result.overlap_order) == (1, 1, 0)
    assert result.exclusions == levels.exclusions
    assert result.covariance == spec.covariance
    assert result.rank_rtol == spec.rank_rtol
    assert result.condition_threshold == 50.0
    assert result.engine_version == ENGINE_VERSION
    assert result.n_obs == levels.date_ord.shape[0] - 1
    assert result.dof == result.n_obs - 3

    # Independent rows and design (A.1, C.8), then digest.py.
    p, z, s = (levels.levels[levels.names.index(n)] for n in ("P", "Z", "S"))
    y = np.log(p[1:] / p[:-1])
    x = np.empty((y.shape[0], 3), dtype=np.float64)
    x[:, 0] = 1.0
    x[:, 1] = np.log(z[1:] / z[:-1])
    x[:, 2] = np.log(s[1:] / s[:-1])
    start = levels.date_ord[:-1].copy()
    end = levels.date_ord[1:].copy()
    assert _bits(result.row_start_date_ord, start)
    assert _bits(result.row_end_date_ord, end)
    assert result.row_digest == digest.row_digest(start, end)
    assert result.sample_digest == digest.sample_digest(
        row_start_date_ord=start,
        row_end_date_ord=end,
        response_name="P",
        intercept=True,
        regressor_names=("Z", "S"),
        y=y,
        x=x,
    )
    assert result.spec_digest == digest.spec_digest(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=("Z", "S"),
        regressor_units=("USD", "EUR/USD"),
        regressor_transforms=("LOG_RETURN", "LOG_RETURN"),
        intercept=True,
        step=1,
        stride=1,
        start_ord=levels.start_ord,
        end_ord=levels.end_ord,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=50.0,
    )


def test_t9_hac_lag_and_unused_row_dates() -> None:
    levels = _motivating_levels()
    hac = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=1)
    result = fit_ols(levels, _spec("P", ("Z", "S"), step=3, stride=2, covariance=hac))
    m = levels.date_ord.shape[0]
    used = set(result.row_start_date_ord.tolist()) | set(
        result.row_end_date_ord.tolist()
    )
    unused = [d for d in levels.date_ord.tolist() if d not in used]
    # A.1 oracle: e_r = 3 + 2r, b_r = e_r - 3, R = floor((m - 4) / 2) + 1.
    n_rows = (m - 1 - 3) // 2 + 1
    used_index = {2 * r for r in range(n_rows)} | {3 + 2 * r for r in range(n_rows)}
    expected = [levels.date_ord[i] for i in range(m) if i not in used_index]
    assert unused == [int(d) for d in expected]
    assert len(unused) >= 1
    assert result.overlap_order == 1
    assert [d.code for d in result.diagnostics] == []
    assert (
        result.spec_digest
        != fit_ols(
            levels,
            _spec(
                "P",
                ("Z", "S"),
                step=3,
                stride=2,
                covariance=CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=2),
            ),
        ).spec_digest
    )


def test_t9_condition_threshold_flags_through_the_facade() -> None:
    levels = _motivating_levels()
    result = fit_ols(levels, _spec("P", ("Z", "S"), condition_threshold=1.0000001))
    flagged = [d for d in result.diagnostics if d.code == "CONDITION_ABOVE_THRESHOLD"]
    assert len(flagged) == 1
    assert flagged[0].value == result.condition_number
    unflagged = fit_ols(levels, _spec("P", ("Z", "S"), condition_threshold=None))
    assert all(d.code != "CONDITION_ABOVE_THRESHOLD" for d in unflagged.diagnostics)


# ================================================================ S7.T10


def test_t10_duplicate_name_raises_before_any_core_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_core_call(*args: object, **kwargs: object) -> CoreAlignedLevels:
        raise AssertionError("align_core must not run")

    monkeypatch.setattr(api, "align_core", no_core_call)
    dates = _ords(0, 1, 2)
    one = _series("SAME", "u", dates, np.array([1.0, 2.0, 3.0]))
    two = _series("SAME", "v", dates.copy(), np.array([4.0, 5.0, 6.0]))
    # start >= end: the core would raise ALIGN_EMPTY_WINDOW. The name check
    # runs first.
    with pytest.raises(AlignmentError) as info:
        align_levels([one, two], start_ord=_BASE_ORD + 5, end_ord=_BASE_ORD)
    assert info.value.series == "SAME"
    assert "'SAME'" in str(info.value)
    assert info.value.__cause__ is None


# ================================================================ S7.T11


def test_t11_hac_lag_above_rows_minus_one() -> None:
    dates = _ords(*range(6))
    y, x = (
        _series("Y", "u", dates, np.array([1.0, 1.2, 1.1, 1.4, 1.3, 1.5])),
        _series("X", "u", dates.copy(), np.array([2.0, 2.3, 2.1, 2.2, 2.6, 2.4])),
    )
    levels = align_levels([y, x], start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10)
    hac = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=5)
    with pytest.raises(ModelSpecError) as info:
        fit_ols(levels, _spec("Y", ("X",), covariance=hac))
    assert "PARAM_LAG_ABOVE_LIMIT" in str(info.value)
    assert isinstance(info.value.__cause__, CoreParameterError)
    assert info.value.__cause__.reason == int(codes.PARAM_LAG_ABOVE_LIMIT)


# ================================================================ S7.T12


def _exact_r_squared(y: list[float], x: list[float], *, intercept: bool) -> Fraction:
    zero = Fraction(0)
    ys = [Fraction(v) for v in y]
    xs = [Fraction(v) for v in x]
    n = len(ys)
    sx, sy = sum(xs, zero), sum(ys, zero)
    sxx = sum((a * a for a in xs), zero)
    sxy = sum((a * b for a, b in zip(xs, ys, strict=True)), zero)
    syy = sum((b * b for b in ys), zero)
    if intercept:
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        level = (sy - slope * sx) / n
        resid = [b - level - slope * a for a, b in zip(xs, ys, strict=True)]
        mean = sy / n
        base = sum(((b - mean) ** 2 for b in ys), zero)
    else:
        slope = sxy / sxx
        resid = [b - slope * a for a, b in zip(xs, ys, strict=True)]
        base = syy
    rss = sum((e * e for e in resid), zero)
    return Fraction(1) - rss / base


def test_t12_intercept_flag_selects_centred_or_uncentred_r_squared() -> None:
    y = [3.1, 2.7, 4.4, 5.2, 4.9, 6.3, 7.1]
    x = [1.0, 0.5, 1.75, 2.5, 2.0, 3.25, 3.5]
    levels = align_levels(
        list(_simple_pair(y, x)), start_ord=_BASE_ORD, end_ord=_BASE_ORD + 20
    )
    for intercept in (True, False):
        result = fit_ols(
            levels,
            _spec("Y", ("X",), transform=Transform.LEVEL, step=0, intercept=intercept),
        )
        exact = _exact_r_squared(y, x, intercept=intercept)
        error = abs(Fraction(result.r_squared) - exact)
        assert error <= Fraction(1, 10**12) * abs(exact), intercept
    centred = _exact_r_squared(y, x, intercept=True)
    uncentred = _exact_r_squared(y, x, intercept=False)
    assert abs(centred - uncentred) > Fraction(1, 10**3)


def test_t12_intercept_read_once_and_passed_to_both_core_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []
    real_assemble = assemble_design
    real_fit = fit_core

    def spy_assemble(
        regressors: NDArray[np.float64], *, intercept: bool
    ) -> NDArray[np.float64]:
        seen.append(("assemble_design", intercept))
        return real_assemble(regressors, intercept=intercept)

    def spy_fit(*args: object, **kwargs: object) -> object:
        seen.append(("fit_core", kwargs["intercept"]))
        return real_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api, "assemble_design", spy_assemble)
    monkeypatch.setattr(api, "fit_core", spy_fit)
    for intercept in (True, False):
        seen.clear()
        fit_ols(_motivating_levels(), _spec("P", ("Z", "S"), intercept=intercept))
        assert seen == [("assemble_design", intercept), ("fit_core", intercept)]


# ======================================== core-to-facade mapping (decision 5)


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        (_BASE_ORD + 5, _BASE_ORD + 5, "ALIGN_EMPTY_WINDOW"),
        (_BASE_ORD, 2**31, "ALIGN_WINDOW_OUT_OF_RANGE"),
        (_BASE_ORD + 2, _BASE_ORD + 3, "ALIGN_EMPTY_INTERSECTION"),
    ],
)
def test_map_alignment_error(start: int, end: int, reason: str) -> None:
    a = _series("A", "u", _ords(0, 1, 3), np.array([1.0, 2.0, 3.0]))
    b = _series("B", "u", _ords(0, 1, 4), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(AlignmentError) as info:
        align_levels([a, b], start_ord=start, end_ord=end)
    assert info.value.reason == reason
    assert info.value.series is None
    assert isinstance(info.value.__cause__, CoreAlignmentError)


def test_map_sample_size_step_not_below_count() -> None:
    levels = align_levels(
        list(_simple_pair([1.0, 2.0, 3.5], [1.0, 2.5, 2.0])),
        start_ord=_BASE_ORD,
        end_ord=_BASE_ORD + 10,
    )
    with pytest.raises(InsufficientObservationsError) as info:
        fit_ols(levels, _spec("Y", ("X",), step=3))
    assert info.value.reason == "SAMPLE_STEP_NOT_BELOW_COUNT"
    assert (info.value.n_available, info.value.n_required) == (3, 4)
    assert isinstance(info.value.__cause__, CoreSampleSizeError)


def test_map_sample_size_rows_not_above_columns_names_k() -> None:
    levels = align_levels(
        list(_simple_pair([1.0, 2.0, 3.5], [1.0, 2.5, 2.0])),
        start_ord=_BASE_ORD,
        end_ord=_BASE_ORD + 10,
    )
    with pytest.raises(InsufficientObservationsError) as info:
        fit_ols(levels, _spec("Y", ("X",)))
    error = info.value
    assert error.reason == "SAMPLE_ROWS_NOT_ABOVE_COLUMNS"
    assert (error.n_available, error.n_required) == (2, 3)
    assert "R=2" in str(error) and "k=2" in str(error)
    assert isinstance(error.__cause__, CoreSampleSizeError)


@pytest.mark.parametrize(
    ("y", "intercept", "reason"),
    [
        ([3.0, 5.0, 9.0, 7.0, 11.0], True, "DEGENERATE_EXACT_FIT"),
        ([4.0, 4.0, 4.0, 4.0, 4.0], True, "DEGENERATE_ZERO_RESPONSE_VARIANCE"),
    ],
)
def test_map_degenerate_fit(y: list[float], intercept: bool, reason: str) -> None:
    x = [1.0, 2.0, 4.0, 3.0, 5.0]
    levels = align_levels(
        list(_simple_pair(y, x)), start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10
    )
    with pytest.raises(DegenerateFitError) as info:
        fit_ols(
            levels,
            _spec("Y", ("X",), transform=Transform.LEVEL, step=0, intercept=intercept),
        )
    assert info.value.reason == reason
    assert info.value.column_names == ()
    assert isinstance(info.value.__cause__, CoreDegenerateFitError)


def test_map_degenerate_fit_columns_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    core_error = CoreDegenerateFitError(
        reason=codes.DEGENERATE_NON_POSITIVE_VARIANCE,
        value=-1.0,
        bound=0.0,
        columns=(0, 2),
    )

    def raise_core(*args: object, **kwargs: object) -> object:
        raise core_error

    monkeypatch.setattr(api, "fit_core", raise_core)
    with pytest.raises(DegenerateFitError) as info:
        fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    assert info.value.reason == "DEGENERATE_NON_POSITIVE_VARIANCE"
    assert info.value.column_names == ("const", "S")
    assert info.value.__cause__ is core_error


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("overflow_in", ["Y", "X"])
def test_map_non_finite_transform_to_degenerate_fit(overflow_in: str) -> None:
    dates = _ords(*range(6))
    normal = np.array([1.0, 1.2, 1.1, 1.4, 1.3, 1.5])
    extreme = np.array([1.0, 1e-300, 1e300, 1.1, 1.2, 1.3])
    other = np.array([2.0, 2.3, 2.1, 2.2, 2.6, 2.4])
    values = {"Y": normal, "X": normal * 0.5 + other}
    values[overflow_in] = extreme
    series = [
        _series("Y", "u", dates, values["Y"]),
        _series("X", "u", dates.copy(), values["X"]),
    ]
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + 10)
    with pytest.raises(DegenerateFitError) as info:
        fit_ols(levels, _spec("Y", ("X",)))
    assert info.value.reason == "INPUT_NON_FINITE"
    assert info.value.column_names == (overflow_in,)
    cause = info.value.__cause__
    assert isinstance(cause, CoreInputError)
    assert cause.reason == int(codes.INPUT_NON_FINITE)


def test_map_other_input_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_error = CoreInputError(reason=codes.INPUT_WRONG_SHAPE, operand=2, positions=())

    def raise_core(*args: object, **kwargs: object) -> object:
        raise core_error

    monkeypatch.setattr(api, "build_rows", raise_core)
    with pytest.raises(CoreInputError) as info:
        fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    assert info.value is core_error
    assert info.value.__cause__ is None


def test_map_wald_coef_input_error_propagates_unchanged() -> None:
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    broken = dataclasses.replace(result, coef=np.array([0.0, np.nan, 1.0]))
    with pytest.raises(CoreInputError) as info:
        wald_test(broken, np.array([[0.0, 1.0, 0.0]]), np.array([0.0]))
    assert info.value.operand == 0
    assert info.value.reason == int(codes.INPUT_NON_FINITE)


@pytest.mark.parametrize(
    ("restriction", "rhs", "reason"),
    [
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([1.0, 1.0]),
            "RESTRICTION_RANK_DEFICIENT",
        ),
        (np.array([[0.0, 1.0]]), np.array([1.0]), "RESTRICTION_WRONG_SHAPE"),
        (np.array([[0.0, 1.0, 0.0]]), np.array([1.0, 2.0]), "RESTRICTION_WRONG_SHAPE"),
        (np.array([[0.0, 1.0, 0.0]]), np.array([np.nan]), "RESTRICTION_NON_FINITE"),
    ],
)
def test_map_restriction_error(
    restriction: NDArray[np.float64], rhs: NDArray[np.float64], reason: str
) -> None:
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    with pytest.raises(RestrictionError) as info:
        wald_test(result, restriction, rhs)
    assert info.value.reason == reason
    assert info.value.shape == restriction.shape
    assert isinstance(info.value.__cause__, CoreRestrictionError)


class _Subclass(np.ndarray):
    pass


def _good_restriction() -> NDArray[np.float64]:
    return np.array([[0.0, 1.0, 0.0]])


def _good_rhs() -> NDArray[np.float64]:
    return np.array([1.0])


@pytest.mark.parametrize(
    ("make", "operand", "reason"),
    [
        (
            lambda: (np.ma.array(_good_restriction()), _good_rhs()),
            2,
            "INPUT_NOT_NDARRAY",
        ),
        (
            lambda: (_good_restriction(), np.ma.array(_good_rhs())),
            3,
            "INPUT_NOT_NDARRAY",
        ),
        (
            lambda: (_good_restriction().view(_Subclass), _good_rhs()),
            2,
            "INPUT_NOT_NDARRAY",
        ),
        (
            lambda: (_good_restriction(), _good_rhs().view(_Subclass)),
            3,
            "INPUT_NOT_NDARRAY",
        ),
        (
            lambda: (_good_restriction().astype(np.float32), _good_rhs()),
            2,
            "INPUT_WRONG_DTYPE",
        ),
        (
            lambda: (_good_restriction(), _good_rhs().astype(np.float32)),
            3,
            "INPUT_WRONG_DTYPE",
        ),
        (lambda: ([[0.0, 1.0, 0.0]], _good_rhs()), 2, "INPUT_NOT_NDARRAY"),
    ],
)
def test_map_restriction_input_error(
    make: Callable[[], tuple[object, object]], operand: int, reason: str
) -> None:
    # A masked array, a subclass or a float32 array reaches the core first,
    # so the restriction digest never sees it (decision 6 order).
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    restriction, rhs = make()
    with pytest.raises(RestrictionError) as info:
        wald_test(result, restriction, rhs)  # type: ignore[arg-type]
    assert info.value.reason == reason
    cause = info.value.__cause__
    assert isinstance(cause, CoreInputError)
    assert cause.operand == operand


# ================================================ check order (decisions 2, 3, 6)


def test_order_align_series_checks() -> None:
    good = _series("A", "u", _ords(0, 1), np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        align_levels(good, start_ord=_BASE_ORD, end_ord=_BASE_ORD + 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        align_levels("AB", start_ord=_BASE_ORD, end_ord=_BASE_ORD + 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        align_levels([], start_ord=True, end_ord=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"series\[1\]"):
        align_levels([good, "B"], start_ord=True, end_ord=_BASE_ORD)  # type: ignore[list-item]


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, "19000", None])
@pytest.mark.parametrize("which", ["start_ord", "end_ord"])
def test_order_align_bounds_before_names(bad: object, which: str) -> None:
    one = _series("SAME", "u", _ords(0, 1), np.array([1.0, 2.0]))
    two = _series("SAME", "u", _ords(0, 1), np.array([1.0, 2.0]))
    bounds: dict[str, object] = {"start_ord": _BASE_ORD, "end_ord": _BASE_ORD + 2}
    bounds[which] = bad
    with pytest.raises(TypeError, match=which):
        align_levels([one, two], **bounds)  # type: ignore[arg-type]


def test_order_align_numpy_integer_bounds_accepted() -> None:
    a = _series("A", "u", _ords(0, 1, 2), np.array([1.0, 2.0, 3.0]))
    levels = align_levels(
        [a],
        start_ord=np.int64(_BASE_ORD),  # type: ignore[arg-type]
        end_ord=np.int32(_BASE_ORD + 2),  # type: ignore[arg-type]
    )
    assert levels.date_ord.tolist() == [_BASE_ORD, _BASE_ORD + 1]
    assert type(levels.start_ord) is int and type(levels.end_ord) is int


def test_order_align_names_before_core_window_check() -> None:
    one = _series("SAME", "u", _ords(0, 1), np.array([1.0, 2.0]))
    two = _series("SAME", "u", _ords(5, 6), np.array([1.0, 2.0]))
    # Empty intersection and out-of-range window: the duplicate wins.
    with pytest.raises(AlignmentError) as info:
        align_levels([one, two], start_ord=_BASE_ORD, end_ord=2**40)
    assert info.value.series == "SAME"


def test_order_fit_spec_type_first() -> None:
    with pytest.raises(TypeError, match="spec must be ModelSpec"):
        fit_ols("not levels", "not a spec")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="levels must be AlignedLevels"):
        fit_ols("not levels", _spec("P", ("Z",)))  # type: ignore[arg-type]


def test_order_fit_missing_series_before_any_core_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_core_call(*args: object, **kwargs: object) -> object:
        raise AssertionError("build_rows must not run")

    monkeypatch.setattr(api, "build_rows", no_core_call)
    hac = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=10**6)
    with pytest.raises(ModelSpecError, match="'NOPE'"):
        fit_ols(_motivating_levels(), _spec("P", ("NOPE",), covariance=hac, step=10**6))


def test_order_wald_result_type_first() -> None:
    with pytest.raises(TypeError, match="result must be OlsResult"):
        wald_test("x", "y", "z")  # type: ignore[arg-type]


def test_order_wald_core_before_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = digest.restriction_digest

    def spy(restriction: NDArray[np.float64], rhs: NDArray[np.float64]) -> str:
        calls.append("digest")
        return real(restriction, rhs)

    monkeypatch.setattr(digest, "restriction_digest", spy)
    result = fit_ols(_motivating_levels(), _spec("P", ("Z", "S")))
    with pytest.raises(RestrictionError):
        wald_test(result, np.array([[0.0, 1.0]]), np.array([0.0]))
    assert calls == []
    wald_test(result, _good_restriction(), _good_rhs())
    assert calls == ["digest"]


# ========================================== rebuild of the core alignment (4)


def _assert_same_core(rebuilt: CoreAlignedLevels, core: CoreAlignedLevels) -> None:
    assert (rebuilt.start_ord, rebuilt.end_ord) == (core.start_ord, core.end_ord)
    for name in (
        "date_ord",
        "positions",
        "levels",
        "excluded_series",
        "excluded_date_ord",
        "excluded_reason",
    ):
        mine, theirs = getattr(rebuilt, name), getattr(core, name)
        assert _bits(mine, theirs), name
        assert mine.flags.c_contiguous and not mine.flags.writeable, name


@pytest.mark.parametrize("seed", range(40))
def test_rebuild_core_alignment_matches_align_core(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_series = int(rng.integers(1, 5))
    series: list[ObservedSeries] = []
    for j in range(n_series):
        span = int(rng.integers(5, 60))
        keep = rng.random(span) < 0.8
        keep[span // 2] = True
        offsets = np.flatnonzero(keep) + int(rng.integers(-5, 5))
        values = rng.standard_normal(offsets.shape[0])
        series.append(_series(f"s{j}", "u", _ords(*offsets.tolist()), values))
    lo = _BASE_ORD + int(rng.integers(-3, 10))
    hi = lo + int(rng.integers(5, 50))
    try:
        core = align_core(
            [s.date_ord for s in series],
            [s.values for s in series],
            start_ord=lo,
            end_ord=hi,
        )
    except CoreAlignmentError:
        with pytest.raises(AlignmentError):
            align_levels(series, start_ord=lo, end_ord=hi)
        return
    levels = align_levels(series, start_ord=lo, end_ord=hi)
    _assert_same_core(api._core_aligned_levels(levels), core)


def test_rebuild_with_no_exclusion() -> None:
    a = _series("A", "u", _ords(0, 1, 2), np.array([1.0, 2.0, 3.0]))
    b = _series("B", "u", _ords(0, 1, 2), np.array([4.0, 5.0, 6.0]))
    core = align_core(
        [a.date_ord, b.date_ord],
        [a.values, b.values],
        start_ord=_BASE_ORD,
        end_ord=_BASE_ORD + 3,
    )
    levels = align_levels([a, b], start_ord=_BASE_ORD, end_ord=_BASE_ORD + 3)
    assert levels.exclusions == ()
    _assert_same_core(api._core_aligned_levels(levels), core)


def test_fit_does_not_change_the_aligned_levels() -> None:
    levels = _motivating_levels()
    before = levels.levels.copy()
    dates_before = levels.date_ord.copy()
    fit_ols(levels, _spec("P", ("Z", "S")))
    assert _bits(levels.levels, before)
    assert _bits(levels.date_ord, dates_before)


def test_rank_error_is_not_raised_when_exact_identity_absent() -> None:
    # Guard against a false rank error: simple returns of Q, Z, S are not an
    # exact linear identity (C.6), so the design has full rank.
    result = fit_ols(
        _motivating_levels(),
        _spec("P", ("Q", "Z", "S"), transform=Transform.SIMPLE_RETURN),
    )
    assert result.singular_values.shape == (4,)
