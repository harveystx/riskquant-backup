# tests/ingest_fixtures/factor_regression/test_canonical_factor_regression_fixture.py
"""
Canonical factor-regression fixture: structure, determinism, adapter, catalog.

Layer 1 checks input-agnostic invariants of the frames. The execution tests
run every catalog analysis through the public facade and compare it with its
recorded outcome. A change of the data, of a generator or of the library that
moves an outcome fails here by name.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from riskquant.ingest_fixtures.factor_regression import (
    analyses,
    calendars,
    collinear_drivers,
    credit_factor_loadings,
    fx_converted_cointegration,
    proxy_calibration,
    rates_spread,
)
from riskquant.ingest_fixtures.factor_regression._generators import DAILY_DT
from riskquant.ingest_fixtures.factor_regression.case import (
    FACTOR_REGRESSION_SERIES_SCHEMA,
    FactorRegressionCase,
)
from riskquant.ingest_fixtures.factor_regression.catalog import (
    CANONICAL_FACTOR_REGRESSION_CASES,
    CANONICAL_FACTOR_REGRESSION_SERIES_DF,
)
from riskquant.ingest_fixtures.factor_regression.observed import (
    aligned_levels_for_case,
    observed_series_for_case,
    observed_series_from_frame,
)
from riskquant.ingest_fixtures.factor_regression.proxy_calibration import (
    population_loadings,
)
from riskquant.solutions.factor_regression import labels
from riskquant.solutions.factor_regression.api import AlignedLevels

_CASES = tuple(CANONICAL_FACTOR_REGRESSION_CASES.values())
_BUILDERS = (
    fx_converted_cointegration._build,
    proxy_calibration._build,
    credit_factor_loadings._build,
    collinear_drivers._build,
    rates_spread._build,
)


@pytest.fixture(scope="module")
def levels_by_case() -> dict[str, AlignedLevels]:
    return analyses.aligned_levels_by_case()


# ------------------------------------------------------------------ structure


def test_combined_frame_has_canonical_schema() -> None:
    assert CANONICAL_FACTOR_REGRESSION_SERIES_DF.schema == (
        FACTOR_REGRESSION_SERIES_SCHEMA
    )


def test_series_names_unique_across_cases() -> None:
    names = [s for case in _CASES for s in case.series_names]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_frame_rows_are_clean(case: FactorRegressionCase) -> None:
    frame = case.frame
    assert frame.get_column("case").unique().to_list() == [case.name]
    assert frame.null_count().sum_horizontal().item() == 0
    values = frame.get_column("value").to_numpy()
    assert bool(np.isfinite(values).all())
    dates = frame.get_column("date").cast(pl.Int32).to_numpy()
    assert int(dates.min()) >= calendars.GRID_START_ORD
    assert int(dates.max()) <= calendars.GRID_END_ORD
    # Weekday only: ordinal 0 is a Thursday.
    assert bool((((dates + 3) % 7) < 5).all())
    per_series = frame.group_by("series_name").agg(
        pl.col("unit").n_unique().alias("n_units"),
        pl.col("date").is_sorted().alias("sorted"),
        pl.col("date").n_unique().alias("n_dates"),
        pl.len().alias("n_rows"),
    )
    assert per_series.get_column("n_units").to_list() == [1] * per_series.height
    assert all(per_series.get_column("sorted").to_list())
    assert per_series.get_column("n_dates").equals(
        per_series.get_column("n_rows"), check_names=False, check_dtypes=False
    )


@pytest.mark.parametrize("builder", _BUILDERS, ids=lambda b: b.__module__)
def test_build_is_deterministic(builder: Callable[[], FactorRegressionCase]) -> None:
    first = builder()
    second = builder()
    assert first.frame.equals(second.frame)
    assert dict(first.truth) == dict(second.truth)


def test_case_truth_is_read_only() -> None:
    case = _CASES[0]
    with pytest.raises(TypeError):
        case.truth["x"] = 1.0  # type: ignore[index]


def test_every_holiday_is_absent_from_its_calendar_mask() -> None:
    for calendar, holidays in calendars.HOLIDAYS.items():
        mask = calendars.calendar_mask(calendar)
        for d in holidays:
            assert not bool(mask[calendars.grid_index_of(d)])


# ------------------------------------------------------------------ case facts


def test_eurgbp_is_the_exact_cross() -> None:
    frame = fx_converted_cointegration.FX_CONVERTED_COINTEGRATION_CASE.frame
    wide = frame.pivot(on="series_name", index="date", values="value").sort("date")
    cross = wide.get_column("USDGBP") / wide.get_column("USDEUR")
    assert cross.equals(wide.get_column("EURGBP"), check_names=False)


def test_proxy_target_starts_at_listing() -> None:
    frame = proxy_calibration.PROXY_CALIBRATION_CASE.frame
    first = (
        frame.filter(pl.col("series_name") == "SMALLCAP_EQ_EU").get_column("date").min()
    )
    assert first == proxy_calibration.TARGET_LISTING_DATE


def test_rates_swap_crosses_zero() -> None:
    frame = rates_spread.RATES_SPREAD_CASE.frame
    swap = (
        frame
        .filter(pl.col("series_name") == "CHF_SWAP_2Y")
        .get_column("value")
        .to_numpy()
    )
    assert float(swap.min()) < 0.0 < float(swap.max())


def test_alignment_shows_every_exclusion_kind() -> None:
    credit = aligned_levels_for_case(credit_factor_loadings.CREDIT_FACTOR_LOADINGS_CASE)
    reasons = {e.reason for e in credit.exclusions}
    assert reasons == {"OUTSIDE_WINDOW", "NOT_COMMON_DATE"}
    proxy = aligned_levels_for_case(proxy_calibration.PROXY_CALIBRATION_CASE)
    assert {e.reason for e in proxy.exclusions} == {"NOT_COMMON_DATE"}


# ------------------------------------------------------------------ adapter


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_observed_series_round_trip(case: FactorRegressionCase) -> None:
    observed = observed_series_for_case(case)
    assert tuple(s.name for s in observed) == case.series_names
    for s in observed:
        rows = case.frame.filter(pl.col("series_name") == s.name)
        assert np.array_equal(
            s.date_ord, rows.get_column("date").cast(pl.Int32).to_numpy()
        )
        assert np.array_equal(s.values, rows.get_column("value").to_numpy())
        assert s.date_ord.dtype == np.int32
        assert s.values.dtype == np.float64


def test_observed_series_order_follows_request() -> None:
    case = _CASES[0]
    names = tuple(reversed(case.series_names))
    observed = observed_series_from_frame(case.frame, series_names=names)
    assert tuple(s.name for s in observed) == names


def test_observed_series_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="not in the frame"):
        observed_series_from_frame(
            CANONICAL_FACTOR_REGRESSION_SERIES_DF, series_names=("NO_SUCH",)
        )


def test_observed_series_wrong_schema_raises() -> None:
    bad = CANONICAL_FACTOR_REGRESSION_SERIES_DF.with_columns(
        pl.col("value").cast(pl.Float32)
    )
    with pytest.raises(ValueError, match="schema"):
        observed_series_from_frame(bad, series_names=("USDEUR",))


# ------------------------------------------------------------------ catalog integrity


def test_analysis_names_unique() -> None:
    names = [
        *(a.name for a in analyses.CANONICAL_OLS_ANALYSES),
        *(h.name for h in analyses.CANONICAL_WALD_HYPOTHESES),
        *(a.name for a in analyses.CANONICAL_UNIT_ROOT_ANALYSES),
    ]
    assert len(names) == len(set(names))


def test_analyses_reference_known_cases_and_series() -> None:
    for a in analyses.CANONICAL_OLS_ANALYSES:
        case = CANONICAL_FACTOR_REGRESSION_CASES[a.case]
        terms = (a.spec.response, *a.spec.regressors)
        assert {t.series for t in terms} <= set(case.series_names)
        assert a.expected_diagnostics <= set(labels.DIAGNOSTIC_NAMES.values())
    for u in analyses.CANONICAL_UNIT_ROOT_ANALYSES:
        case = CANONICAL_FACTOR_REGRESSION_CASES[u.case]
        assert {t.series for t in u.spec.combination.terms} <= set(case.series_names)


def test_wald_hypotheses_target_successful_fits() -> None:
    for h in analyses.CANONICAL_WALD_HYPOTHESES:
        target = analyses.ols_analysis(h.analysis)
        assert target.expected_error is None
        n_coef = int(target.spec.intercept) + len(target.spec.regressors)
        assert h.restriction.shape == (h.rhs.shape[0], n_coef)
        assert not h.restriction.flags.writeable


def test_every_case_has_an_analysis() -> None:
    covered = {a.case for a in analyses.CANONICAL_OLS_ANALYSES}
    assert covered == set(CANONICAL_FACTOR_REGRESSION_CASES)


# ------------------------------------------------------------------ execution


@pytest.mark.parametrize(
    "analysis", analyses.CANONICAL_OLS_ANALYSES, ids=lambda a: a.name
)
def test_ols_analysis_outcome(
    analysis: analyses.OlsAnalysis, levels_by_case: dict[str, AlignedLevels]
) -> None:
    levels = levels_by_case[analysis.case]
    if analysis.expected_error is not None:
        with pytest.raises(analysis.expected_error):
            analyses.run_ols_analysis(analysis, levels)
        return
    result = analyses.run_ols_analysis(analysis, levels)
    assert {d.code for d in result.diagnostics} == analysis.expected_diagnostics


@pytest.mark.parametrize(
    "hypothesis", analyses.CANONICAL_WALD_HYPOTHESES, ids=lambda h: h.name
)
def test_wald_hypothesis_outcome(
    hypothesis: analyses.WaldHypothesis, levels_by_case: dict[str, AlignedLevels]
) -> None:
    target = analyses.ols_analysis(hypothesis.analysis)
    result = analyses.run_ols_analysis(target, levels_by_case[target.case])
    wald = analyses.run_wald_hypothesis(hypothesis, result)
    below = wald.chi2_pvalue < hypothesis.pvalue_level
    assert below is hypothesis.expect_pvalue_below, wald.chi2_pvalue


@pytest.mark.parametrize(
    "analysis", analyses.CANONICAL_UNIT_ROOT_ANALYSES, ids=lambda a: a.name
)
def test_unit_root_analysis_outcome(
    analysis: analyses.UnitRootAnalysis, levels_by_case: dict[str, AlignedLevels]
) -> None:
    levels = levels_by_case[analysis.case]
    if analysis.expected_error is not None:
        with pytest.raises(analysis.expected_error):
            analyses.run_unit_root_analysis(analysis, levels)
        return
    result = analyses.run_unit_root_analysis(analysis, levels)
    assert analyses.tau_below_level(result) == analysis.expected_tau_below_level


def test_hac_proxy_moves_toward_truth_against_daily(
    levels_by_case: dict[str, AlignedLevels],
) -> None:
    # The timing bias is the point of the case: the US loading must rise
    # toward its truth when the horizon grows from 1 to 10 days.
    truth = proxy_calibration.BETA_US
    levels = levels_by_case["proxy_calibration"]
    daily = analyses.run_ols_analysis(analyses.ols_analysis("proxy_daily"), levels)
    hac = analyses.run_ols_analysis(
        analyses.ols_analysis("proxy_10d_overlap_hac"), levels
    )
    assert float(daily.coef[1]) < truth
    assert abs(float(hac.coef[1]) - truth) < 2.0 * float(hac.se[1])


# ------------------------------------------------------------------ population loadings


# Owner cross-check: a 400000-day simulation of the documented process,
# non-overlapping rows. Each value is itself a sample estimate: with 400000 / h
# rows its Monte Carlo standard error is about 0.0027 * sqrt(h) per loading
# (0.012 at h = 20). The closed form is held to 0.01 where that error is
# small against 0.01. At h = 20 the cross-check reads (0.903, 0.283) and lies
# 0.015 (about 1.3 standard errors) below the closed form on the EU loading;
# `test_population_loadings_match_explicit_window_oracle` covers h = 20 with
# an exact oracle instead.
_OWNER_CROSS_CHECK: dict[int, tuple[float, float]] = {
    1: (1.070, 0.098),
    2: (1.013, 0.169),
    5: (0.954, 0.235),
}


def _explicit_window_loadings(h: int) -> npt.NDArray[np.float64]:
    """
    Independent oracle: the same population loadings from explicit windows.

    Stack the ``2h + 1`` session increments that the three windows touch,
    build their block-diagonal covariance from the per-session 3 x 3 block,
    and apply the window indicator matrices. No hand-counted session sums:
    the oracle shares only the documented process with the closed form.
    """
    sigma = np.array(
        [
            proxy_calibration.SIGMA_EQ_INDEX_EU,
            proxy_calibration.SIGMA_EQ_INDEX_US,
            proxy_calibration.SIGMA_TARGET_IDIO,
        ],
        dtype=np.float64,
    )
    rho = proxy_calibration.RHO_INDICES
    corr = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
    block = corr * np.outer(sigma, sigma)
    share_1 = proxy_calibration.SESSION_1_VARIANCE_SHARE
    # Session k = 0 is the session-2 increment of day t - h (US session), then
    # the sessions alternate: odd k is a session-1 (EU) increment, even k a
    # session-2 (US) increment. Session 2h is the US session of day t.
    n_sessions = 2 * h + 1
    share = np.where(np.arange(n_sessions) % 2 == 1, share_1, 1.0 - share_1)
    cov = np.kron(np.diag(share * DAILY_DT), block)
    # Window indicators over the stacked (session, factor) axis.
    target_window = np.zeros(n_sessions)
    target_window[: 2 * h] = 1.0
    us_window = np.zeros(n_sessions)
    us_window[1:] = 1.0
    e_eu = np.array([1.0, 0.0, 0.0])
    e_us = np.array([0.0, 1.0, 0.0])
    e_target = np.array([proxy_calibration.BETA_EU, proxy_calibration.BETA_US, 1.0])
    x_eu = np.kron(target_window, e_eu)
    x_us = np.kron(us_window, e_us)
    y = np.kron(target_window, e_target)
    design = np.stack([x_eu, x_us])
    moments = design @ cov @ design.T
    cross = design @ cov @ y
    out: npt.NDArray[np.float64] = np.linalg.solve(moments, cross)
    return out


@pytest.mark.parametrize("h", sorted(_OWNER_CROSS_CHECK))
def test_population_loadings_match_owner_cross_check(h: int) -> None:
    loadings = population_loadings(np.array([h], dtype=np.int32))[0]
    expected = np.array(_OWNER_CROSS_CHECK[h])
    assert np.abs(loadings - expected).max() < 0.01, (h, loadings, expected)


@pytest.mark.parametrize("h", (1, 2, 5, 20, 60))
def test_population_loadings_match_explicit_window_oracle(h: int) -> None:
    loadings = population_loadings(np.array([h], dtype=np.int32))[0]
    oracle = _explicit_window_loadings(h)
    assert np.abs(loadings - oracle).max() < 1e-12, (h, loadings, oracle)


def test_population_us_loading_increases_with_h_and_stays_below_truth() -> None:
    steps = np.arange(1, 251, dtype=np.int32)
    loadings = population_loadings(steps)
    assert loadings.shape == (250, 2)
    assert loadings.dtype == np.float64
    beta_us = loadings[:, 1]
    assert bool(np.all(np.diff(beta_us) > 0.0))
    assert float(beta_us.max()) < proxy_calibration.BETA_US


def test_population_loadings_reject_bad_steps() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        population_loadings(np.array([1, 0], dtype=np.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        population_loadings(np.array([[1, 2]], dtype=np.int32))
