# src/riskquant/ingest_fixtures/factor_regression/analyses.py
"""
Canonical analyses over the factor-regression cases, with recorded outcomes.

Each analysis is one declared facade call on the alignment of its whole case
(``observed.aligned_levels_for_case``). Each analysis carries the outcome the
canonical data produce. The outcomes are fixture oracles, not library
decisions: the library reports statistics and approves nothing (D104.4). The
fixture tests run every analysis and compare.

Outcome fields
--------------
``expected_error``
    The facade exception type the call raises, or ``None``.
``expected_diagnostics``
    The exact set of ``Diagnostic.code`` names of the fit.
``expected_tau_below_level``
    For a unit-root analysis: the smallest of ``(0.01, 0.05, 0.10)`` at which
    ``statistic < critical value``; ``None`` means the statistic is at or
    above the 10 % critical value.
``pvalue_level`` / ``expect_pvalue_below``
    For a Wald hypothesis: whether ``chi2_pvalue < pvalue_level``.

Declared tolerances
-------------------
``RANK_RTOL = 1e-10``. The rank gate compares singular values of the scaled
design relative to the largest one. A unit-conversion duplicate gives a
relative singular value at rounding level (about 1e-16); the near-collinear
gas hubs give about 1/35. ``1e-10`` separates the two by six orders of
magnitude on each side.

``CONDITION_THRESHOLD = 30.0``. The Belsley, Kuh and Welsch convention for
strong collinearity of a column-scaled design.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from riskquant.ingest_fixtures.factor_regression.catalog import (
    CANONICAL_FACTOR_REGRESSION_CASES,
)
from riskquant.ingest_fixtures.factor_regression.observed import (
    aligned_levels_for_case,
)
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    CombinationTerm,
    CovarianceKind,
    CovarianceSpec,
    DegenerateFitError,
    Deterministic,
    LinearCombination,
    ModelSpec,
    OlsResult,
    RankDeficiencyError,
    Scale,
    ScaleDomainError,
    Term,
    Transform,
    TransformDomainError,
    UnitRootResult,
    UnitRootSpec,
    WaldResult,
    fit_ols,
    unit_root_test,
    wald_test,
)

__all__ = [
    "CANONICAL_OLS_ANALYSES",
    "CANONICAL_UNIT_ROOT_ANALYSES",
    "CANONICAL_WALD_HYPOTHESES",
    "CONDITION_THRESHOLD",
    "CRITICAL_LEVELS",
    "RANK_RTOL",
    "OlsAnalysis",
    "UnitRootAnalysis",
    "WaldHypothesis",
    "aligned_levels_by_case",
    "ols_analysis",
    "run_ols_analysis",
    "run_unit_root_analysis",
    "run_wald_hypothesis",
    "tau_below_level",
]

RANK_RTOL: Final = 1e-10
CONDITION_THRESHOLD: Final = 30.0
# Order of UnitRootResult.critical_values (C.7).
CRITICAL_LEVELS: Final = (0.01, 0.05, 0.10)

_LEVEL_TERM_PRESENT: Final = "LEVEL_TERM_PRESENT"
_R2_ABOVE_DURBIN_WATSON: Final = "R2_ABOVE_DURBIN_WATSON"
_OVERLAP_NOT_CORRECTED: Final = "OVERLAP_NOT_CORRECTED"
_CONDITION_ABOVE_THRESHOLD: Final = "CONDITION_ABOVE_THRESHOLD"


# ------------------------------------------------------------------ contracts


@dataclass(frozen=True, slots=True, kw_only=True)
class OlsAnalysis:
    """One declared ``fit_ols`` call and its recorded outcome."""

    name: str
    case: str
    purpose: str
    spec: ModelSpec
    expected_error: type[Exception] | None
    expected_diagnostics: frozenset[str]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class WaldHypothesis:
    """
    One joint Wald hypothesis ``restriction @ coef = rhs`` on an OLS analysis.

    ``restriction`` columns follow the design order: ``const`` first when the
    spec has an intercept, then the regressors in declared order.
    ``eq=False``: array fields.
    """

    name: str
    analysis: str
    purpose: str
    restriction: npt.NDArray[np.float64]
    rhs: npt.NDArray[np.float64]
    pvalue_level: float
    expect_pvalue_below: bool

    def __post_init__(self) -> None:
        for attr in ("restriction", "rhs"):
            arr = np.array(getattr(self, attr), dtype=np.float64, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, attr, arr)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitRootAnalysis:
    """One declared ``unit_root_test`` call and its recorded outcome."""

    name: str
    case: str
    purpose: str
    spec: UnitRootSpec
    expected_error: type[Exception] | None
    expected_tau_below_level: float | None


# ------------------------------------------------------------------ builders


def _ols(
    response: tuple[str, Transform],
    regressors: Sequence[tuple[str, Transform]],
    *,
    intercept: bool,
    step: int,
    stride: int,
    covariance: CovarianceKind,
    lag: int | None = None,
    condition_threshold: float | None = None,
) -> ModelSpec:
    return ModelSpec(
        response=Term(series=response[0], transform=response[1]),
        regressors=tuple(Term(series=s, transform=t) for s, t in regressors),
        intercept=intercept,
        step=step,
        stride=stride,
        covariance=CovarianceSpec(kind=covariance, lag=lag),
        rank_rtol=RANK_RTOL,
        condition_threshold=condition_threshold,
    )


def _unit_root(
    terms: Sequence[tuple[str, float]],
    *,
    scale: Scale,
    deterministic: Deterministic,
    lag: int,
) -> UnitRootSpec:
    return UnitRootSpec(
        combination=LinearCombination(
            terms=tuple(CombinationTerm(series=s, weight=w) for s, w in terms),
            scale=scale,
        ),
        deterministic=deterministic,
        lag=lag,
        rank_rtol=RANK_RTOL,
    )


_LOG = Transform.LOG_RETURN
_LVL = Transform.LEVEL
_DIF = Transform.DIFFERENCE

_FX: Final = "fx_converted_cointegration"
_PROXY: Final = "proxy_calibration"
_CREDIT: Final = "credit_factor_loadings"
_GAS: Final = "collinear_drivers"
_RATES: Final = "rates_spread"

_CREDIT_FACTORS: Final = (
    ("CREDIT_FACTOR_GLOBAL", _LOG),
    ("CREDIT_FACTOR_EUROPE", _LOG),
)
_PROXY_DRIVERS: Final = (("EQ_INDEX_EU", _LOG), ("EQ_INDEX_US", _LOG))


def _credit_weekly(obligor: str) -> ModelSpec:
    return _ols(
        (obligor, _LOG),
        _CREDIT_FACTORS,
        intercept=True,
        step=5,
        stride=5,
        covariance=CovarianceKind.HC0,
    )


def _proxy(step: int, stride: int, kind: CovarianceKind, lag: int | None) -> ModelSpec:
    return _ols(
        ("SMALLCAP_EQ_EU", _LOG),
        _PROXY_DRIVERS,
        intercept=False,
        step=step,
        stride=stride,
        covariance=kind,
        lag=lag,
    )


# ------------------------------------------------------------------ OLS catalog

CANONICAL_OLS_ANALYSES: Final[tuple[OlsAnalysis, ...]] = (
    # ---- fx_converted_cointegration -------------------------------------
    OlsAnalysis(
        name="fx_level_peer",
        case=_FX,
        purpose=(
            "Level regression of two EUR assets that share a return driver but "
            "are not cointegrated: a high R**2 that the diagnostics flag."
        ),
        spec=_ols(
            ("XLIST_EUR", _LVL),
            (("PEER_EUR", _LVL),),
            intercept=True,
            step=0,
            stride=1,
            covariance=CovarianceKind.IID,
        ),
        expected_error=None,
        expected_diagnostics=frozenset({_LEVEL_TERM_PRESENT, _R2_ABOVE_DURBIN_WATSON}),
    ),
    OlsAnalysis(
        name="fx_return_conversion",
        case=_FX,
        purpose=(
            "Daily log returns of the EUR listing on the USD listing and on "
            "USDEUR: both slopes measure the currency conversion, truth (1, 1)."
        ),
        spec=_ols(
            ("XLIST_EUR", _LOG),
            (("XLIST_USD", _LOG), ("USDEUR", _LOG)),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.HC0,
        ),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="fx_triangle_identity",
        case=_FX,
        purpose=(
            "EURGBP on USDGBP and USDEUR is an exact cross identity: no error "
            "term exists, so the fit refuses to report standard errors."
        ),
        spec=_ols(
            ("EURGBP", _LOG),
            (("USDGBP", _LOG), ("USDEUR", _LOG)),
            intercept=False,
            step=1,
            stride=1,
            covariance=CovarianceKind.IID,
        ),
        expected_error=DegenerateFitError,
        expected_diagnostics=frozenset(),
    ),
    # ---- proxy_calibration -----------------------------------------------
    OlsAnalysis(
        name="proxy_daily",
        case=_PROXY,
        purpose=(
            "One-day Proxy V1 regression: the US close is one session later "
            "than the target close, so the US loading is biased low."
        ),
        spec=_proxy(1, 1, CovarianceKind.IID, None),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="proxy_10d_overlap_iid",
        case=_PROXY,
        purpose=(
            "Ten-day overlapping rows remove most of the timing bias, but IID "
            "errors ignore the MA(9) overlap: the diagnostic fires."
        ),
        spec=_proxy(10, 1, CovarianceKind.IID, None),
        expected_error=None,
        expected_diagnostics=frozenset({_OVERLAP_NOT_CORRECTED}),
    ),
    OlsAnalysis(
        name="proxy_10d_overlap_hac",
        case=_PROXY,
        purpose=(
            "Same rows with a Bartlett HAC covariance of lag 9 = h - 1: the "
            "recommended Proxy V1 calibration."
        ),
        spec=_proxy(10, 1, CovarianceKind.HAC_BARTLETT, 9),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="proxy_10d_non_overlap",
        case=_PROXY,
        purpose=(
            "Non-overlapping ten-day rows (stride = step): no overlap, IID is "
            "admissible, at the cost of a tenth of the rows."
        ),
        spec=_proxy(10, 10, CovarianceKind.IID, None),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    # ---- credit_factor_loadings ------------------------------------------
    OlsAnalysis(
        name="credit_weekly_eu_bank",
        case=_CREDIT,
        purpose="Weekly factor loadings of a European bank (HC0).",
        spec=_credit_weekly("OBLIGOR_EU_BANK"),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="credit_weekly_eu_utility",
        case=_CREDIT,
        purpose="Weekly factor loadings of a European utility (HC0).",
        spec=_credit_weekly("OBLIGOR_EU_UTILITY"),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="credit_weekly_us_industrial",
        case=_CREDIT,
        purpose="Weekly factor loadings of a US industrial (HC0).",
        spec=_credit_weekly("OBLIGOR_US_INDUSTRIAL"),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    # ---- collinear_drivers -----------------------------------------------
    OlsAnalysis(
        name="gas_single_hub",
        case=_GAS,
        purpose="Power on one gas hub: a well-conditioned design.",
        spec=_ols(
            ("POWER_DE_BASE", _LOG),
            (("GAS_TTF", _LOG),),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.IID,
            condition_threshold=CONDITION_THRESHOLD,
        ),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="gas_two_hubs",
        case=_GAS,
        purpose=(
            "Power on two hubs that carry one signal: the condition number "
            "exceeds 30 and the individual slopes are unstable."
        ),
        spec=_ols(
            ("POWER_DE_BASE", _LOG),
            (("GAS_TTF", _LOG), ("GAS_THE", _LOG)),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.IID,
            condition_threshold=CONDITION_THRESHOLD,
        ),
        expected_error=None,
        expected_diagnostics=frozenset({_CONDITION_ABOVE_THRESHOLD}),
    ),
    OlsAnalysis(
        name="gas_one_price_two_units",
        case=_GAS,
        purpose=(
            "The TTF price per MWh and per therm in one design: identical log "
            "returns, so the rank gate refuses the fit."
        ),
        spec=_ols(
            ("POWER_DE_BASE", _LOG),
            (("GAS_TTF", _LOG), ("GAS_TTF_THERM", _LOG)),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.IID,
        ),
        expected_error=RankDeficiencyError,
        expected_diagnostics=frozenset(),
    ),
    # ---- rates_spread ----------------------------------------------------
    OlsAnalysis(
        name="rates_govt_on_swap_difference",
        case=_RATES,
        purpose="Daily rate changes in percent: government on swap, truth 1.",
        spec=_ols(
            ("CHF_GOVT_2Y", _DIF),
            (("CHF_SWAP_2Y", _DIF),),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.HC0,
        ),
        expected_error=None,
        expected_diagnostics=frozenset(),
    ),
    OlsAnalysis(
        name="rates_log_return_through_zero",
        case=_RATES,
        purpose=(
            "Log returns of rates that cross zero are undefined: the domain "
            "check fails loud before any row exists."
        ),
        spec=_ols(
            ("CHF_GOVT_2Y", _LOG),
            (("CHF_SWAP_2Y", _LOG),),
            intercept=True,
            step=1,
            stride=1,
            covariance=CovarianceKind.HC0,
        ),
        expected_error=TransformDomainError,
        expected_diagnostics=frozenset(),
    ),
)


# ------------------------------------------------------------------ Wald catalog

_PROXY_TRUTH: Final = CANONICAL_FACTOR_REGRESSION_CASES[_PROXY].truth
_CREDIT_TRUTH: Final = CANONICAL_FACTOR_REGRESSION_CASES[_CREDIT].truth
_PROXY_TRUE_LOADINGS: Final = np.array(
    [_PROXY_TRUTH["beta_EQ_INDEX_EU"], _PROXY_TRUTH["beta_EQ_INDEX_US"]],
    dtype=np.float64,
)

_I2: Final = np.eye(2, dtype=np.float64)
_EUROPE_ONLY: Final = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
_ZERO_1: Final = np.zeros(1, dtype=np.float64)

CANONICAL_WALD_HYPOTHESES: Final[tuple[WaldHypothesis, ...]] = (
    WaldHypothesis(
        name="fx_conversion_unit_betas",
        analysis="fx_return_conversion",
        purpose="Joint H0: both conversion slopes equal 1.",
        restriction=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        rhs=np.array([1.0, 1.0]),
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="proxy_daily_truth",
        analysis="proxy_daily",
        purpose="Joint H0: daily loadings equal the truth (0.90, 0.30).",
        restriction=_I2,
        rhs=_PROXY_TRUE_LOADINGS,
        pvalue_level=0.05,
        expect_pvalue_below=True,
    ),
    WaldHypothesis(
        name="proxy_10d_overlap_iid_truth",
        analysis="proxy_10d_overlap_iid",
        purpose=(
            "Same H0 with IID errors on overlapping rows: the understated "
            "standard errors reject a true hypothesis."
        ),
        restriction=_I2,
        rhs=_PROXY_TRUE_LOADINGS,
        pvalue_level=0.05,
        expect_pvalue_below=True,
    ),
    WaldHypothesis(
        name="proxy_10d_overlap_hac_truth",
        analysis="proxy_10d_overlap_hac",
        purpose="Same H0 with HAC lag 9: the truth is inside the region.",
        restriction=_I2,
        rhs=_PROXY_TRUE_LOADINGS,
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="proxy_10d_non_overlap_truth",
        analysis="proxy_10d_non_overlap",
        purpose="Same H0 on non-overlapping rows.",
        restriction=_I2,
        rhs=_PROXY_TRUE_LOADINGS,
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="credit_eu_bank_no_europe",
        analysis="credit_weekly_eu_bank",
        purpose="H0: the European bank has no European factor loading.",
        restriction=_EUROPE_ONLY,
        rhs=_ZERO_1,
        pvalue_level=0.01,
        expect_pvalue_below=True,
    ),
    WaldHypothesis(
        name="credit_eu_bank_truth",
        analysis="credit_weekly_eu_bank",
        purpose="Joint H0: both slopes equal the generating slopes.",
        restriction=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        rhs=np.array([
            _CREDIT_TRUTH["beta_CREDIT_FACTOR_GLOBAL_OBLIGOR_EU_BANK"],
            _CREDIT_TRUTH["beta_CREDIT_FACTOR_EUROPE_OBLIGOR_EU_BANK"],
        ]),
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="credit_eu_utility_no_europe",
        analysis="credit_weekly_eu_utility",
        purpose="H0: the European utility has no European factor loading.",
        restriction=_EUROPE_ONLY,
        rhs=_ZERO_1,
        pvalue_level=0.05,
        expect_pvalue_below=True,
    ),
    WaldHypothesis(
        name="credit_us_industrial_no_europe",
        analysis="credit_weekly_us_industrial",
        purpose="H0: the US industrial has no European factor loading (true).",
        restriction=_EUROPE_ONLY,
        rhs=_ZERO_1,
        pvalue_level=0.10,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="gas_two_hubs_sum",
        analysis="gas_two_hubs",
        purpose=(
            "H0: the sum of the two hub slopes equals 0.85. The sum is well "
            "measured even when the individual slopes are not."
        ),
        restriction=np.array([[0.0, 1.0, 1.0]]),
        rhs=np.array([CANONICAL_FACTOR_REGRESSION_CASES[_GAS].truth["beta_sum_gas"]]),
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
    WaldHypothesis(
        name="rates_govt_one_for_one",
        analysis="rates_govt_on_swap_difference",
        purpose="H0: the government rate moves one for one with the swap rate.",
        restriction=np.array([[0.0, 1.0]]),
        rhs=np.array([1.0]),
        pvalue_level=0.05,
        expect_pvalue_below=False,
    ),
)


# ------------------------------------------------------------------ unit-root catalog

CANONICAL_UNIT_ROOT_ANALYSES: Final[tuple[UnitRootAnalysis, ...]] = (
    UnitRootAnalysis(
        name="fx_converted_spread",
        case=_FX,
        purpose=(
            "ln XLIST_EUR - ln XLIST_USD - ln USDEUR: the EUR listing against "
            "the USD listing converted to EUR. Declared vector, N = 1 table."
        ),
        spec=_unit_root(
            (("XLIST_EUR", 1.0), ("XLIST_USD", -1.0), ("USDEUR", -1.0)),
            scale=Scale.LOG,
            deterministic=Deterministic.CONSTANT,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=0.01,
    ),
    UnitRootAnalysis(
        name="fx_peer_spread",
        case=_FX,
        purpose=(
            "The same combination for the peer: correlated returns, no cointegration."
        ),
        spec=_unit_root(
            (("PEER_EUR", 1.0), ("XLIST_USD", -1.0), ("USDEUR", -1.0)),
            scale=Scale.LOG,
            deterministic=Deterministic.CONSTANT,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=None,
    ),
    UnitRootAnalysis(
        name="rates_swap_level",
        case=_RATES,
        purpose="The swap rate alone: a drifting random walk (control).",
        spec=_unit_root(
            (("CHF_SWAP_2Y", 1.0),),
            scale=Scale.LEVEL,
            deterministic=Deterministic.CONSTANT_TREND,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=None,
    ),
    UnitRootAnalysis(
        name="rates_swap_spread",
        case=_RATES,
        purpose="Swap minus government in percent: stationary around 0.20.",
        spec=_unit_root(
            (("CHF_SWAP_2Y", 1.0), ("CHF_GOVT_2Y", -1.0)),
            scale=Scale.LEVEL,
            deterministic=Deterministic.CONSTANT,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=0.01,
    ),
    UnitRootAnalysis(
        name="rates_swap_spread_log_scale",
        case=_RATES,
        purpose="The same spread on LOG scale: rates at or below zero raise.",
        spec=_unit_root(
            (("CHF_SWAP_2Y", 1.0), ("CHF_GOVT_2Y", -1.0)),
            scale=Scale.LOG,
            deterministic=Deterministic.CONSTANT,
            lag=1,
        ),
        expected_error=ScaleDomainError,
        expected_tau_below_level=None,
    ),
    UnitRootAnalysis(
        name="rates_covered_spread_constant",
        case=_RATES,
        purpose=(
            "Covered minus swap with a constant only: the 25 bp a year trend "
            "looks like a unit root."
        ),
        spec=_unit_root(
            (("CHF_COVERED_2Y", 1.0), ("CHF_SWAP_2Y", -1.0)),
            scale=Scale.LEVEL,
            deterministic=Deterministic.CONSTANT,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=None,
    ),
    UnitRootAnalysis(
        name="rates_covered_spread_trend",
        case=_RATES,
        purpose="The same spread with constant and trend: trend-stationary.",
        spec=_unit_root(
            (("CHF_COVERED_2Y", 1.0), ("CHF_SWAP_2Y", -1.0)),
            scale=Scale.LEVEL,
            deterministic=Deterministic.CONSTANT_TREND,
            lag=1,
        ),
        expected_error=None,
        expected_tau_below_level=0.01,
    ),
)


# ------------------------------------------------------------------ runners


def aligned_levels_by_case() -> dict[str, AlignedLevels]:
    """Align every canonical case on its own window."""
    return {
        name: aligned_levels_for_case(case)
        for name, case in CANONICAL_FACTOR_REGRESSION_CASES.items()
    }


def ols_analysis(name: str) -> OlsAnalysis:
    """
    Return the OLS analysis called ``name``.

    Raises
    ------
    KeyError
        No analysis has that name.
    """
    for analysis in CANONICAL_OLS_ANALYSES:
        if analysis.name == name:
            return analysis
    raise KeyError(name)


def run_ols_analysis(analysis: OlsAnalysis, levels: AlignedLevels) -> OlsResult:
    """``fit_ols`` of ``analysis.spec`` on ``levels``; facade errors propagate."""
    return fit_ols(levels, analysis.spec)


def run_wald_hypothesis(hypothesis: WaldHypothesis, result: OlsResult) -> WaldResult:
    """``wald_test`` of ``hypothesis`` on the fit of its OLS analysis."""
    return wald_test(result, hypothesis.restriction, hypothesis.rhs)


def run_unit_root_analysis(
    analysis: UnitRootAnalysis, levels: AlignedLevels
) -> UnitRootResult:
    """``unit_root_test`` of ``analysis.spec`` on ``levels``; errors propagate."""
    return unit_root_test(levels, analysis.spec)


def tau_below_level(result: UnitRootResult) -> float | None:
    """
    Smallest level in ``CRITICAL_LEVELS`` with ``statistic < critical value``.

    Critical values grow toward zero as the level grows, so the first level
    that the statistic passes is the smallest.
    """
    for level, critical in zip(
        result.critical_levels, result.critical_values.tolist(), strict=True
    ):
        if result.statistic < critical:
            return float(level)
    return None
