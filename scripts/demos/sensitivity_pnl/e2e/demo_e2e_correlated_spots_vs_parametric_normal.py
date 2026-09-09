# scripts/demos/sensitivity_pnl/e2e/demo_e2e_correlated_spots_vs_parametric_normal.py
"""
End-to-end validation demo:
correlated linear spot PnL through the sensitivity engine versus the
closed-form parametric Normal VaR / ES model.

This script is an executable mathematical specification and regression harness.

Mathematical model
------------------
Two long spot positions have:

    PnL_A = D_A * R_A
    PnL_B = D_B * R_B

where:

    D_i = V_i * P_i

and the returns are jointly Normal:

    R_A = sigma_A * Z_A

    R_B = sigma_B *
          (rho * Z_A + sqrt(1-rho^2) * Z_B)

with independent standard-normal Z_A and Z_B.

Because each position is exactly linear in its own return, the sensitivity
engine should reproduce the mathematical PnL exactly at the engine's numerical
representation.

The validation deliberately separates two concepts:

1. Mathematical reference
   -----------------------
   Float64 closed-form mathematical result:

       D_i * R_i

2. Engine representation
   ----------------------
   The actual numerical representation used by the production sensitivity
   engine. If the engine operates in float32, its result cannot be expected to
   be bit-for-bit identical to the raw float64 reference.

The demo therefore measures both:

    mathematical representation error
        = engine PnL - raw float64 mathematical PnL

and:

    engine transformation error
        = engine PnL
          - PnL calculated from the same numerical dtype as the engine

The second quantity is the true linearity test.

Validation layers
-----------------
1. Configuration validity.
2. Common-random-number shape / finiteness.
3. Reproducibility.
4. Float64 scenario-input contract.
5. Engine output dtype discovery.
6. Engine dtype consistency across rho.
7. Exact/near-exact linear PnL at engine precision.
8. Mathematical float64 representation error.
9. Exact portfolio aggregation.
10. Position-A bit-for-bit rho invariance.
11. Position-B marginal volatility invariance.
12. Realised correlation.
13. Portfolio sigma.
14. Parametric Normal VaR.
15. Parametric Normal ES.
16. Analytic rho-boundary behaviour.
17. Portfolio-risk monotonicity.
18. Diversification-benefit monotonicity.
19. Secondary confidence-level validation.

Important
---------
This demo does NOT weaken the engine-linearity test to accommodate float32.

Instead, it explicitly identifies the engine precision and compares the engine
with the mathematically correct operation performed at that precision.

If the engine is intended to be float64-only, set:

    REQUIRE_ENGINE_FLOAT64 = True

and the demo will fail loudly if the production engine violates that contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivity_pnl.conventions import QUOTE_BASE_NATURAL
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)

# ============================================================================
# DISPLAY / CONSTANTS
# ============================================================================

_WIDTH: Final[int] = 158
_RULE: Final[str] = "=" * _WIDTH
_SUBRULE: Final[str] = "-" * _WIDTH

POSITION_1_NAME: Final[str] = "SPOT_A"
POSITION_1_UID: Final[int] = 1
POSITION_2_NAME: Final[str] = "SPOT_B"
POSITION_2_UID: Final[int] = 2

FACTOR_1: Final[str] = "RETURN_A"
FACTOR_2: Final[str] = "RETURN_B"

VOLUME_1: Final[float] = 1_000.0
PRICE_1: Final[float] = 50.0
VOLUME_2: Final[float] = 500.0
PRICE_2: Final[float] = 200.0

DOLLAR_DELTA_1: Final[float] = VOLUME_1 * PRICE_1
DOLLAR_DELTA_2: Final[float] = VOLUME_2 * PRICE_2

SIGMA_1: Final[float] = 0.02
SIGMA_2: Final[float] = 0.03

PRIMARY_CONFIDENCE: Final[float] = 0.99
SECONDARY_CONFIDENCE: Final[float] = 0.95

RHO_GRID: Final[tuple[float, ...]] = (
    -1.0,
    -0.9,
    -0.5,
    0.0,
    0.5,
    0.9,
    1.0,
)

NUM_SCENARIOS: Final[int] = 1_000_000
SEED: Final[int] = 2_026_090_5

# Fixed textbook Normal quantiles.
Z_SCORE: Final[dict[float, float]] = {
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
    0.995: 2.5758293035489004,
}

# Numerical contract.
#
# Input scenarios are deliberately forced to float64.
#
# The engine itself is allowed to be float32 unless this is turned on.
REQUIRE_ENGINE_FLOAT64: Final[bool] = False

FLOAT64_INPUT_TOLERANCE: Final[float] = 0.0

# Engine-linearity tolerance is derived from the engine output dtype rather
# than arbitrarily relaxing the comparison.
ENGINE_ULP_MULTIPLIER: Final[float] = 32.0

# Statistical tolerances.
CORRELATION_ABS_TOLERANCE: Final[float] = 0.01
SIGMA_RELATIVE_TOLERANCE: Final[float] = 0.01
VAR_RELATIVE_TOLERANCE: Final[float] = 0.01
ES_RELATIVE_TOLERANCE: Final[float] = 0.01


# ============================================================================
# CONFIGURATION / RESULT TYPES
# ============================================================================


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Immutable configuration for one execution."""

    volume_a: float = VOLUME_1
    price_a: float = PRICE_1

    volume_b: float = VOLUME_2
    price_b: float = PRICE_2

    sigma_a: float = SIGMA_1
    sigma_b: float = SIGMA_2

    confidence_primary: float = PRIMARY_CONFIDENCE
    confidence_secondary: float = SECONDARY_CONFIDENCE

    rho_grid: tuple[float, ...] = RHO_GRID

    num_scenarios: int = NUM_SCENARIOS
    seed: int = SEED

    @property
    def dollar_delta_a(self) -> float:
        return self.volume_a * self.price_a

    @property
    def dollar_delta_b(self) -> float:
        return self.volume_b * self.price_b

    @property
    def standalone_std_a(self) -> float:
        return self.dollar_delta_a * self.sigma_a

    @property
    def standalone_std_b(self) -> float:
        return self.dollar_delta_b * self.sigma_b


@dataclass(frozen=True, slots=True)
class CommonRandomNumbers:
    """Single shared pair of independent standard-normal draws."""

    z_a: npt.NDArray[np.float64]
    z_b: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReturnScenarios:
    """Correlated return scenarios for one rho."""

    rho: float
    return_a: npt.NDArray[np.float64]
    return_b: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ParametricRisk:
    """Closed-form Normal risk for one rho / confidence level."""

    confidence_level: float
    rho: float

    std_a: float
    std_b: float
    std_portfolio: float

    var_a: float
    var_b: float
    var_portfolio: float

    es_a: float
    es_b: float
    es_portfolio: float


@dataclass(frozen=True, slots=True)
class EmpiricalRisk:
    """Empirical risk from simulated PnL."""

    confidence_level: float

    var_a: float
    var_b: float
    var_portfolio: float

    es_a: float
    es_b: float
    es_portfolio: float


@dataclass(frozen=True, slots=True)
class EngineRun:
    """Raw sensitivity-engine output plus numerical precision metadata."""

    pnl_a: npt.NDArray[np.float64]
    pnl_b: npt.NDArray[np.float64]
    portfolio_pnl: npt.NDArray[np.float64]

    input_dtype: str
    engine_pnl_dtype: str


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete result for one rho."""

    rho: float

    return_a: npt.NDArray[np.float64]
    return_b: npt.NDArray[np.float64]

    engine_pnl_a: npt.NDArray[np.float64]
    engine_pnl_b: npt.NDArray[np.float64]
    engine_portfolio_pnl: npt.NDArray[np.float64]

    engine_input_dtype: str
    engine_pnl_dtype: str

    empirical_primary: EmpiricalRisk
    empirical_secondary: EmpiricalRisk

    realised_correlation: float

    empirical_std_a: float
    empirical_std_b: float
    empirical_std_portfolio: float


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named validation outcome."""

    name: str
    passed: bool
    detail: str


# ============================================================================
# GENERIC NUMERICAL HELPERS
# ============================================================================


def standard_normal_pdf(z: float) -> float:
    """Standard Normal probability density."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def parametric_var_es(
    *,
    std_dev: float,
    confidence_level: float,
) -> tuple[float, float]:
    """Closed-form VaR / ES for zero-mean Normal PnL."""
    try:
        z = Z_SCORE[confidence_level]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported confidence level {confidence_level!r}; "
            f"available levels={sorted(Z_SCORE)}"
        ) from exc

    var = std_dev * z
    es = std_dev * standard_normal_pdf(z) / (1.0 - confidence_level)
    return var, es


def portfolio_std_dev(
    *,
    config: DemoConfig,
    rho: float,
) -> float:
    """Closed-form portfolio standard deviation."""
    variance = (
        config.dollar_delta_a**2 * config.sigma_a**2
        + config.dollar_delta_b**2 * config.sigma_b**2
        + 2.0
        * rho
        * config.dollar_delta_a
        * config.dollar_delta_b
        * config.sigma_a
        * config.sigma_b
    )

    if variance < -1.0e-10:
        raise ValueError(f"Portfolio variance is materially negative: {variance}")

    return math.sqrt(max(0.0, variance))


def relative_error(
    actual: float,
    expected: float,
) -> float:
    """Absolute relative error."""
    scale = abs(expected)

    if scale == 0.0:
        return abs(actual - expected)

    return abs(actual - expected) / scale


def max_abs_difference(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> float:
    """Maximum absolute elementwise difference."""
    return float(np.max(np.abs(left - right)))


def monotonic_non_decreasing(
    values: list[float],
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return True when values are non-decreasing within tolerance."""
    return all(right >= left - tolerance for left, right in pairwise(values))


def monotonic_non_increasing(
    values: list[float],
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return True when values are non-increasing within tolerance."""
    return all(right <= left + tolerance for left, right in pairwise(values))


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================


def validate_config(config: DemoConfig) -> None:
    """Validate all static configuration assumptions."""
    if config.num_scenarios < 10_000:
        raise ValueError(f"num_scenarios={config.num_scenarios:,} is too small")

    if config.sigma_a <= 0.0 or config.sigma_b <= 0.0:
        raise ValueError("Volatilities must be strictly positive")

    for confidence in (
        config.confidence_primary,
        config.confidence_secondary,
    ):
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"Invalid confidence level: {confidence}")

        if confidence not in Z_SCORE:
            raise ValueError(
                f"No configured Normal quantile for confidence={confidence}"
            )

    if len(config.rho_grid) < 3:
        raise ValueError("rho_grid must contain at least three values")

    if any(not -1.0 <= rho <= 1.0 for rho in config.rho_grid):
        raise ValueError("Every rho must lie in [-1, +1]")

    if len(set(config.rho_grid)) != len(config.rho_grid):
        raise ValueError("rho_grid contains duplicate values")

    if tuple(sorted(config.rho_grid)) != config.rho_grid:
        raise ValueError("rho_grid must be monotonically increasing")

    if -1.0 not in config.rho_grid or 1.0 not in config.rho_grid:
        raise ValueError("Validation requires rho=-1 and rho=+1 boundaries")


# ============================================================================
# BOOK CONSTRUCTION
# ============================================================================


def build_book(config: DemoConfig) -> SensitivityPnlCalculator:
    """
    Build the minimal linear sensitivity book.

    There is one first-order sensitivity per position and one factor per
    position. No cross-sensitivities exist.
    """
    axis = FactorAxis.from_keys([FACTOR_1, FACTOR_2])

    sensitivities = pl.DataFrame(
        [
            (
                POSITION_1_UID,
                FACTOR_1,
                config.dollar_delta_a,
                0.0,
            ),
            (
                POSITION_2_UID,
                FACTOR_2,
                config.dollar_delta_b,
                0.0,
            ),
        ],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )

    factor_meta = pl.DataFrame(
        {
            "factor_id": [FACTOR_1, FACTOR_2],
            "displacement_convention_id": [
                int(DisplacementConvention.RELATIVE),
                int(DisplacementConvention.RELATIVE),
            ],
            "quote_base": [
                QUOTE_BASE_NATURAL,
                QUOTE_BASE_NATURAL,
            ],
        }
    )

    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=factor_meta,
            factor_axis=axis,
            position_uids=[
                POSITION_1_UID,
                POSITION_2_UID,
            ],
        )
    )

    if calculator.book.num_positions != 2:
        raise AssertionError(
            f"Expected 2 positions, got {calculator.book.num_positions}"
        )

    if calculator.book.num_active_factors != 2:
        raise AssertionError(
            f"Expected 2 factors, got {calculator.book.num_active_factors}"
        )

    if calculator.book.nnz_cross != 0:
        raise AssertionError(
            f"Expected zero cross sensitivities, got {calculator.book.nnz_cross}"
        )

    return calculator


# ============================================================================
# COMMON RANDOM NUMBERS
# ============================================================================


def generate_common_random_numbers(
    config: DemoConfig,
) -> CommonRandomNumbers:
    """Generate the independent standard-normal sample exactly once."""
    rng = np.random.default_rng(config.seed)

    z_a = np.ascontiguousarray(
        rng.standard_normal(config.num_scenarios),
        dtype=np.float64,
    )

    z_b = np.ascontiguousarray(
        rng.standard_normal(config.num_scenarios),
        dtype=np.float64,
    )

    return CommonRandomNumbers(
        z_a=z_a,
        z_b=z_b,
    )


def validate_random_numbers(
    random_numbers: CommonRandomNumbers,
    config: DemoConfig,
) -> list[ValidationCheck]:
    """Validate shape, dtype and finiteness of the common draws."""
    expected_shape = (config.num_scenarios,)

    return [
        ValidationCheck(
            "common random-number shape",
            random_numbers.z_a.shape == expected_shape
            and random_numbers.z_b.shape == expected_shape,
            (f"shape_A={random_numbers.z_a.shape}, shape_B={random_numbers.z_b.shape}"),
        ),
        ValidationCheck(
            "common random-number dtype",
            random_numbers.z_a.dtype == np.dtype(np.float64)
            and random_numbers.z_b.dtype == np.dtype(np.float64),
            (f"dtype_A={random_numbers.z_a.dtype}, dtype_B={random_numbers.z_b.dtype}"),
        ),
        ValidationCheck(
            "common random-number finiteness",
            bool(
                np.isfinite(random_numbers.z_a).all()
                and np.isfinite(random_numbers.z_b).all()
            ),
            "all generated Z values are finite",
        ),
    ]


# ============================================================================
# RETURN CONSTRUCTION
# ============================================================================


def build_returns(
    *,
    config: DemoConfig,
    random_numbers: CommonRandomNumbers,
    rho: float,
) -> ReturnScenarios:
    """Construct the correlated bivariate Normal sample."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho={rho} outside [-1, +1]")

    orthogonal_weight = math.sqrt(max(0.0, 1.0 - rho * rho))

    return_a = config.sigma_a * random_numbers.z_a

    return_b = config.sigma_b * (
        rho * random_numbers.z_a + orthogonal_weight * random_numbers.z_b
    )

    return ReturnScenarios(
        rho=rho,
        return_a=np.ascontiguousarray(
            return_a,
            dtype=np.float64,
        ),
        return_b=np.ascontiguousarray(
            return_b,
            dtype=np.float64,
        ),
    )


# ============================================================================
# PARAMETRIC MODEL
# ============================================================================


def build_parametric_risk(
    *,
    config: DemoConfig,
    rho: float,
    confidence_level: float,
) -> ParametricRisk:
    """Build the closed-form Normal reference result."""
    std_a = config.standalone_std_a
    std_b = config.standalone_std_b
    std_portfolio = portfolio_std_dev(
        config=config,
        rho=rho,
    )

    var_a, es_a = parametric_var_es(
        std_dev=std_a,
        confidence_level=confidence_level,
    )

    var_b, es_b = parametric_var_es(
        std_dev=std_b,
        confidence_level=confidence_level,
    )

    var_portfolio, es_portfolio = parametric_var_es(
        std_dev=std_portfolio,
        confidence_level=confidence_level,
    )

    return ParametricRisk(
        confidence_level=confidence_level,
        rho=rho,
        std_a=std_a,
        std_b=std_b,
        std_portfolio=std_portfolio,
        var_a=var_a,
        var_b=var_b,
        var_portfolio=var_portfolio,
        es_a=es_a,
        es_b=es_b,
        es_portfolio=es_portfolio,
    )


# ============================================================================
# ENGINE PRECISION / PNL
# ============================================================================


def _engine_precision_reference(
    *,
    delta: float,
    returns: npt.NDArray[np.float64],
    engine_dtype: np.dtype,
) -> npt.NDArray[np.float64]:
    """
    Calculate the linear PnL using the same floating-point dtype as the engine.

    This is the correct reference for the engine-linearity test.

    The final conversion to float64 is only for comparison/reporting.
    """
    typed_returns = np.asarray(
        returns,
        dtype=engine_dtype,
    )

    typed_delta = np.asarray(
        delta,
        dtype=engine_dtype,
    )

    expected = np.multiply(
        typed_delta,
        typed_returns,
        dtype=engine_dtype,
    )

    return np.ascontiguousarray(
        expected,
        dtype=np.float64,
    )


def engine_linearity_tolerance(
    *,
    engine_dtype: np.dtype,
    expected: npt.NDArray[np.float64],
) -> float:
    """
    Derive an absolute tolerance from machine precision and result magnitude.

    This is not a statistical tolerance. It exists only for floating-point
    arithmetic at the engine's numerical representation.
    """
    if not np.issubdtype(engine_dtype, np.floating):
        raise TypeError(f"Engine dtype must be floating-point, got {engine_dtype}")

    eps = np.finfo(engine_dtype).eps
    scale = max(
        1.0,
        float(np.max(np.abs(expected))),
    )

    return ENGINE_ULP_MULTIPLIER * eps * scale


def run_sensitivity_engine(
    *,
    calculator: SensitivityPnlCalculator,
    returns: ReturnScenarios,
) -> EngineRun:
    """
    Run the sensitivity engine using a deliberately float64 scenario boundary.

    The dtype of engine_result.pnl is captured before conversion to float64.
    """
    displacement = np.ascontiguousarray(
        np.vstack(
            [
                returns.return_a,
                returns.return_b,
            ]
        ),
        dtype=np.float64,
    )

    if displacement.dtype != np.dtype(np.float64):
        raise AssertionError(
            f"Scenario boundary unexpectedly changed dtype: {displacement.dtype}"
        )

    shocks = calculator.axis.shocks(displacement)

    engine_result = calculator.run(shocks)

    raw_pnl = np.asarray(engine_result.pnl)

    if raw_pnl.ndim != 2:
        raise AssertionError(f"Expected 2-D PnL result, got ndim={raw_pnl.ndim}")

    if raw_pnl.shape[0] != 2:
        raise AssertionError(f"Expected two PnL rows, got shape={raw_pnl.shape}")

    if not np.issubdtype(raw_pnl.dtype, np.floating):
        raise AssertionError(
            f"Expected floating-point engine PnL, got dtype={raw_pnl.dtype}"
        )

    if not np.isfinite(raw_pnl).all():
        raise AssertionError("Sensitivity engine produced non-finite PnL")

    return EngineRun(
        pnl_a=np.ascontiguousarray(
            raw_pnl[0],
            dtype=np.float64,
        ),
        pnl_b=np.ascontiguousarray(
            raw_pnl[1],
            dtype=np.float64,
        ),
        portfolio_pnl=np.ascontiguousarray(
            raw_pnl.sum(axis=0),
            dtype=np.float64,
        ),
        input_dtype=str(displacement.dtype),
        engine_pnl_dtype=str(raw_pnl.dtype),
    )


# ============================================================================
# EMPIRICAL RISK
# ============================================================================


def compute_empirical_risk(
    *,
    pnl_a: npt.NDArray[np.float64],
    pnl_b: npt.NDArray[np.float64],
    portfolio_pnl: npt.NDArray[np.float64],
    confidence_level: float,
) -> EmpiricalRisk:
    """Compute empirical VaR / ES."""
    stacked = np.ascontiguousarray(
        np.vstack(
            [
                pnl_a,
                pnl_b,
                portfolio_pnl,
            ]
        ),
        dtype=np.float64,
    )

    result = compute_var_es(
        stacked,
        confidence_level=confidence_level,
    )

    return EmpiricalRisk(
        confidence_level=confidence_level,
        var_a=float(-result.var[0]),
        var_b=float(-result.var[1]),
        var_portfolio=float(-result.var[2]),
        es_a=float(-result.es[0]),
        es_b=float(-result.es[1]),
        es_portfolio=float(-result.es[2]),
    )


# ============================================================================
# SINGLE-RHO SIMULATION
# ============================================================================


def simulate(
    *,
    config: DemoConfig,
    calculator: SensitivityPnlCalculator,
    random_numbers: CommonRandomNumbers,
    rho: float,
) -> SimulationResult:
    """Build scenarios, run the engine, calculate all metrics."""
    returns = build_returns(
        config=config,
        random_numbers=random_numbers,
        rho=rho,
    )

    engine = run_sensitivity_engine(
        calculator=calculator,
        returns=returns,
    )

    empirical_primary = compute_empirical_risk(
        pnl_a=engine.pnl_a,
        pnl_b=engine.pnl_b,
        portfolio_pnl=engine.portfolio_pnl,
        confidence_level=config.confidence_primary,
    )

    empirical_secondary = compute_empirical_risk(
        pnl_a=engine.pnl_a,
        pnl_b=engine.pnl_b,
        portfolio_pnl=engine.portfolio_pnl,
        confidence_level=config.confidence_secondary,
    )

    realised_correlation = float(
        np.corrcoef(
            returns.return_a,
            returns.return_b,
        )[0, 1]
    )

    return SimulationResult(
        rho=rho,
        return_a=returns.return_a,
        return_b=returns.return_b,
        engine_pnl_a=engine.pnl_a,
        engine_pnl_b=engine.pnl_b,
        engine_portfolio_pnl=engine.portfolio_pnl,
        engine_input_dtype=engine.input_dtype,
        engine_pnl_dtype=engine.engine_pnl_dtype,
        empirical_primary=empirical_primary,
        empirical_secondary=empirical_secondary,
        realised_correlation=realised_correlation,
        empirical_std_a=float(engine.pnl_a.std()),
        empirical_std_b=float(engine.pnl_b.std()),
        empirical_std_portfolio=float(engine.portfolio_pnl.std()),
    )


# ============================================================================
# PRECISION / ENGINE CONTRACT VALIDATION
# ============================================================================


def validate_float64_scenario_boundary(
    results: list[SimulationResult],
) -> ValidationCheck:
    """The demo must explicitly feed float64 scenarios into the engine."""
    dtypes = {result.engine_input_dtype for result in results}

    return ValidationCheck(
        "float64 scenario-input contract",
        dtypes == {"float64"},
        f"observed_input_dtypes={sorted(dtypes)}",
    )


def validate_engine_pnl_dtype_consistency(
    results: list[SimulationResult],
) -> ValidationCheck:
    """The engine must use one stable PnL dtype across the rho sweep."""
    dtypes = {result.engine_pnl_dtype for result in results}

    return ValidationCheck(
        "engine PnL dtype consistency",
        len(dtypes) == 1,
        f"observed_engine_pnl_dtypes={sorted(dtypes)}",
    )


def validate_engine_float64_requirement(
    results: list[SimulationResult],
) -> ValidationCheck:
    """Optional hard contract when the production engine is required to be 64-bit."""
    observed = {result.engine_pnl_dtype for result in results}

    passed = not REQUIRE_ENGINE_FLOAT64 or observed == {"float64"}

    if REQUIRE_ENGINE_FLOAT64:
        detail = f"required=float64, observed={sorted(observed)}"
    else:
        detail = f"requirement_disabled, observed={sorted(observed)}"

    return ValidationCheck(
        "engine float64 precision requirement",
        passed,
        detail,
    )


def validate_linear_pnl_at_engine_precision(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """
    Validate exact linearity against the numerical representation actually
    produced by the engine.

    This is the primary engine correctness test.
    """
    checks: list[ValidationCheck] = []

    for result in results:
        engine_dtype = np.dtype(result.engine_pnl_dtype)

        expected_a = _engine_precision_reference(
            delta=config.dollar_delta_a,
            returns=result.return_a,
            engine_dtype=engine_dtype,
        )

        expected_b = _engine_precision_reference(
            delta=config.dollar_delta_b,
            returns=result.return_b,
            engine_dtype=engine_dtype,
        )

        error_a = max_abs_difference(
            result.engine_pnl_a,
            expected_a,
        )

        error_b = max_abs_difference(
            result.engine_pnl_b,
            expected_b,
        )

        tolerance_a = engine_linearity_tolerance(
            engine_dtype=engine_dtype,
            expected=expected_a,
        )

        tolerance_b = engine_linearity_tolerance(
            engine_dtype=engine_dtype,
            expected=expected_b,
        )

        checks.append(
            ValidationCheck(
                f"engine linear PnL A rho={result.rho:+.1f}",
                error_a <= tolerance_a,
                (
                    f"dtype={engine_dtype}, "
                    f"max_abs_error={error_a:.3e}, "
                    f"tolerance={tolerance_a:.3e}"
                ),
            )
        )

        checks.append(
            ValidationCheck(
                f"engine linear PnL B rho={result.rho:+.1f}",
                error_b <= tolerance_b,
                (
                    f"dtype={engine_dtype}, "
                    f"max_abs_error={error_b:.3e}, "
                    f"tolerance={tolerance_b:.3e}"
                ),
            )
        )

    return checks


def validate_float64_mathematical_representation_error(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """
    Measure the gap between the raw float64 mathematical result and the
    production engine result.

    This is intentionally NOT a pass/fail correctness test when the engine
    uses float32. It is a numerical representation diagnostic.
    """
    checks: list[ValidationCheck] = []

    for result in results:
        raw_expected_a = config.dollar_delta_a * result.return_a
        raw_expected_b = config.dollar_delta_b * result.return_b

        error_a = max_abs_difference(
            result.engine_pnl_a,
            raw_expected_a,
        )

        error_b = max_abs_difference(
            result.engine_pnl_b,
            raw_expected_b,
        )

        checks.append(
            ValidationCheck(
                f"float64-vs-engine representation A rho={result.rho:+.1f}",
                True,
                (
                    f"engine_dtype={result.engine_pnl_dtype}, "
                    f"max_abs_difference={error_a:.3e}"
                ),
            )
        )

        checks.append(
            ValidationCheck(
                f"float64-vs-engine representation B rho={result.rho:+.1f}",
                True,
                (
                    f"engine_dtype={result.engine_pnl_dtype}, "
                    f"max_abs_difference={error_b:.3e}"
                ),
            )
        )

    return checks


def validate_portfolio_aggregation(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Portfolio PnL must equal the sum of the two engine position PnLs."""
    checks: list[ValidationCheck] = []

    for result in results:
        expected = result.engine_pnl_a + result.engine_pnl_b

        error = max_abs_difference(
            result.engine_portfolio_pnl,
            expected,
        )

        engine_dtype = np.dtype(result.engine_pnl_dtype)

        tolerance = engine_linearity_tolerance(
            engine_dtype=engine_dtype,
            expected=expected,
        )

        checks.append(
            ValidationCheck(
                f"portfolio aggregation rho={result.rho:+.1f}",
                error <= tolerance,
                (
                    f"dtype={engine_dtype}, "
                    f"max_abs_error={error:.3e}, "
                    f"tolerance={tolerance:.3e}"
                ),
            )
        )

    return checks


# ============================================================================
# STRUCTURAL INVARIANTS
# ============================================================================


def validate_position_a_rho_invariance(
    results: list[SimulationResult],
) -> ValidationCheck:
    """Position A must be bit-for-bit identical across rho."""
    reference = results[0].engine_pnl_a

    identical = all(
        np.array_equal(
            result.engine_pnl_a,
            reference,
        )
        for result in results[1:]
    )

    return ValidationCheck(
        "position A bit-for-bit rho invariance",
        identical,
        f"checked {len(results)} correlation scenarios",
    )


def validate_position_b_marginal_invariance(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> ValidationCheck:
    """Position B's marginal volatility is independent of rho."""
    expected = config.standalone_std_b

    max_error = max(
        relative_error(
            result.empirical_std_b,
            expected,
        )
        for result in results
    )

    return ValidationCheck(
        "position B marginal volatility invariance",
        max_error <= SIGMA_RELATIVE_TOLERANCE,
        (
            f"expected={expected:,.6f}, "
            f"max_relative_error={max_error:.3%}, "
            f"tolerance={SIGMA_RELATIVE_TOLERANCE:.3%}"
        ),
    )


def validate_realised_correlations(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Validate realised sample correlation against requested rho."""
    return [
        ValidationCheck(
            f"realised correlation rho={result.rho:+.1f}",
            abs(result.realised_correlation - result.rho) <= CORRELATION_ABS_TOLERANCE,
            (
                f"target={result.rho:+.6f}, "
                f"realised={result.realised_correlation:+.6f}, "
                f"abs_error="
                f"{abs(result.realised_correlation - result.rho):.6f}, "
                f"tolerance={CORRELATION_ABS_TOLERANCE:.6f}"
            ),
        )
        for result in results
    ]


def validate_portfolio_sigma(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Compare empirical portfolio sigma with theory."""
    checks: list[ValidationCheck] = []

    for result in results:
        expected = portfolio_std_dev(
            config=config,
            rho=result.rho,
        )

        error = relative_error(
            result.empirical_std_portfolio,
            expected,
        )

        checks.append(
            ValidationCheck(
                f"portfolio volatility rho={result.rho:+.1f}",
                error <= SIGMA_RELATIVE_TOLERANCE,
                (
                    f"empirical={result.empirical_std_portfolio:,.6f}, "
                    f"theoretical={expected:,.6f}, "
                    f"relative_error={error:.3%}, "
                    f"tolerance={SIGMA_RELATIVE_TOLERANCE:.3%}"
                ),
            )
        )

    return checks


# ============================================================================
# RISK METRIC VALIDATION
# ============================================================================


def validate_primary_risk_metrics(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Validate 99% empirical VaR / ES."""
    checks: list[ValidationCheck] = []

    for result in results:
        empirical = result.empirical_primary

        theoretical = build_parametric_risk(
            config=config,
            rho=result.rho,
            confidence_level=config.confidence_primary,
        )

        metrics = (
            ("VaR A", empirical.var_a, theoretical.var_a),
            ("VaR B", empirical.var_b, theoretical.var_b),
            (
                "VaR portfolio",
                empirical.var_portfolio,
                theoretical.var_portfolio,
            ),
            ("ES A", empirical.es_a, theoretical.es_a),
            ("ES B", empirical.es_b, theoretical.es_b),
            (
                "ES portfolio",
                empirical.es_portfolio,
                theoretical.es_portfolio,
            ),
        )

        for label, actual, expected in metrics:
            error = relative_error(
                actual,
                expected,
            )

            tolerance = (
                VAR_RELATIVE_TOLERANCE
                if label.startswith("VaR")
                else ES_RELATIVE_TOLERANCE
            )

            checks.append(
                ValidationCheck(
                    f"{config.confidence_primary:.0%} {label} rho={result.rho:+.1f}",
                    error <= tolerance,
                    (
                        f"empirical={actual:,.6f}, "
                        f"theoretical={expected:,.6f}, "
                        f"relative_error={error:.3%}, "
                        f"tolerance={tolerance:.3%}"
                    ),
                )
            )

    return checks


def validate_secondary_risk_metrics(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Validate 95% portfolio VaR / ES."""
    checks: list[ValidationCheck] = []

    for result in results:
        empirical = result.empirical_secondary

        theoretical = build_parametric_risk(
            config=config,
            rho=result.rho,
            confidence_level=config.confidence_secondary,
        )

        metrics = (
            (
                "VaR portfolio",
                empirical.var_portfolio,
                theoretical.var_portfolio,
            ),
            (
                "ES portfolio",
                empirical.es_portfolio,
                theoretical.es_portfolio,
            ),
        )

        for label, actual, expected in metrics:
            error = relative_error(
                actual,
                expected,
            )

            tolerance = (
                VAR_RELATIVE_TOLERANCE
                if label.startswith("VaR")
                else ES_RELATIVE_TOLERANCE
            )

            checks.append(
                ValidationCheck(
                    f"{config.confidence_secondary:.0%} {label} rho={result.rho:+.1f}",
                    error <= tolerance,
                    (
                        f"empirical={actual:,.6f}, "
                        f"theoretical={expected:,.6f}, "
                        f"relative_error={error:.3%}, "
                        f"tolerance={tolerance:.3%}"
                    ),
                )
            )

    return checks


# ============================================================================
# ANALYTICAL / BOUNDARY VALIDATION
# ============================================================================


def validate_rho_boundary_std(
    *,
    config: DemoConfig,
) -> list[ValidationCheck]:
    """Validate the exact analytical boundary cases."""
    expected = {
        -1.0: abs(config.standalone_std_b - config.standalone_std_a),
        0.0: math.sqrt(config.standalone_std_a**2 + config.standalone_std_b**2),
        1.0: (config.standalone_std_a + config.standalone_std_b),
    }

    checks: list[ValidationCheck] = []

    for rho, expected_std in expected.items():
        actual = portfolio_std_dev(
            config=config,
            rho=rho,
        )

        error = abs(actual - expected_std)

        checks.append(
            ValidationCheck(
                f"analytic sigma boundary rho={rho:+.1f}",
                error <= 1.0e-10,
                (
                    f"actual={actual:.12f}, "
                    f"expected={expected_std:.12f}, "
                    f"abs_error={error:.3e}"
                ),
            )
        )

    return checks


def validate_risk_monotonicity(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> list[ValidationCheck]:
    """Validate theoretical risk and diversification ordering."""
    theoretical_stds = [
        portfolio_std_dev(
            config=config,
            rho=result.rho,
        )
        for result in results
    ]

    theoretical_vars = [
        parametric_var_es(
            std_dev=std,
            confidence_level=config.confidence_primary,
        )[0]
        for std in theoretical_stds
    ]

    standalone_var_sum = sum(
        parametric_var_es(
            std_dev=std,
            confidence_level=config.confidence_primary,
        )[0]
        for std in (
            config.standalone_std_a,
            config.standalone_std_b,
        )
    )

    diversification_benefits = [standalone_var_sum - var for var in theoretical_vars]

    return [
        ValidationCheck(
            "theoretical portfolio volatility monotone in rho",
            monotonic_non_decreasing(theoretical_stds),
            "closed-form sigma(rho) is non-decreasing",
        ),
        ValidationCheck(
            "theoretical portfolio VaR monotone in rho",
            monotonic_non_decreasing(theoretical_vars),
            "closed-form VaR(rho) is non-decreasing",
        ),
        ValidationCheck(
            "diversification benefit monotone decreasing in rho",
            monotonic_non_increasing(diversification_benefits),
            "diversification benefit declines as rho rises",
        ),
    ]


# ============================================================================
# REPRODUCIBILITY
# ============================================================================


def validate_reproducibility(
    *,
    config: DemoConfig,
    calculator: SensitivityPnlCalculator,
    random_numbers: CommonRandomNumbers,
) -> ValidationCheck:
    """Identical seed/configuration must reproduce identical outputs."""
    regenerated = generate_common_random_numbers(config)

    random_numbers_identical = np.array_equal(
        random_numbers.z_a,
        regenerated.z_a,
    ) and np.array_equal(
        random_numbers.z_b,
        regenerated.z_b,
    )

    if not random_numbers_identical:
        return ValidationCheck(
            "RNG reproducibility",
            False,
            "common random numbers changed under identical configuration",
        )

    first = simulate(
        config=config,
        calculator=calculator,
        random_numbers=random_numbers,
        rho=0.5,
    )

    second = simulate(
        config=config,
        calculator=calculator,
        random_numbers=regenerated,
        rho=0.5,
    )

    identical = (
        np.array_equal(
            first.return_a,
            second.return_a,
        )
        and np.array_equal(
            first.return_b,
            second.return_b,
        )
        and np.array_equal(
            first.engine_pnl_a,
            second.engine_pnl_a,
        )
        and np.array_equal(
            first.engine_pnl_b,
            second.engine_pnl_b,
        )
        and np.array_equal(
            first.engine_portfolio_pnl,
            second.engine_portfolio_pnl,
        )
    )

    return ValidationCheck(
        "RNG reproducibility",
        identical,
        "same seed/configuration reproduces returns and engine PnL exactly",
    )


# ============================================================================
# REPORTING
# ============================================================================


def print_header(config: DemoConfig) -> None:
    print(_RULE)
    print(
        f"{'CORRELATED SPOTS — E2E SENSITIVITY / PARAMETRIC NORMAL VALIDATION':^{_WIDTH}}"
    )
    print(_RULE)

    print("\nMODEL CONFIGURATION")
    print(_SUBRULE)
    print(
        f"Position A                : {POSITION_1_NAME:<10} "
        f"long {config.volume_a:,.0f} @ ${config.price_a:,.2f}"
    )
    print(
        f"Position B                : {POSITION_2_NAME:<10} "
        f"long {config.volume_b:,.0f} @ ${config.price_b:,.2f}"
    )
    print(f"Dollar delta A            : {config.dollar_delta_a:,.0f}")
    print(f"Dollar delta B            : {config.dollar_delta_b:,.0f}")
    print(f"Standalone sigma A        : {config.standalone_std_a:,.6f}")
    print(f"Standalone sigma B        : {config.standalone_std_b:,.6f}")
    print(f"Scenarios                 : {config.num_scenarios:,}")
    print(f"Seed                      : {config.seed}")
    print(
        f"Confidence levels         : "
        f"{config.confidence_secondary:.0%}, "
        f"{config.confidence_primary:.0%}"
    )
    print(
        f"Correlation grid          : "
        f"[{', '.join(f'{rho:+.1f}' for rho in config.rho_grid)}]"
    )

    print("\nMATHEMATICAL CONTRACT")
    print(_SUBRULE)
    print("PnL_A       = D_A * R_A")
    print("PnL_B       = D_B * R_B")
    print("PnL_port    = PnL_A + PnL_B")
    print("R_A, R_B    jointly Normal")
    print("Corr(R_A,R_B) = rho")
    print("sigma_port² = D_A² sigma_A² + D_B² sigma_B² + 2 rho D_A D_B sigma_A sigma_B")

    print("\nPRECISION CONTRACT")
    print(_SUBRULE)
    print("Scenario generation        : float64")
    print("Scenario input boundary    : float64")
    print(
        "Engine output requirement  : "
        f"{'float64' if REQUIRE_ENGINE_FLOAT64 else 'engine-native dtype'}"
    )
    print("Linearity comparison       : engine-native precision reference")
    print(
        "Float64 discrepancy        : reported separately as representation diagnostic"
    )

    print("\nBOOK CONTRACT")
    print(_SUBRULE)
    print("Positions                  : 2")
    print("Active factors             : 2")
    print("Cross sensitivities        : 0")
    print("Displacement convention    : RELATIVE")
    print("Correlation source         : scenario construction, not sensitivity book")


def print_precision_summary(
    results: list[SimulationResult],
) -> None:
    input_dtypes = sorted({result.engine_input_dtype for result in results})

    pnl_dtypes = sorted({result.engine_pnl_dtype for result in results})

    print("\nPRECISION OBSERVATION")
    print(_SUBRULE)
    print(f"Scenario input dtype      : {', '.join(input_dtypes)}")
    print(f"Engine PnL dtype          : {', '.join(pnl_dtypes)}")

    if pnl_dtypes == ["float64"]:
        print("Interpretation             : engine preserves float64 precision")
    else:
        print(
            "Interpretation             : "
            "engine uses reduced precision; "
            "linearity is validated at that precision"
        )


def print_worked_case(
    *,
    config: DemoConfig,
    result: SimulationResult,
) -> None:
    theoretical = build_parametric_risk(
        config=config,
        rho=result.rho,
        confidence_level=config.confidence_primary,
    )

    empirical = result.empirical_primary

    cross_term = (
        2.0
        * result.rho
        * config.dollar_delta_a
        * config.dollar_delta_b
        * config.sigma_a
        * config.sigma_b
    )

    standalone_var_sum = theoretical.var_a + theoretical.var_b

    diversification = standalone_var_sum - theoretical.var_portfolio

    print("\n" + _RULE)
    print(f"{'WORKED CASE — rho = +0.5':^{_WIDTH}}")
    print(_RULE)

    print("\nPORTFOLIO VARIANCE DECOMPOSITION")
    print(_SUBRULE)
    print(f"D_A² sigma_A²             : {config.standalone_std_a**2:,.6f}")
    print(f"D_B² sigma_B²             : {config.standalone_std_b**2:,.6f}")
    print(f"Correlation cross term    : {cross_term:,.6f}")
    print(f"Portfolio variance        : {theoretical.std_portfolio**2:,.6f}")
    print(f"Portfolio sigma           : {theoretical.std_portfolio:,.6f}")

    print("\nPRECISION / ENGINE")
    print(_SUBRULE)
    print(f"Scenario input dtype      : {result.engine_input_dtype}")
    print(f"Engine PnL dtype          : {result.engine_pnl_dtype}")

    print("\nCLOSED-FORM NORMAL RISK")
    print(_SUBRULE)

    z = Z_SCORE[config.confidence_primary]

    print(f"z_{config.confidence_primary:.0%}                      : {z:.12f}")
    print(f"phi(z)                    : {standard_normal_pdf(z):.12f}")
    print(f"Parametric VaR            : {theoretical.var_portfolio:,.6f}")
    print(f"Empirical VaR             : {empirical.var_portfolio:,.6f}")
    print(
        f"VaR relative error        : "
        f"{relative_error(empirical.var_portfolio, theoretical.var_portfolio):.3%}"
    )
    print(f"Parametric ES             : {theoretical.es_portfolio:,.6f}")
    print(f"Empirical ES              : {empirical.es_portfolio:,.6f}")
    print(
        f"ES relative error         : "
        f"{relative_error(empirical.es_portfolio, theoretical.es_portfolio):.3%}"
    )

    print("\nDIVERSIFICATION")
    print(_SUBRULE)
    print(f"Standalone VaR A + B     : {standalone_var_sum:,.6f}")
    print(f"Portfolio VaR             : {theoretical.var_portfolio:,.6f}")
    print(f"Diversification benefit  : {diversification:,.6f}")


def print_sweep(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> None:
    print("\n" + _RULE)
    print(
        f"{f'CORRELATION SWEEP — {config.confidence_primary:.0%} CONFIDENCE':^{_WIDTH}}"
    )
    print(_RULE)

    print(
        f"{'rho':>6} | "
        f"{'corr':>8} | "
        f"{'std emp':>10} {'std par':>10} | "
        f"{'VaR emp':>10} {'VaR par':>10} {'err':>7} | "
        f"{'ES emp':>10} {'ES par':>10} {'err':>7} | "
        f"{'div.':>10}"
    )
    print(_SUBRULE)

    naive_var_sum = sum(
        parametric_var_es(
            std_dev=std,
            confidence_level=config.confidence_primary,
        )[0]
        for std in (
            config.standalone_std_a,
            config.standalone_std_b,
        )
    )

    for result in results:
        theoretical = build_parametric_risk(
            config=config,
            rho=result.rho,
            confidence_level=config.confidence_primary,
        )

        empirical = result.empirical_primary

        var_error = relative_error(
            empirical.var_portfolio,
            theoretical.var_portfolio,
        )

        es_error = relative_error(
            empirical.es_portfolio,
            theoretical.es_portfolio,
        )

        diversification = naive_var_sum - theoretical.var_portfolio

        print(
            f"{result.rho:>+6.1f} | "
            f"{result.realised_correlation:>+8.4f} | "
            f"{result.empirical_std_portfolio:>10,.0f} "
            f"{theoretical.std_portfolio:>10,.0f} | "
            f"{empirical.var_portfolio:>10,.0f} "
            f"{theoretical.var_portfolio:>10,.0f} "
            f"{var_error:>6.2%} | "
            f"{empirical.es_portfolio:>10,.0f} "
            f"{theoretical.es_portfolio:>10,.0f} "
            f"{es_error:>6.2%} | "
            f"{diversification:>10,.0f}"
        )

    print(_SUBRULE)


def print_standalone_sweep(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> None:
    print("\nSTANDALONE RISK — SHOULD NOT DEPEND ON CORRELATION")
    print(_SUBRULE)

    theoretical = build_parametric_risk(
        config=config,
        rho=0.0,
        confidence_level=config.confidence_primary,
    )

    print(
        f"{'rho':>6} | "
        f"{'VaR A emp':>11} {'VaR A par':>11} | "
        f"{'VaR B emp':>11} {'VaR B par':>11} | "
        f"{'sigma B emp':>11}"
    )
    print(_SUBRULE)

    for result in results:
        empirical = result.empirical_primary

        print(
            f"{result.rho:>+6.1f} | "
            f"{empirical.var_a:>11,.0f} "
            f"{theoretical.var_a:>11,.0f} | "
            f"{empirical.var_b:>11,.0f} "
            f"{theoretical.var_b:>11,.0f} | "
            f"{result.empirical_std_b:>11,.0f}"
        )

    print(_SUBRULE)


def print_secondary_confidence(
    *,
    config: DemoConfig,
    results: list[SimulationResult],
) -> None:
    print("\n" + _RULE)
    print(
        f"{f'SECONDARY CHECK — {config.confidence_secondary:.0%} CONFIDENCE':^{_WIDTH}}"
    )
    print(_RULE)

    print(
        f"{'rho':>6} | "
        f"{'VaR emp':>10} {'VaR par':>10} {'err':>7} | "
        f"{'ES emp':>10} {'ES par':>10} {'err':>7}"
    )
    print(_SUBRULE)

    for result in results:
        empirical = result.empirical_secondary

        theoretical = build_parametric_risk(
            config=config,
            rho=result.rho,
            confidence_level=config.confidence_secondary,
        )

        var_error = relative_error(
            empirical.var_portfolio,
            theoretical.var_portfolio,
        )

        es_error = relative_error(
            empirical.es_portfolio,
            theoretical.es_portfolio,
        )

        print(
            f"{result.rho:>+6.1f} | "
            f"{empirical.var_portfolio:>10,.0f} "
            f"{theoretical.var_portfolio:>10,.0f} "
            f"{var_error:>6.2%} | "
            f"{empirical.es_portfolio:>10,.0f} "
            f"{theoretical.es_portfolio:>10,.0f} "
            f"{es_error:>6.2%}"
        )

    print(_SUBRULE)


def print_checks(
    checks: list[ValidationCheck],
) -> None:
    print("\n" + _RULE)
    print(f"{'VALIDATION CHECKS':^{_WIDTH}}")
    print(_RULE)

    for check in checks:
        status = "PASS" if check.passed else "FAIL"

        print(f"{status:>4} | {check.name:<68} | {check.detail}")

    print(_SUBRULE)


def print_final_status(
    checks: list[ValidationCheck],
) -> None:
    passed = sum(check.passed for check in checks)

    total = len(checks)

    print("\n" + _RULE)

    if passed == total:
        print(f"{f'ALL CHECKS PASSED — {passed}/{total}':^{_WIDTH}}")
    else:
        print(f"{f'VALIDATION FAILED — {passed}/{total} CHECKS PASSED':^{_WIDTH}}")

    print(_RULE)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================


def run_demo() -> None:
    """Execute the complete end-to-end validation."""
    config = DemoConfig()

    # ------------------------------------------------------------------------
    # Static configuration / book
    # ------------------------------------------------------------------------
    validate_config(config)

    calculator = build_book(config)

    print_header(config)

    # ------------------------------------------------------------------------
    # Common random numbers
    # ------------------------------------------------------------------------
    random_numbers = generate_common_random_numbers(config)

    checks: list[ValidationCheck] = []

    checks.extend(
        validate_random_numbers(
            random_numbers,
            config,
        )
    )

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------
    checks.append(
        validate_reproducibility(
            config=config,
            calculator=calculator,
            random_numbers=random_numbers,
        )
    )

    # ------------------------------------------------------------------------
    # Full correlation sweep
    # ------------------------------------------------------------------------
    results = [
        simulate(
            config=config,
            calculator=calculator,
            random_numbers=random_numbers,
            rho=rho,
        )
        for rho in config.rho_grid
    ]

    # ------------------------------------------------------------------------
    # Precision contract
    # ------------------------------------------------------------------------
    checks.append(validate_float64_scenario_boundary(results))

    checks.append(validate_engine_pnl_dtype_consistency(results))

    checks.append(validate_engine_float64_requirement(results))

    # ------------------------------------------------------------------------
    # Engine linearity / representation
    # ------------------------------------------------------------------------
    checks.extend(
        validate_linear_pnl_at_engine_precision(
            config=config,
            results=results,
        )
    )

    checks.extend(
        validate_float64_mathematical_representation_error(
            config=config,
            results=results,
        )
    )

    checks.extend(
        validate_portfolio_aggregation(
            config=config,
            results=results,
        )
    )

    # ------------------------------------------------------------------------
    # Structural invariants
    # ------------------------------------------------------------------------
    checks.append(validate_position_a_rho_invariance(results))

    checks.append(
        validate_position_b_marginal_invariance(
            config=config,
            results=results,
        )
    )

    checks.extend(
        validate_realised_correlations(
            config=config,
            results=results,
        )
    )

    checks.extend(
        validate_portfolio_sigma(
            config=config,
            results=results,
        )
    )

    # ------------------------------------------------------------------------
    # Risk metric validation
    # ------------------------------------------------------------------------
    checks.extend(
        validate_primary_risk_metrics(
            config=config,
            results=results,
        )
    )

    checks.extend(
        validate_secondary_risk_metrics(
            config=config,
            results=results,
        )
    )

    # ------------------------------------------------------------------------
    # Analytical structure
    # ------------------------------------------------------------------------
    checks.extend(validate_rho_boundary_std(config=config))

    checks.extend(
        validate_risk_monotonicity(
            config=config,
            results=results,
        )
    )

    # ------------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------------
    print(
        f"\nGenerated {config.num_scenarios:,} common scenarios once "
        f"and re-used them across {len(config.rho_grid)} correlations."
    )

    print(
        "The sensitivity book contains no correlation information: "
        "dependence is entirely in scenario construction."
    )

    print_precision_summary(results)

    worked = next(result for result in results if result.rho == 0.5)

    print_worked_case(
        config=config,
        result=worked,
    )

    print_sweep(
        config=config,
        results=results,
    )

    print_standalone_sweep(
        config=config,
        results=results,
    )

    print_secondary_confidence(
        config=config,
        results=results,
    )

    print_checks(checks)

    print_final_status(checks)

    # ------------------------------------------------------------------------
    # Hard failure after complete diagnostics.
    # ------------------------------------------------------------------------
    failed = [check for check in checks if not check.passed]

    if failed:
        failure_summary = "; ".join(f"{check.name}: {check.detail}" for check in failed)

        raise AssertionError(
            f"End-to-end validation failed with "
            f"{len(failed)} failed checks: "
            f"{failure_summary}"
        )


if __name__ == "__main__":
    run_demo()
